"""InternVL3.5-8B wrapper with VAUQ core-region attention masking.

Ported to match the `get_logits` / `get_logits_masked` interface used by
`vauq/scoring.py`. Uses the HF `AutoModelForImageTextToText` checkpoint.
"""

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# InternVL3.5-8B-HF image context token id.
IMG_CONTEXT_TOKEN_ID = 151671


class InternVL3:
    def __init__(self, version="InternVL3_5-8B", device=None):
        self.version = version
        self.build_model()
        self.device = getattr(self.model, "device", None) or next(
            self.model.parameters()
        ).device

    def build_model(self):
        model_name = "OpenGVLab/InternVL3_5-8B-HF"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer

    def _prepare_inputs(self, image, question):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.model.device)
        return inputs

    def generate_with_ids(self, image, question, temp=0.1):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=64, do_sample=True, temperature=temp
            )
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        answer = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return answer, generated_ids

    def get_logits(self, image, question, generated_ids):
        inputs = self._prepare_inputs(image, question)
        full_ids = torch.cat([inputs.input_ids, generated_ids], dim=1)
        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                pixel_values=inputs.pixel_values,
                return_dict=True,
            )
        prompt_len = inputs.input_ids.shape[1]
        return outputs.logits[0, prompt_len - 1 : -1]

    def get_logits_masked(
        self, image, question, generated_ids, topk_ratio=0.6, layer_range=(10, 25)
    ):
        inputs = self._prepare_inputs(image, question)
        full_ids = torch.cat([inputs.input_ids, generated_ids], dim=1)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                pixel_values=inputs.pixel_values,
                output_attentions=True,
                return_dict=True,
            )

        positions = (full_ids[0] == IMG_CONTEXT_TOKEN_ID).nonzero(as_tuple=True)[0]
        first_pos, last_pos = positions[0].item(), positions[-1].item() + 1

        attentions = torch.stack(outputs.attentions, dim=0).squeeze(1)
        selected_vis_attentions = (
            attentions[layer_range[0] : layer_range[1]][
                :, :, prompt_len:, first_pos:last_pos
            ]
            .mean(0)
            .mean(0)
            .mean(0)
        )
        k = max(1, int(selected_vis_attentions.numel() * topk_ratio))
        _, top_k_indices = torch.topk(selected_vis_attentions, k)

        attention_mask = torch.cat(
            [inputs.attention_mask, torch.ones_like(generated_ids)], dim=1
        )
        attention_mask[0, first_pos + top_k_indices] = 0

        with torch.no_grad():
            outputs_masked = self.model(
                input_ids=full_ids,
                pixel_values=inputs.pixel_values,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return outputs_masked.logits[0, prompt_len - 1 : -1]
