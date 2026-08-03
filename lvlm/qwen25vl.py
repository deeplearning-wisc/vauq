"""Qwen2.5-VL-7B wrapper with VAUQ core-region attention masking.

Requires `qwen-vl-utils` (pip install qwen-vl-utils). Ported to match the
`get_logits` / `get_logits_masked` interface used by `vauq/scoring.py`.
"""

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen25VL:
    def __init__(self, version="Qwen2.5-VL-7B-Instruct", device=None):
        self.version = version
        self.build_model()
        self.device = getattr(self.model, "device", None) or next(
            self.model.parameters()
        ).device

    def build_model(self):
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="eager",
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.tokenizer = self.processor.tokenizer
        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|vision_start|>"
        )
        self.vision_end_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|vision_end|>"
        )

    def _prepare_inputs(self, image, question):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        return inputs

    def generate_with_ids(self, image, question, temp=0.1):
        inputs = self._prepare_inputs(image, question)
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
        # drop trailing EOS so scored tokens match the answer content
        return answer, generated_ids[:, :-1]

    def get_logits(self, image, question, generated_ids):
        inputs = self._prepare_inputs(image, question)
        full_ids = torch.cat([inputs.input_ids, generated_ids], dim=1)
        attention_mask = torch.cat(
            [inputs.attention_mask, torch.ones_like(generated_ids)], dim=1
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
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
        attention_mask = torch.cat(
            [inputs.attention_mask, torch.ones_like(generated_ids)], dim=1
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                output_attentions=True,
                return_dict=True,
            )

        prompt_ids = inputs.input_ids.tolist()[0]
        first_pos = prompt_ids.index(self.vision_start_token_id) + 1
        last_pos = prompt_ids.index(self.vision_end_token_id)

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
        position_ids = torch.arange(
            0, full_ids.shape[-1], device=full_ids.device
        ).unsqueeze(0)

        with torch.no_grad():
            outputs_masked = self.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                return_dict=True,
            )
        return outputs_masked.logits[0, prompt_len - 1 : -1]
