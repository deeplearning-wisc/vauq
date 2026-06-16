import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


class LLaVA:
    """LLaVA 1.5 wrapper with VAUQ core-region attention masking."""

    def __init__(self, version="llava-1.5-7b-hf", device=None):
        self.version = version
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        model_name = f"llava-hf/{self.version}"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self.device)
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
        ).to(self.device)
        return inputs, image

    def generate_with_ids(self, image, question, temp=0.1):
        inputs, _ = self._prepare_inputs(image, question)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=temp,
            )
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        answer = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return answer, generated_ids

    def get_logits(self, image, question, generated_ids):
        inputs, _ = self._prepare_inputs(image, question)
        full_ids = torch.cat([inputs.input_ids, generated_ids], dim=1)

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                pixel_values=inputs.pixel_values,
                output_attentions=True,
                return_dict=True,
            )

        prompt_len = inputs.input_ids.shape[1]
        return outputs.logits[0, prompt_len - 1 : -1]

    def get_logits_masked(
        self,
        image,
        question,
        generated_ids,
        topk_ratio=0.6,
        layer_range=(10, 25),
    ):
        """Forward pass with core vision tokens masked via attention."""
        inputs, _ = self._prepare_inputs(image, question)
        full_ids = torch.cat([inputs.input_ids, generated_ids], dim=1)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                pixel_values=inputs.pixel_values,
                output_attentions=True,
                return_dict=True,
            )

        vision_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        positions = (full_ids[0] == vision_token_id).nonzero(as_tuple=True)[0]
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

        num_tokens = selected_vis_attentions.numel()
        k = max(1, int(num_tokens * topk_ratio))
        _, top_k_indices = torch.topk(selected_vis_attentions, k)

        attention_mask = torch.cat(
            [inputs.attention_mask, torch.ones_like(generated_ids)], dim=1
        )
        absolute_masked_indices = first_pos + top_k_indices
        attention_mask[0, absolute_masked_indices] = 0

        with torch.no_grad():
            outputs_masked = self.model(
                input_ids=full_ids,
                pixel_values=inputs.pixel_values,
                attention_mask=attention_mask,
                return_dict=True,
            )

        return outputs_masked.logits[0, prompt_len - 1 : -1]
