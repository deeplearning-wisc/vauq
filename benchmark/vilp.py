import os
from io import BytesIO

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image


class VILP:
    """ViLP benchmark: paired images per question (2 cases per row)."""

    DATASET_REPO = "ViLP/ViLP"
    PARQUET_FILE = "ViLP.parquet"

    def __init__(self, token=None):
        token = token or os.environ.get("HF_TOKEN")
        parquet_path = hf_hub_download(
            repo_id=self.DATASET_REPO,
            filename=self.PARQUET_FILE,
            repo_type="dataset",
            token=token,
        )
        self.ds = pd.read_parquet(parquet_path)
        self.ds = self.ds.drop(columns=["image3", "answer3"], errors="ignore")

    def obtain_size(self):
        return len(self.ds) * 2

    def retrieve(self, idx):
        original_idx = idx // 2
        image_num = (idx % 2) + 1

        row = self.ds.iloc[original_idx]
        question = (
            f"{row['question']}\n"
            "NOTE: Provide only the final answer. Do not provide unrelated details."
        )

        image_bytes = row[f"image{image_num}"]
        image = None
        if isinstance(image_bytes, bytes):
            try:
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception as exc:
                print(f"Error loading image for idx {idx}: {exc}")

        return {
            "idx": idx,
            "original_idx": original_idx,
            "case": image_num,
            "img": image,
            "question": question,
            "gt_ans": row[f"answer{image_num}"],
        }
