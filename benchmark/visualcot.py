import json
import os

from PIL import Image


class VisualCot:
    """VisualCoT loader (GQA + TextCap + TextVQA subsets, 500 samples each).

    Paths are configurable because VisualCoT ships as annotation JSON files plus a
    separate image tree. Set them via constructor args or environment variables:
        VISCOT_ANN_DIR   directory containing gqa.json, textcap.json, textvqa.json
        VISCOT_IMAGE_DIR image root (with gqa/images and textvqa/train_images)
    """

    ANNOTATION_FILES = ("gqa.json", "textcap.json", "textvqa.json")

    def __init__(self, annotation_dir=None, image_base_path=None, per_file_limit=500):
        annotation_dir = annotation_dir or os.environ.get("VISCOT_ANN_DIR")
        image_base_path = image_base_path or os.environ.get("VISCOT_IMAGE_DIR")
        if not annotation_dir or not image_base_path:
            raise ValueError(
                "VisualCoT requires local data. Set VISCOT_ANN_DIR (folder with "
                "gqa.json/textcap.json/textvqa.json) and VISCOT_IMAGE_DIR (image root)."
            )
        self.image_base_path = image_base_path

        self.ds = []
        for name in self.ANNOTATION_FILES:
            with open(os.path.join(annotation_dir, name), "r") as f:
                self.ds.extend(json.load(f)[:per_file_limit])

    def obtain_size(self):
        return len(self.ds)

    def retrieve(self, idx):
        row = self.ds[idx]
        dataset = row["dataset"]
        image_filename = os.path.basename(row["image"][0])

        if dataset == "gqa":
            image_path = os.path.join(self.image_base_path, "gqa", "images", image_filename)
        elif dataset in ("textvqa", "textcap"):
            image_path = os.path.join(
                self.image_base_path, "textvqa", "train_images", image_filename
            )
        else:
            image_path = os.path.join(
                self.image_base_path, row["image"][0].replace("cot/", "")
            )

        img = Image.open(image_path).convert("RGB")

        question = row["conversations"][0]["value"].replace("<image>\n", "")
        question = question.replace(
            " Please provide the bounding box coordinate of the region that can "
            "help you answer the question better.",
            "",
        )
        question = (
            f"{question.strip()}\n"
            "NOTE: Provide only the final answer. Do not provide unrelated details."
        )

        return {
            "idx": idx,
            "img": img,
            "question": question,
            "gt_ans": row["conversations"][3]["value"],
            "bbox": row["conversations"][1]["value"],
        }
