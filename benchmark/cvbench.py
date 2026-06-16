from datasets import load_dataset


class CVBench:
    def __init__(self):
        self.ds = load_dataset("nyu-visionx/CV-Bench", "2D")["test"]

    def obtain_size(self):
        return len(self.ds)

    def retrieve(self, idx):
        row = self.ds[idx]
        question = (
            f"{row['prompt']}\n"
            "Your answer should be only the letter of the option."
        )
        return {
            "idx": idx,
            "img": row["image"],
            "question": question,
            "gt_ans": row["answer"],
            "choices": row["choices"],
        }
