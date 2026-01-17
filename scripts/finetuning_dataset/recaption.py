import csv
import os
import re
from pathlib import Path

import sglang as sgl
from sglang.lang.chat_template import get_chat_template
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.finetuning_dataset.datasets_to_csv import COCOKarpathyDataset, Multi30kDataset


@sgl.function
def image_qa(s, image_path, question):
    s += sgl.user(sgl.image(image_path) + question)
    s += sgl.assistant("The image shows " + sgl.gen("answer"))


def append_to_csv(file_path, image_path, captions, new_caption):
    """Append image_path and caption to the CSV file if image_path is not already present."""
    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([image_path, *captions, new_caption])


def main(dataset_name="coco_karpathy", batch_size=64):
    finetuning_datasets_root_dir = Path("path/to/finetuning_datasets")
    root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    file_path = root_dir / f"recaption_data/{dataset_name}_recap.csv"
    if os.path.exists(file_path):
        return

    if dataset_name == "multi30k_task1":
        csv_file = root_dir / "recaption_data/multi30k_task2_recap.csv"
        data = {}
        with open(csv_file, mode="r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data[row[0]] = row[-1]

            dataset = Multi30kDataset(
                dataset_path=finetuning_datasets_root_dir / "multi30k",
                flickr30k_images_path=finetuning_datasets_root_dir / "multi30k/flickr30k-images",
                coco_images_path=finetuning_datasets_root_dir / "coco",
                task="task1",
                split="train",
            )

            for img_path, captions, *_ in tqdm(iter(dataset)):
                img_path = str(img_path)
                append_to_csv(
                    file_path, str(img_path), [re.sub(r"\n+", " ", c).strip() for c in captions], data[img_path]
                )
        return

    runtime = sgl.Runtime(model_path="lmms-lab/llama3-llava-next-8b")
    runtime.endpoint.chat_template = get_chat_template("llama-3-instruct-llava")

    sgl.set_default_backend(runtime)
    print(f"chat template: {runtime.endpoint.chat_template.name}")

    if dataset_name == "coco_karpathy":
        dataset = COCOKarpathyDataset(finetuning_datasets_root_dir / "coco")
    elif dataset_name == "coco_karpathy_restval":
        dataset = COCOKarpathyDataset(finetuning_datasets_root_dir / "coco", "restval")
    elif dataset_name == "multi30k_task2":
        dataset = Multi30kDataset(
            dataset_path=finetuning_datasets_root_dir / "multi30k",
            flickr30k_images_path=finetuning_datasets_root_dir / "multi30k/flickr30k-images",
            coco_images_path=finetuning_datasets_root_dir / "coco",
            task="task2",
            split="train",
        )
    elif dataset_name == "multi30k_task2_val":
        dataset = Multi30kDataset(
            dataset_path=finetuning_datasets_root_dir / "multi30k",
            flickr30k_images_path=finetuning_datasets_root_dir / "multi30k/flickr30k-images",
            coco_images_path=finetuning_datasets_root_dir / "coco",
            task="task2",
            split="val",
        )
    elif dataset_name == "multi30k_task1_val":
        dataset = Multi30kDataset(
            dataset_path=finetuning_datasets_root_dir / "multi30k",
            flickr30k_images_path=finetuning_datasets_root_dir / "multi30k/flickr30k-images",
            coco_images_path=finetuning_datasets_root_dir / "coco",
            task="task1",
            split="val",
        )
    else:
        raise Exception(f"Dataset {dataset_name} not supported!")

    def collate_fn(data):
        img_paths, captions, *_ = zip(*data)
        return img_paths, captions

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for img_paths, captions, *_ in tqdm(dataloader):
        states = image_qa.run_batch(
            [
                {
                    "image_path": str(ip),
                    "question": "Describe the image in detail. If you see text or objects, describe them in detail, as well as all other aspects of the foreground and background.",
                }
                for ip in img_paths
            ],
            max_new_tokens=64,
            progress_bar=False,
            temperature=0,
        )
        for i, s in enumerate(states):
            reply = re.sub(r"\n+", " ", s["answer"]).strip()
            reply = reply[0].upper() + reply[1:]
            append_to_csv(file_path, str(img_paths[i]), [re.sub(r"\n+", " ", c).strip() for c in captions[i]], reply)

    runtime.shutdown()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    main(dataset_name="coco_karpathy")
    main(dataset_name="coco_karpathy_restval")
    main(dataset_name="multi30k_task2")
    main(dataset_name="multi30k_task1")
    main(dataset_name="multi30k_task2_val")
    main(dataset_name="multi30k_task1_val")
