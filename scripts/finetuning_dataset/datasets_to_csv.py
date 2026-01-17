import csv
import gzip
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Union

from torch.utils.data import Dataset
from tqdm import tqdm


class ImageParagraphsDataset(Dataset):
    def __init__(self, dataset_path, split="train"):
        assert split in ["train", "val", "test"], "Split must be 'train', 'val', or 'test'."
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.data = self._load_data()

    def _load_data(self):
        # Load the split file
        split_file_path = self.dataset_path / f"{self.split}_split.json"
        with open(split_file_path, "r", encoding="utf-8") as f:
            split_ids = set(json.load(f))

        # Load the main JSON file
        json_file_path = self.dataset_path / "paragraphs_v1.json"
        with open(json_file_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        data = []
        for item in items:
            image_id = item.get("image_id")
            url = item.get("url")
            paragraph = item.get("paragraph")

            if image_id in split_ids and url and paragraph:
                # Extract the filename from the URL
                filename = url.split("/")[-1]

                # Determine the directory prefix
                directory = url.split("/")[-2]
                prefix = directory + "_"

                # Construct the image path
                image_path = self.dataset_path / "images" / (prefix + filename)

                data.append((image_path, paragraph))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, paragraph = self.data[idx]
        return image_path, paragraph


class DOCCIDataset(Dataset):
    """
    wget https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines
    wget https://storage.googleapis.com/docci/data/docci_images.tar.gz
    tar -xvzf docci_images.tar.gz
    """

    def __init__(self, dataset_path, split):
        assert split in ["qual_dev", "qual_test", "train", "test"]
        self.dataset_path = Path(dataset_path)
        self.image_folder_path = self.dataset_path / "images"
        self.jsonl_file_path = self.dataset_path / "docci_descriptions.jsonlines"
        self.data = self._load_data(self.jsonl_file_path, split)

    def _load_data(self, jsonl_file_path, target_split):
        data = []
        with open(jsonl_file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("split") == target_split:
                    image_path = os.path.join(self.image_folder_path, item.get("image_file"))
                    data.append((image_path, item.get("description")))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, description = self.data[idx]
        return image_path, description


class COCOKarpathyDataset(Dataset):
    """
    Setup:
        wget https://github.com/Delphboy/karpathy-splits/raw/refs/heads/main/dataset_coco.json
        mv dataset_coco.json annotations/coco_karpathy.json
        wget http://images.cocodataset.org/zips/val2014.zip
        wget http://images.cocodataset.org/zips/train2014.zip
        unzip val2014.zip
        unzip train2014.zip

    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        split: str = "train",
        lang: str = "en",
    ) -> None:
        if split not in ["train", "test", "val", "restval"]:
            raise ValueError(f"{split} is not supported")

        if lang == "en":
            label_path = Path(dataset_path) / "annotations" / "coco_karpathy.json"
        else:
            raise NotImplementedError(f"Language {lang} is not supported!")

        print("Loading Coco labels...")
        with open(label_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        label_dict: DefaultDict[str, Dict[str, List[str]]] = defaultdict(dict)
        for img in data["images"]:
            img_split = img["split"]
            img_p = img["filename"]
            captions = [c["raw"] for c in img["sentences"]]
            label_dict[img_split][img_p] = captions
        label_dict = label_dict[split]

        samples = []
        for img_name, captions in tqdm(label_dict.items(), desc="Loading Coco dataset", disable=None, leave=True):
            img_path = Path(dataset_path) / ("train2014" if "train2014" in img_name else "val2014") / img_name
            # Aggregation
            new_sample = (img_path, captions)
            samples.append(new_sample)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, captions = self.samples[idx]
        return img_path, captions


class Multi30kDataset(Dataset):
    """Multi30k Dataset."""

    def __init__(
        self,
        dataset_path: Union[str, Path],
        flickr30k_images_path: Union[str, Path],
        coco_images_path: Union[str, Path],
        split: str = "train",
        lang: str = "en",
        task: str = "task1",
    ) -> None:
        img_path = None
        if task == "task1":
            assert lang in ["en", "de", "fr", "cs"], "Language not supported"
            allowed_splits = [
                "train",
                "val",
                "test_2016_flickr",
                "test_2017_flickr",
                "test_2017_mscoco",
                "test_2018_flickr",
            ]
            if split not in allowed_splits:
                allowed_split_str = ", ".join(allowed_splits)
                raise ValueError(f"{split} is not supported for task1 ({allowed_split_str})")

            # Identify the annotation files one for each of the 5 captions per image
            annotation_files_en = [Path(dataset_path) / "data/task1/raw" / f"{split}.en.gz"]
            annotation_files = [Path(dataset_path) / "data/task1/raw" / f"{split}.{lang}.gz"]
            img_path_file = Path(dataset_path) / "data/task1/image_splits" / f"{split}.txt"

            if split == "test_2017_flickr":
                img_path = Path(dataset_path) / "task1_test_2017"
            elif split == "test_2018_flickr":
                img_path = Path(dataset_path) / "task1_test_2018"
            elif split == "test_2017_mscoco":
                img_path = coco_images_path
            else:
                img_path = flickr30k_images_path

        elif task == "task2":
            assert lang in ["en", "de"], "Language not supported"
            allowed_splits = ["train", "val", "test_2016"]
            if split not in allowed_splits:
                allowed_split_str = ", ".join(allowed_splits)
                raise ValueError(f"{split} is not supported for task2 ({allowed_split_str})")
            annotation_files_en = [Path(dataset_path) / "data/task2/raw" / f"{split}.{i + 1}.en.gz" for i in range(5)]
            annotation_files = [Path(dataset_path) / "data/task2/raw" / f"{split}.{i + 1}.{lang}.gz" for i in range(5)]
            img_path_file = Path(dataset_path) / "data/task2/image_splits" / f"{split}_images.txt"
            img_path = flickr30k_images_path

        img_path = Path(img_path)

        # Create the datapipe
        def read_gzip_file(filepath):
            with gzip.open(filepath, "rb") as f:
                return [line.decode("UTF-8").strip().replace("\n", "") for line in f.readlines()]

        def read_file(filepath):
            with open(filepath, "r") as f:
                return [line.strip().replace("\n", "") for line in f.readlines()]

        annotations_en = [read_gzip_file(a) for a in annotation_files_en]
        annotations = [read_gzip_file(a) for a in annotation_files]
        images = read_file(img_path_file)

        if task == "task1" and split == "test_2017_mscoco":
            images = [
                img_path / ("train2014" if "train2014" in filename else "val2014") / filename.split("#")[0]
                for filename in images
            ]

        else:
            images = [img_path / filename for filename in images]

        datapipe = list(zip(images, zip(*annotations), zip(*annotations_en)))

        # Iterate over the samples
        samples = []
        for img_path, captions_de, captions_en in tqdm(
            datapipe,
            desc="Loading Multi30k dataset",
            disable=None,
            leave=True,
        ):
            samples.append((img_path, captions_en, captions_de))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, captions_en, captions_de = self.samples[idx]
        return img_path, captions_en, captions_de


def write_tuples_to_csv(file_path, data):
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(data)


if __name__ == "__main__":
    finetuning_datasets_root_dir = Path("path/to/finetuning_datasets")
    root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    # Load img_paths that should be ignored
    multi30k_ignore_file = root_dir / "ignore_data/multi30k_task1_test_2017_mscoco_ignore.csv"
    if not multi30k_ignore_file.exists():
        multi30k_ignore_file.parent.mkdir(parents=True)
        ignore_list = []
        dataset_multi30k = Multi30kDataset(
            dataset_path=finetuning_datasets_root_dir / "multi30k",
            flickr30k_images_path=finetuning_datasets_root_dir / "multi30k/flickr30k-images",
            coco_images_path=finetuning_datasets_root_dir / "coco",
            task="task1",
            split="test_2017_mscoco",
        )
        for img_path, *_ in tqdm(iter(dataset_multi30k)):
            ignore_list.append((img_path,))
        write_tuples_to_csv(multi30k_ignore_file, ignore_list)

    # Write gt data to csv
    coco_karpathy_gt_data = root_dir / "gt_data/coco_karpathy_train_gt.csv"
    coco_karpathy_gt_data.parent.mkdir(parents=True)
    if not coco_karpathy_gt_data.exists():
        dataset_coco_karpathy = COCOKarpathyDataset(finetuning_datasets_root_dir / "coco", split="train")
        data = []
        img_count = 0
        for img_path, captions in tqdm(iter(dataset_coco_karpathy)):
            if img_path in ignore_list:
                continue
            img_count += 1
            for caption in captions:
                caption = caption.replace("\n", "").strip()
                data.append((img_path, "en", caption, None))
        write_tuples_to_csv(coco_karpathy_gt_data, data)
        print(img_count, len(data))

    # Write gt data to csv
    coco_karpathy_restval_gt_data = root_dir / "gt_data/coco_karpathy_restval_gt.csv"
    if not coco_karpathy_restval_gt_data.exists():
        dataset_coco_karpathy = COCOKarpathyDataset(finetuning_datasets_root_dir / "coco", split="restval")
        data = []
        img_count = 0
        for img_path, captions in tqdm(iter(dataset_coco_karpathy)):
            if img_path in ignore_list:
                continue
            img_count += 1
            for caption in captions:
                caption = caption.replace("\n", "").strip()
                data.append((img_path, "en", caption, None))
        write_tuples_to_csv(coco_karpathy_restval_gt_data, data)
        print(img_count, len(data))

    multi30k_task2_train_gt_data = root_dir / "gt_data/multi30k_task2_train_gt.csv"
    if not multi30k_task2_train_gt_data.exists():
        dataset_multi30k = Multi30kDataset(
            dataset_path=finetuning_datasets_root_dir / "multi30k",
            flickr30k_images_path=finetuning_datasets_root_dir / "multi30k/flickr30k-images",
            coco_images_path=finetuning_datasets_root_dir / "coco",
            task="task2",
            split="train",
            lang="de",
        )
        data = []
        for img_path, captions_en, captions_de in tqdm(iter(dataset_multi30k)):
            for caption_en, caption_de in zip(captions_en, captions_de):
                caption_en = caption_en.replace("\n", "").strip()
                caption_de = caption_de.replace("\n", "").strip()
                data.append((img_path, "en", caption_en, None))
                data.append((img_path, "de", caption_de, None))
        write_tuples_to_csv(multi30k_task2_train_gt_data, data)

    multi30k_task1_train_gt_data = root_dir / "gt_data/multi30k_task1_train_gt.csv"
    if not multi30k_task1_train_gt_data.exists():
        dataset_multi30k = Multi30kDataset(
            dataset_path=finetuning_datasets_root_dir / "multi30k",
            flickr30k_images_path=finetuning_datasets_root_dir / "multi30k/flickr30k-images",
            coco_images_path=finetuning_datasets_root_dir / "coco",
            task="task1",
            split="train",
            lang="de",
        )
        data = []
        for img_path, captions_en, captions_de in tqdm(iter(dataset_multi30k)):
            for caption_en, caption_de in zip(captions_en, captions_de):
                caption_en = caption_en.replace("\n", "").strip()
                caption_de = caption_de.replace("\n", "").strip()
                data.append((img_path, "de", caption_en, caption_de))
        dataset_multi30k = Multi30kDataset(
            dataset_path=finetuning_datasets_root_dir / "multi30k",
            flickr30k_images_path=finetuning_datasets_root_dir / "multi30k/flickr30k-images",
            coco_images_path=finetuning_datasets_root_dir / "coco",
            task="task1",
            split="train",
            lang="fr",
        )
        for img_path, captions_en, captions_fr in tqdm(iter(dataset_multi30k)):
            for caption_en, caption_fr in zip(captions_en, captions_fr):
                caption_en = caption_en.replace("\n", "").strip()
                caption_fr = caption_fr.replace("\n", "").strip()
                data.append((img_path, "fr", caption_en, caption_fr))
        write_tuples_to_csv(multi30k_task1_train_gt_data, data)

    # Multi30k task 1 validation
    for lang in ["de", "fr"]:
        multi30k_data = root_dir / f"gt_data/multi30k_task1_val_{lang}_gt.csv"
        if multi30k_data.exists():
            continue
        dataset_multi30k = Multi30kDataset(
            dataset_path=finetuning_datasets_root_dir / "multi30k",
            flickr30k_images_path=finetuning_datasets_root_dir / "multi30k/flickr30k-images",
            coco_images_path=finetuning_datasets_root_dir / "coco",
            task="task1",
            split="val",
            lang=lang,
        )
        data = []
        for img_path, captions_en, captions_fr in tqdm(iter(dataset_multi30k)):
            for caption_en, caption_fr in zip(captions_en, captions_fr):
                caption_en = caption_en.replace("\n", "").strip()
                caption_fr = caption_fr.replace("\n", "").strip()
                data.append((img_path, lang, caption_en, caption_fr))
        write_tuples_to_csv(multi30k_data, data)

    # Multi30k task 2 validation
    for lang in ["en", "de"]:
        multi30k_data = root_dir / f"gt_data/multi30k_task2_val_{lang}_gt.csv"
        if multi30k_data.exists():
            continue
        dataset_multi30k = Multi30kDataset(
            dataset_path=finetuning_datasets_root_dir / "multi30k",
            flickr30k_images_path=finetuning_datasets_root_dir / "multi30k/flickr30k-images",
            coco_images_path=finetuning_datasets_root_dir / "coco",
            task="task2",
            split="val",
            lang="de",
        )
        data = []
        for img_path, captions_en, captions_de in tqdm(iter(dataset_multi30k)):
            for caption_en, caption_de in zip(captions_en, captions_de):
                if lang == "en":
                    caption_en = caption_en.replace("\n", "").strip()
                    data.append((img_path, "en", caption_en, None))
                elif lang == "de":
                    caption_de = caption_de.replace("\n", "").strip()
                    data.append((img_path, "de", caption_de, None))
        write_tuples_to_csv(multi30k_data, data)

    image_paragraphs_train_gt_data = root_dir / "gt_data/image_paragraphs_train_gt.csv"
    if not image_paragraphs_train_gt_data.exists():
        dataset_image_paragraphs = ImageParagraphsDataset(
            finetuning_datasets_root_dir / "image_paragraphs", split="train"
        )
        data = []
        for img_path, caption in tqdm(iter(dataset_image_paragraphs)):
            caption = re.sub(r"\n+", " ", caption).strip()
            data.append((img_path, "en", caption, None))
        write_tuples_to_csv(image_paragraphs_train_gt_data, data)

    docci_train_gt_data = root_dir / "gt_data/docci_train_gt.csv"
    if not docci_train_gt_data.exists():
        dataset_docci = DOCCIDataset(finetuning_datasets_root_dir / "docci", split="train")
        data = []
        for img_path, caption in tqdm(iter(dataset_docci)):
            caption = re.sub(r"\n+", " ", caption).strip()
            data.append((img_path, "en", caption, None))
        write_tuples_to_csv(docci_train_gt_data, data)
