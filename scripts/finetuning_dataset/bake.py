import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import webdataset as wds
from tqdm import tqdm
from transformers import AutoTokenizer

from caption_via_translation.process.processing_florence2 import Florence2Processor

COUNT_PER_TAR = 5_000


def preprocess_translation(img_path, annotations, sub_annotation_type, tokenizer, encoder_tokenizer, construct_prompts):
    lang, source, target = annotations

    prompt = f"<LANG_{lang.upper()}><TRANSLATE>{source}"

    prompts = construct_prompts([prompt])
    texts = [target]
    prompt_ids = encoder_tokenizer(
        prompts,
        return_tensors="np",
    )["input_ids"]
    output_ids = tokenizer(
        texts,
        return_tensors="np",
    )["input_ids"]
    prompt_ids = prompt_ids[0].tolist()
    output_ids = output_ids[0].tolist()

    sample = {
        "img_path": img_path,
        "prompt": prompts[0],
        "prompt_ids": prompt_ids,
        "output": target,
        "output_ids": output_ids,
        "type": "translation",
        "lang": lang.lower(),
    }
    return sample


def preprocess_caption(img_path, annotations, sub_annotation_type, tokenizer, encoder_tokenizer, construct_prompts):
    if len(annotations) == 2:
        lang, caption = annotations
    else:
        lang, caption, caption2 = annotations
        if len(caption2) > 0 and lang != "en":
            caption = caption2

    if sub_annotation_type == "short":
        prompt = f"<LANG_{lang.upper()}><CAPTION>"
    elif sub_annotation_type == "detailed":
        prompt = f"<LANG_{lang.upper()}><DETAILED_CAPTION>"
    elif sub_annotation_type == "more_detailed":
        prompt = f"<LANG_{lang.upper()}><MORE_DETAILED_CAPTION>"
    else:
        raise Exception()

    prompts = construct_prompts([prompt])
    texts = [caption]
    prompt_ids = encoder_tokenizer(
        prompts,
        return_tensors="np",
    )["input_ids"]
    output_ids = tokenizer(
        texts,
        return_tensors="np",
    )["input_ids"]
    prompt_ids = prompt_ids[0].tolist()
    output_ids = output_ids[0].tolist()

    sample = {
        "img_path": img_path,
        "prompt": prompts[0],
        "prompt_ids": prompt_ids,
        "output": caption,
        "output_ids": output_ids,
        "type": "caption",
        "lang": lang.lower(),
    }
    return sample


# Map annotation types to their preprocessing functions
preprocessing_functions = {
    "translation": preprocess_translation,
    "caption": preprocess_caption,
}


# Function to read and process CSV files
def read_and_process_csv(csv_file, annotation_type, sub_annotation_type, processing_kwargs):
    data = defaultdict(list)
    preprocessing_function = preprocessing_functions.get(annotation_type)
    if preprocessing_function is None:
        raise ValueError(f"No preprocessing function for annotation type '{annotation_type}'")

    print(f"Process {csv_file} ...")
    with open(csv_file, mode="r", newline="") as f:
        reader = csv.reader(f)
        for row in tqdm(reader):
            img_path, *annotations = row
            processed_annotations = preprocessing_function(
                img_path, annotations, sub_annotation_type, **processing_kwargs
            )
            data[img_path].append(processed_annotations)

    return data


def main(output_dir="data/finetuning_dataset", is_val=False):
    root_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    model_name = "microsoft/Florence-2-base"
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b",
        add_bos_token=True,
        add_eos_token=True,
        padding_side="right",
        truncation_side="right",
    )
    processor = Florence2Processor.from_pretrained(model_name, new_tokenizer=tokenizer, use_encoder_tokenizer=True)
    processing_kwargs = dict(
        tokenizer=processor.tokenizer,
        encoder_tokenizer=processor.encoder_tokenizer,
        construct_prompts=processor._construct_prompts,
    )

    # Initialize output directory
    os.makedirs(output_dir, exist_ok=True)

    if not is_val:
        data = [
            # CAPTION
            # REAL COCO train en
            (root_dir / "gt_data/coco_karpathy_train_gt.csv", "caption", "short"),
            (root_dir / "gt_data/coco_karpathy_restval_gt.csv", "caption", "short"),
            # REAL Multi30k task 2 train en de
            (root_dir / "gt_data/multi30k_task2_train_gt.csv", "caption", "short"),
            # REAL IP train en
            (root_dir / "gt_data/image_paragraphs_train_gt.csv", "caption", "detailed"),
            # REAL DOCCI train en
            (root_dir / "gt_data/docci_train_gt.csv", "caption", "more_detailed"),
            # COCO train de, fr, es, zh, ru
            (root_dir / "translation_data/coco_karpathy_translate.csv", "caption", "short"),
            (root_dir / "translation_data/coco_karpathy_restval_translate.csv", "caption", "short"),
            # IP de, fr, es, zh, ru
            (root_dir / "translation_data/image_paragraphs_translate.csv", "caption", "detailed"),
            # DOCCI de, fr, es, zh, ru
            (root_dir / "translation_data/docci_translate.csv", "caption", "more_detailed"),
            # Multi30k task 2 fr es zh ru
            (root_dir / "translation_data/multi30k_task2_translate.csv", "caption", "short"),
            # TRANSLATION
            # REAL Multi30k task 1 train en->de, en->fr
            (root_dir / "gt_data/multi30k_task1_train_gt.csv", "translation", None),
            # Multi30k task 1 train en->es, en->zh, en->ru
            (root_dir / "translation_data/multi30k_task1_translate.csv", "translation", None),
            # COCO train en->de, en->fr, en->es, en->zh, en->ru
            (root_dir / "translation_data/coco_karpathy_translate.csv", "translation", None),
            (root_dir / "translation_data/coco_karpathy_restval_translate.csv", "translation", None),
            # IP en->de, en->fr, en->es, en->zh, en->ru
            (root_dir / "translation_data/image_paragraphs_translate.csv", "translation", None),
            # DOCCI en->de, en->fr, en->es, en->zh, en->ru
            (root_dir / "translation_data/docci_translate.csv", "translation", None),
        ]
        shuffle = True
        unfold = False

    else:
        data = [
            (
                root_dir / "gt_data/multi30k_task1_val_de_gt.csv",
                "translation",
                None,
            ),
            (
                root_dir / "gt_data/multi30k_task1_val_fr_gt.csv",
                "translation",
                None,
            ),
            (
                root_dir / "gt_data/multi30k_task2_val_en_gt.csv",
                "caption",
                "short",
            ),
            (
                root_dir / "gt_data/multi30k_task2_val_de_gt.csv",
                "caption",
                "short",
            ),
            (
                root_dir / "translation_data/multi30k_task1_val_es_translate.csv",
                "translation",
                None,
            ),
            (
                root_dir / "translation_data/multi30k_task1_val_ru_translate.csv",
                "translation",
                None,
            ),
            (
                root_dir / "translation_data/multi30k_task1_val_zh_translate.csv",
                "translation",
                None,
            ),
            (
                root_dir / "translation_data/multi30k_task2_val_es_translate.csv",
                "caption",
                "short",
            ),
            (
                root_dir / "translation_data/multi30k_task2_val_fr_translate.csv",
                "caption",
                "short",
            ),
            (
                root_dir / "translation_data/multi30k_task2_val_ru_translate.csv",
                "caption",
                "short",
            ),
            (
                root_dir / "translation_data/multi30k_task2_val_zh_translate.csv",
                "caption",
                "short",
            ),
        ]
        shuffle = False
        unfold = True

    # Collect and group all data by img_path
    print("Process annotations ...")
    all_data = defaultdict(list)
    for csv_file, annotation_type, sub_annotation_type in data:
        file_data = read_and_process_csv(csv_file, annotation_type, sub_annotation_type, processing_kwargs)
        for img_path, annotations in tqdm(file_data.items()):
            all_data[img_path].extend(annotations)

    image_paths = list(all_data.keys())

    if shuffle:
        random.shuffle(image_paths)

    # Write data to WebDataset
    with wds.ShardWriter(f"{output_dir}/%06d.tar", maxcount=COUNT_PER_TAR, encoder=True) as sink:
        i = 0
        for img_path in tqdm(image_paths):
            data = all_data[img_path]

            # Read image data (assuming images are stored locally)
            with open(img_path, "rb") as img_file:
                img_data = img_file.read()

            if not unfold:
                # Create a unique key for each data item
                key = f"{i:06d}"

                # Write to WebDataset
                sink.write(
                    {
                        "__key__": key,
                        "jpg": img_data,
                        "data": json.dumps(data).encode("utf-8"),  # Adjust as needed based on annotation format
                    }
                )
                i += 1
            else:
                has_cap = []
                random.shuffle(data)
                for d in data:
                    if d["type"] == "caption" and d["lang"] in has_cap:
                        continue

                    if d["type"] == "caption":
                        has_cap.append(d["lang"])
                    # Create a unique key for each data item
                    key = f"{i:06d}"
                    # Write to WebDataset
                    sink.write(
                        {
                            "__key__": key,
                            "jpg": img_data,
                            "data": json.dumps([d]).encode("utf-8"),  # Adjust as needed based on annotation format
                        }
                    )
                    i += 1


if __name__ == "__main__":
    main()
