import csv
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

lang_map = {
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
}
lang_codes = list(lang_map.keys())


class CustomCsvDataset(Dataset):
    def __init__(self, csv_file, lang_codes, num_captions=5):
        self.data = []
        self.lang_codes = lang_codes
        self.num_captions = num_captions  # We have 5 captions: caption_1 to caption_5

        # Open the CSV file and read it
        with open(csv_file, mode="r") as file:
            csv_reader = csv.reader(file)

            # Iterate over each row in the CSV
            for row in csv_reader:
                # Append the row (as a list) into the data list
                self.data.append(row)

    def __len__(self):
        # Return the number of data points multiplied by the number of languages and captions
        return len(self.data) * len(self.lang_codes) * self.num_captions

    def __getitem__(self, idx):
        # Calculate which row, language, and caption this index corresponds to
        row_idx = idx // (len(self.lang_codes) * self.num_captions)
        lang_caption_idx = idx % (len(self.lang_codes) * self.num_captions)
        lang_idx = lang_caption_idx // self.num_captions
        caption_idx = lang_caption_idx % self.num_captions

        # Get the data point at the calculated row index
        row = self.data[row_idx]
        image_path = row[0]
        description = row[-1]
        captions = row[1:-1]

        # Get the specific caption and language code for the current index
        caption = captions[caption_idx]
        lang_code = self.lang_codes[lang_idx]

        # Return the data along with the specific caption and language code
        return (image_path, caption, description, lang_code)


def main(
    dataset_name="coco_karpathy", batch_size=8, target_langs=None, sep_by_lang=False, num_captions=5, continue_i=None
):
    root_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    if target_langs is None:
        target_langs = lang_codes

    if not sep_by_lang:
        target_langs = [target_langs]
    else:
        target_langs = [[t] for t in target_langs]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang="eng_Latn")
    translator = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B", torch_dtype=torch.float16)
    translator.to(device)
    translator.eval()

    for tlangs in target_langs:
        if len(target_langs) == 1:
            file_path = root_dir / f"translation_data/{dataset_name}_translate.csv"
        else:
            assert len(tlangs) == 1
            file_path = root_dir / f"translation_data/{dataset_name}_{tlangs[0]}_translate.csv"

        if os.path.exists(file_path) and continue_i is None:
            return

        dataset = CustomCsvDataset(
            root_dir / f"recaption_data/{dataset_name}_recap.csv",
            tlangs,
            num_captions=num_captions,
        )
        print(f"Dataset: {dataset_name} with size {len(dataset)} and {tlangs}")

        def collate_fn(data):
            img_paths, captions, descriptions, langs = zip(*data)

            inputs = tokenizer(
                [f"{c} ### ({d})" for c, d in zip(captions, descriptions)],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            decoder_input_ids = torch.tensor(
                [[tokenizer.convert_tokens_to_ids(lang_map[lang])] for lang in langs], dtype=torch.long
            )
            inputs["decoder_input_ids"] = decoder_input_ids

            return inputs, img_paths, captions, descriptions, langs

        def append_to_csv(file_path, *args):
            with open(file_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([*args])

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        for i, (inputs, img_paths, captions, descriptions, langs) in enumerate(tqdm(dataloader)):
            if continue_i is not None and i <= continue_i:
                continue
            with torch.no_grad():
                inputs.to(device)
                translated_tokens = translator.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    do_sample=False,
                    length_penalty=0.9,
                )

            decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

            # Identify translations that failed with simple rules
            retry = []
            decoded_retry = []
            for i, translation in enumerate(decoded):
                new_translation = translation.split("(")[0].split("#")[0]
                model_included_details = len(translation) - len(new_translation) > 0
                model_includes_translation = len(new_translation) > 0
                if not model_included_details or not model_includes_translation:
                    retry.append(i)

            # Translate without context
            if len(retry) > 0:
                inputs = tokenizer(
                    [c for i, c in enumerate(captions) if i in retry],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                decoder_input_ids = torch.tensor(
                    [[tokenizer.convert_tokens_to_ids(lang_map[lang])] for i, lang in enumerate(langs) if i in retry],
                    dtype=torch.long,
                )
                inputs["decoder_input_ids"] = decoder_input_ids
                with torch.no_grad():
                    inputs.to(device)
                    translated_tokens = translator.generate(
                        **inputs,
                        max_length=64,
                        num_beams=4,
                        do_sample=False,
                        length_penalty=0.9,
                    )
                decoded_retry = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

            # Collect results
            for i, (translation, img_path, caption, description, lang) in enumerate(
                zip(decoded, img_paths, captions, descriptions, langs)
            ):
                if i in retry:
                    translation = decoded_retry[retry.index(i)].strip()
                else:
                    translation = translation.split("(")[0].split("#")[0].strip()

                # Fix punctiation
                if translation[-1] not in ".!?":
                    translation = translation + "."

                # Fix punctuation in chinese
                if lang == "zh" and translation[-1] == ".":
                    translation = translation[:-1] + "。"

                append_to_csv(file_path, img_path, lang, caption, translation)


if __name__ == "__main__":
    main(dataset_name="multi30k_task1_val", target_langs=["es", "zh", "ru"], sep_by_lang=True, num_captions=1)
    main(dataset_name="multi30k_task2_val", target_langs=["fr", "es", "zh", "ru"], sep_by_lang=True)
    main(dataset_name="coco_karpathy_restval", target_langs=["de", "fr", "es", "zh", "ru"], continue_i=78541)
    main(dataset_name="coco_karpathy")
    main(dataset_name="multi30k_task2")
    main(dataset_name="multi30k_task1", target_langs=["es", "zh", "ru"], num_captions=1, batch_size=64)
