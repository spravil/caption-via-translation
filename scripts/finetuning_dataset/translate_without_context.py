import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scripts.finetuning_dataset.datasets_to_csv import DOCCIDataset, ImageParagraphsDataset, write_tuples_to_csv

lang_map = {
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
}
lang_codes = list(lang_map.keys())


class LangAugmentDataset(Dataset):
    def __init__(self, base_dataset, lang_codes):
        self.base_dataset = base_dataset
        self.lang_codes = lang_codes

    def __len__(self):
        # Return the number of data points multiplied by the number of languages
        return len(self.base_dataset) * len(self.lang_codes)

    def __getitem__(self, idx):
        # Calculate which row and language this index corresponds to
        row_idx = idx // len(self.lang_codes)
        lang_idx = idx % len(self.lang_codes)

        # Get the data point from the base dataset at the calculated row index
        data = self.base_dataset[row_idx]

        # Get the specific language code for the current index
        lang_code = self.lang_codes[lang_idx]

        # Return the data along with the specific language code
        return (*data, lang_code)


def main(dataset_name="image_paragraphs", batch_size=8):
    finetuning_datasets_root_dir = Path("path/to/finetuning_datasets")
    root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    file_path = root_dir / f"translation_data/{dataset_name}_translate.csv"
    if os.path.exists(file_path):
        return

    if dataset_name == "image_paragraphs":
        dataset = LangAugmentDataset(
            ImageParagraphsDataset(finetuning_datasets_root_dir / "image_paragraphs", split="train"), lang_codes
        )
    elif dataset_name == "docci":
        dataset = LangAugmentDataset(DOCCIDataset(finetuning_datasets_root_dir / "docci", split="train"), lang_codes)
    else:
        raise Exception()

    print(f"Dataset: {dataset_name} with size {len(dataset)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang="eng_Latn")
    translator = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B", torch_dtype=torch.float16)
    translator.to(device)
    translator.eval()

    def collate_fn(data):
        img_paths, captions, langs = zip(*data)

        inputs = tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        decoder_input_ids = torch.tensor(
            [[tokenizer.convert_tokens_to_ids(lang_map[lang])] for lang in langs], dtype=torch.long
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs, img_paths, captions, langs

    results = []
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for inputs, img_paths, captions, langs in tqdm(dataloader):
        with torch.no_grad():
            inputs.to(device)
            translated_tokens = translator.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                do_sample=False,
                length_penalty=1.0,
            )

        decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        # Collect results
        for i, (translation, img_path, caption, lang) in enumerate(zip(decoded, img_paths, captions, langs)):
            # Fix punctiation
            if translation[-1] not in ".!?":
                translation = translation + "."

            # Fix punctuation in chinese
            if lang == "zh" and translation[-1] == ".":
                translation = translation[:-1] + "。"

            results.append((img_path, lang, caption, translation))

    write_tuples_to_csv(file_path, results)


if __name__ == "__main__":
    main(dataset_name="image_paragraphs")
    main(dataset_name="docci")
