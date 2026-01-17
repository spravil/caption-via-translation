import argparse
import glob
import os
import random
from multiprocessing import Pool
from pathlib import Path

import nltk
import pandas as pd
from tqdm import tqdm

# Download the required resource if you haven't already
nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize


def process_tar_file(args):
    input_data_file_path, output_folder = args

    if (output_folder / input_data_file_path.name).exists():
        return

    # Load dfs that we want to merge
    df = pd.read_parquet(input_data_file_path)

    output_df = []
    for index, row in df.iterrows():
        key = row["__key__"] if "__key__" in row else f"{int(row['key']):09d}"
        output = dict(__key__=key, data=[])

        if row["caption_en_de"] is not None and len(row["caption_en_de"]) == 1:
            en, de = row["caption_en_de"][0]

            if len(en) > 25:
                # Split into sentences
                sentences_de = sent_tokenize(de, language="german")
                sentences_en = sent_tokenize(en, language="english")

                # Fix en data
                sentences_en = [a.capitalize() for a in sentences_en]
                en = " ".join(sentences_en)

                # Short Caption
                output["data"].append(
                    dict(type="caption", lang="de", prompt="<LANG_DE><CAPTION>", output=sentences_de[0])
                )
                output["data"].append(
                    dict(type="caption", lang="en", prompt="<LANG_EN><CAPTION>", output=sentences_en[0])
                )

                # See if the caption is detailed or more detailed
                if len(sentences_de) == 2:
                    output["data"].append(
                        dict(type="caption", lang="de", prompt="<LANG_DE><DETAILED_CAPTION>", output=de)
                    )
                    output["data"].append(
                        dict(type="caption", lang="en", prompt="<LANG_EN><DETAILED_CAPTION>", output=en)
                    )
                elif len(sentences_de) >= 3:
                    output["data"].append(
                        dict(type="caption", lang="de", prompt="<LANG_DE><MORE_DETAILED_CAPTION>", output=de)
                    )
                    output["data"].append(
                        dict(type="caption", lang="en", prompt="<LANG_EN><MORE_DETAILED_CAPTION>", output=en)
                    )

                # Direct translation
                output["data"].append(
                    dict(type="translation", lang="de", prompt=f"<LANG_DE><TRANSLATE>{en}", output=de)
                )

                # Random mix translation
                if len(sentences_de) == len(sentences_en) and len(sentences_de) > 1:
                    list_length = len(sentences_de)
                    list_idx = list(range(list_length))
                    random.shuffle(list_idx)
                    num_items_to_select = random.randint(2, list_length)
                    list_idx = list_idx[:num_items_to_select]
                    sentence_de = " ".join(sentences_de[j] for j in list_idx)
                    sentence_en = " ".join(sentences_en[j] for j in list_idx)
                    output["data"].append(
                        dict(
                            type="translation",
                            lang="de",
                            prompt=f"<LANG_DE><TRANSLATE>{sentence_en}",
                            output=sentence_de,
                        )
                    )

        for lang in ["de", "fr", "es", "zh", "ru"]:
            if row[f"translation_en_{lang}"] is not None:
                # Create single sentence translations
                for t in row[f"translation_en_{lang}"]:
                    en, non_en = t

                    if len(en) > 25:
                        output["data"].append(
                            dict(
                                type="translation",
                                lang=lang,
                                prompt=f"<LANG_{lang.upper()}><TRANSLATE>{en}",
                                output=non_en,
                            )
                        )

                # Create "fake" paragraph translation
                if len(row[f"translation_en_{lang}"]) > 1:
                    sentences_en = []
                    sentences_non_en = []
                    for t in row[f"translation_en_{lang}"]:
                        en, non_en = t

                        if len(en) > 25:
                            sentences_en.append(en)
                            sentences_non_en.append(non_en)

                    if len(sentences_en) > 1:
                        list_length = len(sentences_en)
                        list_idx = list(range(list_length))
                        random.shuffle(list_idx)
                        num_items_to_select = random.randint(2, list_length)
                        list_idx = list_idx[:num_items_to_select]
                        sentence_non_en = " ".join(sentences_non_en[j] for j in list_idx)
                        sentence_en = " ".join(sentences_en[j] for j in list_idx)
                        output["data"].append(
                            dict(
                                type="translation",
                                lang=lang,
                                prompt=f"<LANG_{lang.upper()}><TRANSLATE>{sentence_en}",
                                output=sentence_non_en,
                            )
                        )

        # Make sure that we at least have one annotation
        if len(output) == 0:
            print(f"Incomplete: {row['__key__']}")
            continue

        output_df.append(output)

    pd.DataFrame(output_df).to_parquet(output_folder / input_data_file_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        default="data/cc12m_annotations/train",
        help="Path to the input parquets.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="data/cc12m_annotations_preprocessed/train",
        help="Path to the output folder.",
    )
    parser.add_argument("--num_workers", "-w", type=int, default=8, help="Number of worker processes to use.")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Start all jobs
    parquet_files = [Path(p) for p in glob.glob(os.path.join(args.input_folder, "*.parquet"))]
    with Pool(args.num_workers) as pool:
        list(
            tqdm(
                pool.imap(
                    process_tar_file,
                    list(
                        zip(
                            parquet_files,
                            [Path(args.output_folder)] * len(parquet_files),
                        )
                    ),
                ),
                total=len(parquet_files),
            )
        )
