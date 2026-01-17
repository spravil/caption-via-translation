import argparse
import os
from pathlib import Path

import pandas as pd

targets = {
    "cc12m_en_caption": ("caption_en", "en"),
    "cc12m_de_caption": ("caption_de", "de"),
    "cc12m_es_caption": ("caption_es", "es"),
    "cc12m_zh_caption": ("caption_zh", "zh"),
    "cc12m_es_translation": ("translation_en_es", "es"),
    "cc12m_zh_translation": ("translation_en_zh", "zh"),
}

cap_prompt = {
    "short": "<LANG_{lang}><CAPTION>",
    "detailed": "<LANG_{lang}><DETAILED_CAPTION>",
    "more_detailed": "<LANG_{lang}><MORE_DETAILED_CAPTION>",
}
translation_prompt = "<LANG_{lang}><TRANSLATE>{source}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="input_data_file_path",
        default="data/cc12m_annotations/eval",
        help="Path to the input parquets.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_folder",
        default="data/cc12m_annotations_preprocessed/eval",
        help="Path to the output folder.",
    )
    args = parser.parse_args()

    target_filename = "cc12m-train-2175.parquet"
    input_data_file_path = Path(args.input_data_file_path) / target_filename
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_parquet(input_data_file_path)

    for target_folder, (column_name, lang) in targets.items():
        output_df = []

        if "caption" in column_name:
            captions = df[["key", column_name, column_name + "_style"]]

            for _, row in captions.iterrows():
                key = row["__key__"] if "__key__" in row else f"{int(row['key']):09d}"
                output = dict(__key__=key, data=[])

                caption, caption_style = row[column_name], row[column_name + "_style"]
                if caption is None:
                    continue
                prompt = cap_prompt[caption_style].format(lang=lang.upper())
                output["data"].append(dict(type="caption", lang=lang, prompt=prompt, output=caption))
                output_df.append(output)

        else:
            translations = df[["key", column_name]]
            for _, row in translations.iterrows():
                key = row["__key__"] if "__key__" in row else f"{int(row['key']):09d}"
                output = dict(__key__=key, data=[])

                translation = row[column_name]
                if translation is None:
                    continue
                assert len(translation) == 2
                source, target = translation
                prompt = translation_prompt.format(source=source, lang=lang.upper())
                output["data"].append(dict(type="translation", lang=lang, prompt=prompt, output=target))
                output_df.append(output)

        out = Path(output_folder) / target_folder
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(output_df).to_parquet(out / target_filename)


if __name__ == "__main__":
    main()
