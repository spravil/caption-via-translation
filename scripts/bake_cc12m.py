import argparse
import glob
import io
import json
import os
import tarfile
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoTokenizer

from caption_via_translation.process.processing_florence2 import Florence2Processor

# This script overwrites a webdataset annonation data field with a parquet file

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["TOKENIZERS_PARALLELISM"] = "1"


def process_tar_file(args):
    # file_path: The file path to the "old" data that should be updated
    # extra_base_url: The folder containing the new annotations i.e. patch
    # new_output_folder: The output folder
    file_path, extra_base_url, new_output_folder, tokenizer, encoder_tokenizer, construct_prompts = args

    # Get the actual output file and patch file
    filename = os.path.basename(file_path).split(".")[0]
    output_file_path = new_output_folder / (filename + ".tar")
    column_url = extra_base_url / (filename + ".parquet")

    # Check if the patch file exists
    if not os.path.exists(column_url):
        print(f"Skip {file_path}")
        return

    # Load patch file
    column_src = pd.read_parquet(column_url)

    total = 0
    skipped = 0
    new_file_contents = {}
    with tarfile.open(file_path, "r:") as tar:
        try:
            members = tar.getmembers()
        except Exception as e:
            print(f"Error reading tar file {file_path}: {str(e)}")
            return

        file_contents = {member.name: tar.extractfile(member).read() for member in members}
        print(f"Processing {file_path} with {len(file_contents)} files.")

        # For all files in the tar
        for filename, filebytes in file_contents.items():
            # Only keep the jpg files
            if ".jpg" not in filename:
                continue

            total += 1

            key = filename.split(".")[0]

            # Get annotation
            annotation = column_src.loc[column_src["__key__"] == key].to_dict("records")
            if len(annotation) == 0:
                skipped += 1
                continue

            # Check that annotation has entries
            annotation = annotation[0]["data"]
            if len(annotation) == 0:
                skipped += 1
                continue

            # Check if image is corrupted
            image_is_okay = False
            try:
                image_stream = io.BytesIO(filebytes)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", UserWarning)
                    with Image.open(image_stream) as img:
                        img.load()
                        img.verify()
                    if len(w) == 0:
                        image_is_okay = True
                    else:
                        image_is_okay = False
            except Exception as e:
                print(f"Warning: image of {key} can not be loaded: {str(e)}")
                image_is_okay = False
            if not image_is_okay:
                skipped += 1
                continue

            # Tokenize annotations
            tokenized_annotations = []
            for sample in annotation:
                prompts = construct_prompts([sample["prompt"]])
                texts = [sample["output"]]
                prompt_ids = encoder_tokenizer(
                    prompts,
                    return_tensors="np",
                )["input_ids"]
                output_ids = tokenizer(
                    texts,
                    return_tensors="np",
                )["input_ids"]

                sample["prompt_ids"] = prompt_ids[0].tolist()
                sample["output_ids"] = output_ids[0].tolist()
                tokenized_annotations.append(sample)

            # Add jpg and annotation to new tar
            annotation = json.dumps(tokenized_annotations).encode("utf-8")
            new_file_contents[filename] = filebytes
            new_file_contents[filename.replace(".jpg", ".data")] = annotation

        del members
        del file_contents

    # Write new tar
    with tarfile.open(output_file_path, "w:") as tar:
        for name, content in new_file_contents.items():
            tarinfo = tarfile.TarInfo(name=name)
            tarinfo.size = len(content)
            tar.addfile(tarinfo, io.BytesIO(content))

    print(f"Done with {file_path}: Skipped {skipped} of {total} images.")


def main(tokenizer, encoder_tokenizer, construct_prompts):
    parser = argparse.ArgumentParser(description="Process tar files and prepare dataset.")
    parser.add_argument(
        "--extra_base_url",
        "-e",
        type=str,
        required=True,
        help="Path to the base URL for extra data.",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        required=True,
        help="Path to the folder containing tar files.",
    )
    parser.add_argument(
        "--new_output_folder",
        "-n",
        type=str,
        required=True,
        help="Path to the folder to save processed files.",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=8,
        help="Number of worker processes to use.",
    )
    parser.add_argument("--download_from_hf", action="store_true")

    args = parser.parse_args()

    # Make sure the output folder exists
    os.makedirs(args.new_output_folder, exist_ok=True)

    # Download the hf dataset
    if args.download_from_hf:
        print("Download pixparse/cc12m-wds huggingface dataset ...")
        snapshot_download(
            repo_id="pixparse/cc12m-wds",
            repo_type="dataset",
            local_dir=args.output_folder,
        )

    # Start all jobs
    tar_files = glob.glob(os.path.join(args.output_folder, "*.tar"))

    missing = []
    tar_files_clean = []
    for tar_path in tar_files:
        base = os.path.splitext(os.path.basename(tar_path))[0]  # e.g. foo.tar -> foo
        parquet_path = os.path.join(args.extra_base_url, base + ".parquet")
        if not os.path.isfile(parquet_path):
            missing.append((tar_path, parquet_path))
        else:
            tar_files_clean.append(tar_path)
    tar_files = tar_files_clean

    with Pool(args.num_workers) as pool:
        list(
            tqdm(
                pool.imap(
                    process_tar_file,
                    list(
                        zip(
                            tar_files,
                            [Path(args.extra_base_url)] * len(tar_files),
                            [Path(args.new_output_folder)] * len(tar_files),
                            [tokenizer] * len(tar_files),
                            [encoder_tokenizer] * len(tar_files),
                            [construct_prompts] * len(tar_files),
                        )
                    ),
                ),
                total=len(tar_files),
            )
        )


if __name__ == "__main__":
    model_name = "microsoft/Florence-2-base"
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b",
        add_bos_token=True,
        add_eos_token=True,
        padding_side="right",
        truncation_side="right",
    )
    processor = Florence2Processor.from_pretrained(model_name, new_tokenizer=tokenizer, use_encoder_tokenizer=True)
    tokenizer = processor.tokenizer
    encoder_tokenizer = processor.encoder_tokenizer
    construct_prompts = processor._construct_prompts

    main(tokenizer, encoder_tokenizer, construct_prompts)
