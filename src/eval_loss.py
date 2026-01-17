import argparse
import json
from pathlib import Path

import torch
import torchvision
from fvcore.nn import FlopCountAnalysis, parameter_count
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import PadDataCollator, create_eval_datasets, prepare_model


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model performance on datasets.")

    # Dataset config
    parser.add_argument(
        "--eval_dataset_root_path",
        type=str,
        default="data/eval_dataset/",
        help="Root path to evaluation datasets.",
    )
    parser.add_argument(
        "--dataset_stats_file",
        type=str,
        default="data/tok_counts.json",
        help="Path to the dataset statistics file.",
    )

    # Model config
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Florence-2-base",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to the pretrained model.",
    )
    parser.add_argument("--image_size", type=int, default=224, help="Input image size for evaluation.")
    parser.add_argument(
        "--max_encoder_seq_length",
        type=int,
        default=447,
        help="Maximum encoder sequence length.",
    )
    parser.add_argument(
        "--max_decoder_seq_length",
        type=int,
        default=1024,
        help="Maximum decoder sequence length.",
    )
    parser.add_argument(
        "--insert_lang_token",
        action="store_true",
        help="Insert language token if specified.",
    )
    parser.add_argument("--no_bos_token", action="store_true", help="Remove decoder start token id.")

    # Training details config
    parser.add_argument(
        "--training_image_size",
        type=int,
        default=224,
        help="Image size used during training.",
    )
    parser.add_argument(
        "--training_batch_size",
        type=int,
        default=1024,
        help="Batch size used during training.",
    )
    parser.add_argument("--training_steps", type=int, default=10000, help="Number of training steps.")
    parser.add_argument(
        "--balancing_type",
        type=str,
        default="task-lang-balancing",
        help="Type of data balancing.",
    )

    # Run config
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")

    parser.add_argument(
        "--dev_run",
        action="store_true",
        help="If specified, run in development mode with reduced dataset.",
    )
    parser.add_argument(
        "--skip_macs",
        action="store_true",
        help="If specified, skip the macs calculation.",
    )

    return parser.parse_args()


def load_dataset_stats(dataset_stats_file):
    with open(dataset_stats_file, "r") as file:
        return json.load(file)


def calculate_average_token_lengths(data):
    average_prompt_token_length = {}
    average_output_token_length = {}

    for task in ["caption", "translation"]:
        average_prompt_token_length[task] = {}
        average_output_token_length[task] = {}
        for lang in data["total_prompt_count"][task]:
            total_prompt = data["total_prompt_count"][task][lang]
            total_output = data["total_output_count"][task][lang]
            sample_count = data["sample_count"][task][lang]
            average_prompt_token_length[task][lang] = total_prompt / sample_count
            average_output_token_length[task][lang] = total_output / sample_count

    return average_prompt_token_length, average_output_token_length


def print_average_token_lengths(average_prompt_token_length, average_output_token_length):
    for task, langs in average_prompt_token_length.items():
        for lang, _ in langs.items():
            print(
                f"{task}-{lang}",
                average_prompt_token_length[task][lang],
                average_output_token_length[task][lang],
            )


def create_transform(image_size):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(image_size, image_size), interpolation=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def setup_model(model_name, pretrained_path):
    return prepare_model(model_name, Path("."), pretrained_path)


def count_trainable_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    return total_params


def measure_model_complexity(model, training_image_size, training_batch_size, training_steps, device, torch_dtype):
    input = (
        torch.zeros(1, 128, dtype=torch.long).to(device),
        torch.zeros(1, 3, training_image_size, training_image_size, dtype=torch_dtype).to(device),
        None,
        torch.zeros(1, 128, dtype=torch.long).to(device),
    )
    report = {}
    report["model_macs"] = FlopCountAnalysis(model, input).total()
    report["model_params"] = parameter_count(model)[""]
    report["trainable_model_params"] = count_trainable_parameters(model)
    report["training_seen_samples"] = training_steps * training_batch_size
    report["forward_backward_factor"] = 1 + report["trainable_model_params"] / report["model_params"]
    report["training_macs"] = report["training_seen_samples"] * report["model_macs"] * report["forward_backward_factor"]
    return report


def evaluate_model_on_datasets(model, processor, eval_datasets, batch_size, dev_run, device, torch_dtype):
    results = {}
    data_collator = PadDataCollator()
    encoder_pad_token_id = processor.tokenizer.pad_token_id
    if processor.encoder_tokenizer is not None:
        encoder_pad_token_id = processor.encoder_tokenizer.pad_token_id
    data_collator.encoder_pad_token_id = encoder_pad_token_id
    data_collator.max_vocb_size = model.config.vocab_size

    for key in eval_datasets.keys():
        print(f"Eval {key} ...")
        losses = []
        dataloader = DataLoader(eval_datasets[key], batch_size=batch_size, collate_fn=data_collator)
        for b in tqdm(dataloader):
            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            decoder_input_ids = b["labels"].to(device)
            decoder_attention_mask = decoder_input_ids != -100
            decoder_input_ids[decoder_input_ids == -100] = processor.tokenizer.pad_token_id
            pixel_values = b["pixel_values"].to(device, torch_dtype)

            # Append decoder start token id
            decoder_start_token_tensor = torch.tensor(
                [[processor.tokenizer.eos_token_id]] * decoder_input_ids.size(dim=0)
            ).to(device)
            decoder_input_ids = torch.cat([decoder_start_token_tensor, decoder_input_ids], dim=1)
            decoder_attention_mask = torch.cat(
                [
                    torch.ones(decoder_start_token_tensor.size(), dtype=torch.int64).to(device),
                    decoder_attention_mask,
                ],
                dim=1,
            )

            with torch.no_grad():
                # Predict logits
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    pixel_values=pixel_values,
                )
                logits = outputs.logits

                # Calculate loss
                loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = decoder_input_ids[..., 1:].contiguous()
                shift_attention_mask_batch = decoder_attention_mask[..., 1:].contiguous()
                loss = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(
                    1
                ) / shift_attention_mask_batch.sum(1)

            losses.extend(loss.cpu().tolist())

            if dev_run:
                break

        results[key] = sum(losses) / len(losses), len(losses)
    return results


def summarize_results(results):
    mixup = {
        "mean": [
            "en_caption",
            "de_caption",
            "es_caption",
            "zh_caption",
            "es_translation",
            "zh_translation",
        ],
        "unseen": ["es_caption", "zh_caption"],
        "ood": ["es_translation", "zh_translation"],
        "id": ["en_caption", "de_caption"],
    }
    report = {}

    for k, v in mixup.items():
        p, c = 0, 0
        for e in v:
            loss, count = results[e]
            p += loss * count
            c += count
        report[k] = p / c

    for k, v in results.items():
        p, c = v
        report[k] = p

    return report


def main():
    args = parse_arguments()

    transform = create_transform(args.image_size)
    model, processor = setup_model(args.model_name, args.pretrained_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if "cuda" in device else torch.float32
    model.to(device).to(torch_dtype)
    model.eval()

    report = {"model_name": args.model_name, "checkpoint_path": args.pretrained_path}

    if not args.skip_macs:
        report.update(
            measure_model_complexity(
                model,
                args.training_image_size,
                args.training_batch_size,
                args.training_steps,
                device,
                torch_dtype,
            )
        )
        print(report)
        print(report["training_macs"] / 10**9, "TMAC")

    eval_dataset_paths = {
        "en_caption": (
            Path(args.eval_dataset_root_path) / "cc12m_en_caption/cc12m-train-2175.tar",
            4396,
        ),
        "de_caption": (
            Path(args.eval_dataset_root_path) / "cc12m_de_caption/cc12m-train-2175.tar",
            3086,
        ),
        "es_translation": (
            Path(args.eval_dataset_root_path) / "cc12m_es_translation/cc12m-train-2175.tar",
            4080,
        ),
        "zh_translation": (
            Path(args.eval_dataset_root_path) / "cc12m_zh_translation/cc12m-train-2175.tar",
            3786,
        ),
        "es_caption": (
            Path(args.eval_dataset_root_path) / "cc12m_es_caption/cc12m-train-2175.tar",
            4396,
        ),
        "zh_caption": (
            Path(args.eval_dataset_root_path) / "cc12m_zh_caption/cc12m-train-2175.tar",
            3447,
        ),
    }

    eval_datasets = create_eval_datasets(
        eval_dataset_paths,
        transform,
        args.max_encoder_seq_length,
        args.max_decoder_seq_length,
        processor,
        insert_lang_token=args.insert_lang_token,
        no_bos_token=args.no_bos_token,
    )

    results = evaluate_model_on_datasets(
        model,
        processor,
        eval_datasets,
        args.batch_size,
        args.dev_run,
        device,
        torch_dtype,
    )

    report.update(summarize_results(results))
    print(report)

    with open(Path(args.pretrained_path) / "report.json", "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
