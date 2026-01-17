import glob
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
import sacrebleu
import torch
import torchvision
import webdataset as wds
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorMixin

from eval import Multi30kDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

is_main_process = False


class NoAccelerateTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        return DataLoader(
            dataset,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Source:  https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self._eval_dataloaders[dataloader_key]

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=0,  # self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}
        return dataloader


class DummyDataset(Dataset):
    def __init__(
        self,
        num_samples,
        max_encoder_seq_length,
        max_decoder_seq_length,
        image_shape=(3, 224, 224),
        pad_token_id=0,
    ):
        self.num_samples = num_samples
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length
        self.image_shape = image_shape
        self.pad_token_id = pad_token_id

        # Generate a single sample with random data
        self.sample = self._generate_sample()

    def _generate_sample(self):
        # Randomly generate IDs for the prompt
        prompt_ids = np.random.randint(1, 100, size=self.max_encoder_seq_length).tolist()
        # Randomly generate pixel values for the image
        image = np.random.rand(*self.image_shape).astype(np.float32)

        return {
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * self.max_encoder_seq_length, dtype=torch.long),
            "labels": torch.tensor([self.pad_token_id] * self.max_decoder_seq_length, dtype=torch.long),
            "pixel_values": torch.tensor(image, dtype=torch.float32),
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": torch.clone(self.sample["input_ids"]),
            "attention_mask": torch.clone(self.sample["attention_mask"]),
            "labels": torch.clone(self.sample["labels"]),
            "pixel_values": torch.clone(self.sample["pixel_values"]),
        }


def get_human_readable_count(number: int) -> str:
    """Abbreviates an integer number with K, M, B, T for thousands, millions, billions, and trillions."""
    labels = ["", "K", "M", "B", "T"]
    for idx, label in enumerate(labels):
        if number < 1000:
            return f"{number:.0f} {label}".rstrip()
        number /= 1000
    return f"{number:.0f} T"


def get_formatted_model_size(total_model_size: float) -> str:
    """Formats the model size in MB."""
    return f"{total_model_size:,.3f}"


def print_model_summary(model: torch.nn.Module):
    """
    Print a summary of the model including layer types, parameters, and size.

    Args:
        model (nn.Module): The PyTorch model to summarize.
    """

    if not is_main_process:
        return

    def count_params(layer, skip_requires_grad_check=False):
        return sum(p.numel() for p in layer.parameters() if skip_requires_grad_check or p.requires_grad)

    total_params = sum(count_params(layer, skip_requires_grad_check=True) for layer in model.children())
    trainable_params = sum(count_params(layer) for layer in model.children() if layer.parameters())
    non_trainable_params = total_params - trainable_params

    model_size = total_params * 4e-6  # Assuming 4 bytes per parameter (float32)

    print(f"{'Layer':<30} {'Param #':<15}")
    print("=" * 45)

    for layer in model.children():
        param_count = count_params(layer)
        print(f"{layer.__class__.__name__:<30} {get_human_readable_count(param_count):<15}")

    print("=" * 45)
    print(f"{get_human_readable_count(trainable_params):<15} Trainable params")
    print(f"{get_human_readable_count(non_trainable_params):<15} Non-trainable params")
    print(f"{get_human_readable_count(total_params):<15} Total params")
    print(f"{get_formatted_model_size(model_size):<15} Total estimated model params size (MB)")


def print_on_main_rank(*args):
    if is_main_process:
        print(*args)


def _apply_transform(sample, transform):
    if transform is None:
        return sample
    image, data = sample
    image = transform(image)
    return image, data


def _select_sample(sample):
    image, data = sample
    selected_entry = data[0]
    sample = dict(
        image=image,
        lang=selected_entry["lang"],
        type=selected_entry["type"],
    )
    if "prompt_ids" not in selected_entry:
        sample["prompt_ids"] = selected_entry["prompt_ids"]
        sample["output_ids"] = selected_entry["output_ids"]
    else:
        sample["prompt"] = selected_entry["prompt"]
        sample["output"] = selected_entry["output"]
    return sample


def _get_data_key(data):
    return f"{data['type']}-{data['lang']}"


def _dynamic_weighted_sampling(sample, target_distribution, counter=None):
    image, data = sample
    total_count = sum(counter.values())
    prob = [int(target_distribution[_get_data_key(d)] * total_count >= counter[_get_data_key(d)]) + 1e-6 for d in data]
    selected_entry = random.choices(data, prob)[0]
    counter[_get_data_key(selected_entry)] += 1
    sample = dict(
        image=image,
        lang=selected_entry["lang"],
        type=selected_entry["type"],
    )
    if "prompt_ids" not in selected_entry:
        sample["prompt_ids"] = selected_entry["prompt_ids"]
        sample["output_ids"] = selected_entry["output_ids"]
    else:
        sample["prompt"] = selected_entry["prompt"]
        sample["output"] = selected_entry["output"]
    return sample


def decode_json(sample):
    sample["data"] = json.loads(sample["data"].decode())
    return sample


def dummy_nodesplitter(src, group=None):
    yield from src


@dataclass
class PadDataCollator(DataCollatorMixin):
    max_vocb_size = None
    encoder_pad_token_id = None

    def __call__(self, features, return_tensors=None):
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=self.encoder_pad_token_id,
        )
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features],
            batch_first=True,
            padding_value=0,
        )

        # We only create labels, the model handles the rest
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in features],
            batch_first=True,
            padding_value=-100,  # -100 instead of pad token id so that the model ignores them in loss calculation
        )

        # Remove invalid tokens
        mask = padded_labels >= self.max_vocb_size
        if torch.any(mask):
            print(f"Warning: {mask.sum()} ignored ids, {padded_labels[mask]}, {self.max_vocb_size}")

        padded_labels[mask] = -100

        return dict(
            input_ids=padded_input_ids,
            labels=padded_labels,
            pixel_values=torch.stack([f["pixel_values"] for f in features]),
            attention_mask=padded_attention_mask,
        )


def _tokenize(
    sample,
    max_encoder_seq_length,
    max_decoder_seq_length,
    processor,
    insert_lang_token=False,
    no_bos_token=False,
):
    if "prompt_ids" in sample and "output_ids" in sample:
        labels = torch.tensor(sample["output_ids"], dtype=torch.long)
        if insert_lang_token:
            tokenizer = processor.tokenizer
            cap_token_id = tokenizer.convert_tokens_to_ids(f"<LANG_{sample['lang'].upper()}>")
            new_seq_length = labels.size()[0] + 1
            new_labels = torch.zeros((new_seq_length), dtype=torch.long)
            new_labels[0] = labels[0]
            new_labels[1] = cap_token_id
            new_labels[2:] = labels[1:]
            labels = new_labels

        if no_bos_token:
            labels = labels[1:]

        input_ids = torch.tensor(sample["prompt_ids"], dtype=torch.long)
        attention_mask = torch.tensor([1] * len(sample["prompt_ids"]), dtype=torch.long)

        return_data = {
            "input_ids": input_ids[:max_encoder_seq_length],
            "attention_mask": attention_mask[:max_encoder_seq_length],
            "labels": labels[:max_decoder_seq_length],
            "pixel_values": sample["image"],
        }
        return return_data

    assert insert_lang_token is False

    # Extract necessary attributes from the processor upfront
    tokenizer = processor.encoder_tokenizer or processor.tokenizer
    construct_prompts = processor._construct_prompts

    # Prepare prompts and texts
    prompt = construct_prompts([sample["prompt"]])
    output = [sample["output"]]

    # Tokenize prompts
    prompt_data = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=447,  # 1024 - 577
        return_token_type_ids=False,
    )

    # Tokenize texts for decoder labels
    output_data = processor.tokenizer(
        output,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
        return_token_type_ids=False,
    )

    labels = output_data["input_ids"][0]

    if no_bos_token:
        labels = labels[1:]

    input_ids = prompt_data["input_ids"][0]
    attention_mask = torch.ones_like(input_ids)

    # Directly assign the first elements
    return_data = {
        "input_ids": input_ids[:max_encoder_seq_length],
        "attention_mask": attention_mask[:max_encoder_seq_length],
        "labels": labels[:max_decoder_seq_length],
        "pixel_values": sample["image"],
    }

    return return_data


def get_balancing_data(balancing_type):
    if balancing_type == "task-lang-balancing":
        target_distribution = {
            "caption-de": 1,
            "caption-en": 1,
            "translation-de": 1,
            "translation-es": 1,
            "translation-fr": 1,
            "translation-zh": 1,
            "translation-ru": 1,
        }
    elif balancing_type == "task-balancing":
        target_distribution = {
            "caption-de": 5,
            "caption-en": 5,
            "translation-de": 2,
            "translation-es": 2,
            "translation-fr": 2,
            "translation-zh": 2,
            "translation-ru": 2,
        }
    elif balancing_type == "lang-balancing":
        target_distribution = {
            "caption-de": 1,
            "caption-en": 2,
            "translation-de": 1,
            "translation-es": 2,
            "translation-fr": 2,
            "translation-zh": 2,
            "translation-ru": 2,
        }
    elif balancing_type == "generalist_finetune":
        target_distribution = {
            "caption-de": 1,
            "caption-en": 1,
            "caption-es": 1,
            "caption-fr": 1,
            "caption-zh": 1,
            "caption-ru": 1,
            "translation-de": 1,
            "translation-es": 1,
            "translation-fr": 1,
            "translation-zh": 1,
            "translation-ru": 1,
        }
    else:
        raise Exception(f"Balancing type {balancing_type} does not exist!")

    # Normalize target distribution
    target_distribution = {
        label: target_distribution[label] / (sum(target_distribution.values()) + 1e-6)
        for label in target_distribution.keys()
    }
    return target_distribution


def create_train_dataset(
    train_dataset_path,
    transform,
    balancing_type,
    max_encoder_seq_length,
    max_decoder_seq_length,
    processor,
    insert_lang_token,
    no_bos_token,
):
    # Get target dist
    target_distribution = get_balancing_data(balancing_type)
    print_on_main_rank(f"Train balancing method {balancing_type} with distribution {target_distribution} is active.")

    dataset = (
        wds.WebDataset(
            str(train_dataset_path),
            resampled=True,
            nodesplitter=wds.split_by_node,
            handler=wds.warn_and_continue,
        )
        .shuffle(1000)
        .decode("pil", handler=wds.warn_and_continue)
        .map(decode_json, handler=wds.warn_and_continue)
        .to_tuple("jpg", "data", handler=wds.warn_and_continue)
        .map(
            partial(_apply_transform, transform=transform),
            handler=wds.warn_and_continue,
        )
        .map(
            partial(
                _dynamic_weighted_sampling,
                target_distribution=target_distribution,
                counter=Counter(),
            ),
            handler=wds.warn_and_continue,
        )
        .map(
            partial(
                _tokenize,
                max_encoder_seq_length=max_encoder_seq_length,
                max_decoder_seq_length=max_decoder_seq_length,
                processor=processor,
                insert_lang_token=insert_lang_token,
                no_bos_token=no_bos_token,
            ),
            handler=wds.warn_and_continue,
        )
    )
    return dataset


def create_eval_datasets(
    eval_dataset_paths,
    transform,
    max_encoder_seq_length,
    max_decoder_seq_length,
    processor,
    insert_lang_token,
    no_bos_token,
):
    eval_datasets = {}
    for key, (eval_dataset_path, length) in eval_dataset_paths.items():
        eval_dataset = (
            wds.WebDataset(
                str(eval_dataset_path),
                resampled=False,
                nodesplitter=dummy_nodesplitter,
                shardshuffle=False,
            )
            .decode("pil")
            .map(decode_json)
            .to_tuple("jpg", "data")
            .map(partial(_apply_transform, transform=transform))
            .map(_select_sample)
            .map(
                partial(
                    _tokenize,
                    max_encoder_seq_length=max_encoder_seq_length,
                    max_decoder_seq_length=max_decoder_seq_length,
                    processor=processor,
                    insert_lang_token=insert_lang_token,
                    no_bos_token=no_bos_token,
                )
            )
        )
        if length is None:
            length = sum(1 for _ in iter(eval_dataset))
        print_on_main_rank(key, length)
        eval_datasets[key] = eval_dataset.with_length(length)
    return eval_datasets


def prepare_model(
    model_name,
    init_root_path,
    pretrained_path=None,
    freeze_vision_encoder=True,
    freeze_encoder=False,
    freeze_decoder=True,
    freeze_decoder_head_and_embeddings=False,
):
    from caption_via_translation.process.processing_florence2 import Florence2Processor

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b",
        add_bos_token=True,
        add_eos_token=True,
        padding_side="right",
        truncation_side="right",
    )
    processor = Florence2Processor.from_pretrained(
        "microsoft/Florence-2-large",
        new_tokenizer=tokenizer,
        use_encoder_tokenizer=True,
    )

    print(model_name)

    model_name = model_name.lower().strip()

    if model_name == "florence-gemma-2-large" or model_name == "florence-gemma-2-xlarge":
        from caption_via_translation.models.gemma2 import (
            FlorenceGemma2Config,
            FlorenceGemma2ForConditionalGeneration,
        )

        config_cls = FlorenceGemma2Config
        model_cls = FlorenceGemma2ForConditionalGeneration
        if model_name == "florence-gemma-2-large":
            init_path = "data/pretrained/FlorenceGemma2-large/"
        else:
            init_path = "data/pretrained/FlorenceGemma2-xlarge/"

    else:
        from caption_via_translation.models.florence2 import (
            Florence2Config,
            Florence2ForConditionalGeneration,
        )

        config_cls = Florence2Config
        model_cls = Florence2ForConditionalGeneration

        if model_name == "microsoft/florence-2-base":
            init_path = "data/pretrained/Florence-2-base/"
        else:
            init_path = "data/pretrained/Florence-2-large/"

    init_path = Path(init_path)

    if pretrained_path is not None:
        print_on_main_rank(f"Load pretrained model from {pretrained_path} ...")
        model = model_cls.from_pretrained(pretrained_path)
    else:
        config = config_cls.from_json_file(init_path / "config.json")
        model = model_cls(config)
        state_dict = torch.load(init_root_path / init_path / "init_state_dict.pt", weights_only=False)
        print_on_main_rank(model.load_state_dict(state_dict, False))

    if freeze_vision_encoder:
        print_on_main_rank("Freezing vision tower of Florence-2 ...")
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        model.image_projection.requires_grad = False
        for param in model.image_pos_embed.parameters():
            param.requires_grad = False
        for param in model.image_proj_norm.parameters():
            param.requires_grad = False

    if freeze_decoder:
        if model_name.startswith("florence-gemma-2") or model_name.startswith("florence-gemmax-2"):
            print_on_main_rank("Freezing decoder of Florence-Gemma-2 ...")
            substrings_to_train = [
                "post_encoder_attention_layernorm",
                "encoder_attn_gate",
                "encoder_attn",
                "embed_tokens",
            ]
            for name, param in model.language_model.model.decoder.named_parameters():
                if not any(substring in name for substring in substrings_to_train):
                    param.requires_grad = False
        elif model_name.startswith("microsoft/florence"):
            print_on_main_rank("Freezing decoder of Florence-2 ...")
            substrings_to_train = [
                "embed_tokens",
            ]
            for name, param in model.language_model.model.decoder.named_parameters():
                if not any(substring in name for substring in substrings_to_train):
                    param.requires_grad = False

    if freeze_decoder_head_and_embeddings:
        print_on_main_rank("Freezing decoder head and embeddings ...")
        model.language_model.lm_head.weight.requires_grad = False
        model.language_model.model.decoder.embed_tokens.weight.requires_grad = False
    else:
        model.language_model.lm_head.weight.requires_grad = True
        model.language_model.model.decoder.embed_tokens.weight.requires_grad = True

    return model, processor


def random_init_model():
    # Capture the initial weights of the decoder
    initial_weights = {name: param.clone() for name, param in model.language_model.model.decoder.named_parameters()}

    def _init_weights(module):
        std = model.language_model.model.decoder.config.init_std
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.weight, 1.0)
            torch.nn.init.constant_(module.bias, 0)

    # Apply the custom initialization to the decoder
    model.language_model.model.decoder.apply(_init_weights)

    # Check if all weights have changed
    all_weights_changed = True
    for name, param in model.language_model.model.decoder.named_parameters():
        if torch.equal(initial_weights[name], param):
            print_on_main_rank(f"Weight {name} did not change.")
            all_weights_changed = False

    if all_weights_changed:
        print_on_main_rank("All weights have been reinitialized.")
    else:
        print_on_main_rank("Some weights were not reinitialized.")

    del initial_weights


# Define a callback that runs after evaluation.
class CaptionEvalCallback(TrainerCallback):
    def __init__(self, trainer, processor, batch_size=8, metric_prefix=None, **dataset_kwargs):
        self.trainer = trainer
        self.processor = processor
        self.batch_size = batch_size
        self.dataset_kwargs = dataset_kwargs
        self.metric_prefix = metric_prefix

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = self.trainer.model
        dataset = Multi30kDataset(**self.dataset_kwargs)
        lang = self.dataset_kwargs["lang"]

        device, torch_dtype = model.device, model.dtype

        def collate_fn(data):
            images, img_paths, src, tgt = zip(*data)
            src = [s[0].strip() for s in src]
            tgt = [t[0].strip() for t in tgt]
            prompt = [f"<LANG_{lang.upper()}><TRANSLATE>" + s for s in src]
            inputs = self.processor(
                prompt,
                images=images,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            return inputs, src, tgt

        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        sources, predictions, references = [], [], []
        for inputs, src, tgt in tqdm(dataloader, desc=f"Evaluating Multi30k Translation ({lang})"):
            with torch.no_grad(), torch.inference_mode():
                inputs.to(device, torch_dtype)
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=128,
                    num_beams=4,
                    do_sample=False,
                    use_cache=False,
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            sources += src
            predictions += generated_text
            references += tgt
        bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="zh" if lang == "zh" else "13a").score
        self.trainer.log(
            {
                f"eval_{self.metric_prefix}_bleu": bleu,
                "eval_predictions": predictions[:10],
                "eval_references": references[:10],
            }
        )
        return control


if __name__ == "__main__":
    # TODO move these args to the arg parser
    train_dataset_path = os.environ["TRAIN_DATASET_PATH"]
    eval_dataset_root_path = os.environ["EVAL_DATASET_ROOT_PATH"]
    model_name = os.environ.get("MODEL_NAME", "microsoft/Florence-2-large")
    use_pretrained_decoder = os.environ.get("USE_PRETRAINED_DECODER", "true").lower() in ("true", "1", "t")
    tie_head = os.environ.get("TIE_HEAD", "true").lower() in ("true", "1", "t")
    max_encoder_seq_length = int(os.environ.get("MAX_ENCODER_SEQ_LENGTH", 447))
    max_decoder_seq_length = int(os.environ.get("MAX_DECODER_SEQ_LENGTH", 512))
    image_size = int(os.environ.get("IMAGE_SIZE", 224))
    use_dummy_dataset = os.environ.get("USE_DUMMY_DATASET", "false").lower() in (
        "true",
        "1",
        "t",
    )
    pretrained_path = os.environ.get("PRETRAINED_PATH", None)
    insert_lang_token = os.environ.get("INSERT_LANG_TOKEN", "false").lower() in (
        "true",
        "1",
        "t",
    )
    balancing_type = os.environ.get("BALANCING_TYPE", "lang-balancing")
    init_root_path = Path(os.environ.get("INIT_ROOT_PATH", "."))
    is_generalist_finetune = os.environ.get("IS_GENERALIST_FINETUNE", "false").lower() in ("true", "1", "t")
    multi30k_paths = os.environ.get("MULTI30K_PATHS", None)
    no_bos_token = os.environ.get("NO_BOS_TOKEN", "false").lower() in ("true", "1", "t")

    if not os.path.exists(train_dataset_path[: train_dataset_path.rfind("/")]):
        raise Exception(f"Train path {train_dataset_path} does not exist.")
    if not is_generalist_finetune and not os.path.exists(eval_dataset_root_path):
        raise Exception(f"Val path {eval_dataset_root_path} does not exist.")
    if is_generalist_finetune and not os.path.exists(eval_dataset_root_path[: eval_dataset_root_path.rfind("/")]):
        raise Exception(f"Val path {eval_dataset_root_path} does not exist.")

    if not is_generalist_finetune:
        eval_dataset_root_path = Path(eval_dataset_root_path)

    # Parse trainer arguments
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()[0]
    training_args.custom_args = dict(
        train_dataset_path=train_dataset_path,
        eval_dataset_root_path=eval_dataset_root_path,
        model_name=model_name,
        use_pretrained_decoder=use_pretrained_decoder,
        tie_head=tie_head,
        max_encoder_seq_length=max_encoder_seq_length,
        max_decoder_seq_length=max_decoder_seq_length,
        image_size=image_size,
        use_dummy_dataset=use_dummy_dataset,
    )
    is_main_process = training_args.should_log

    # Create transform
    transform = torchvision.transforms.Compose(
        transforms=[
            torchvision.transforms.Resize(size=(image_size, image_size), interpolation=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model_kwargs = dict(
        freeze_vision_encoder=True,
        freeze_encoder=False,
        freeze_decoder=True,
        freeze_decoder_head_and_embeddings=False,
    )
    if model_name.lower().startswith("microsoft/florence"):
        model_kwargs["freeze_decoder"] = False

    if is_generalist_finetune:
        model_kwargs["freeze_vision_encoder"] = True
        model_kwargs["freeze_encoder"] = False
        model_kwargs["freeze_decoder"] = False
        model_kwargs["freeze_decoder_head_and_embeddings"] = False

    if not model_name.lower().startswith("microsoft/florence") and is_generalist_finetune:
        model_kwargs["freeze_decoder"] = True

    if no_bos_token:
        model_kwargs["freeze_decoder_head_and_embeddings"] = True

    # Setup model
    model, processor = prepare_model(model_name, init_root_path, pretrained_path, **model_kwargs)

    if no_bos_token:
        model.language_model.config.decoder_start_token_id = model.language_model.config.bos_token_id

    # Setup data collator
    data_collator = PadDataCollator()

    # Extract pad token ids for encoder and decoder
    encoder_pad_token_id = processor.tokenizer.pad_token_id
    if processor.encoder_tokenizer is not None:
        encoder_pad_token_id = processor.encoder_tokenizer.pad_token_id
    pad_token_id = processor.tokenizer.pad_token_id
    data_collator.encoder_pad_token_id = encoder_pad_token_id
    data_collator.max_vocb_size = model.config.vocab_size

    # Print model stats
    print_model_summary(model)

    # Setup datasets
    if not use_dummy_dataset:
        # Create the train dataset
        dataset = create_train_dataset(
            train_dataset_path=train_dataset_path,
            transform=transform,
            balancing_type=balancing_type if not is_generalist_finetune else "generalist_finetune",
            max_encoder_seq_length=max_encoder_seq_length,
            max_decoder_seq_length=max_decoder_seq_length,
            processor=processor,
            insert_lang_token=insert_lang_token,
            no_bos_token=no_bos_token,
        )
        if not is_generalist_finetune:
            eval_dataset_paths = {
                "en_caption": (
                    eval_dataset_root_path / "cc12m_en_caption/cc12m-train-2175.tar",
                    4396,
                ),
                "de_caption": (
                    eval_dataset_root_path / "cc12m_de_caption/cc12m-train-2175.tar",
                    3086,
                ),
                "es_translation": (
                    eval_dataset_root_path / "cc12m_es_translation/cc12m-train-2175.tar",
                    4080,
                ),
                "zh_translation": (
                    eval_dataset_root_path / "cc12m_zh_translation/cc12m-train-2175.tar",
                    3786,
                ),
                "es_caption": (
                    eval_dataset_root_path / "cc12m_es_caption/cc12m-train-2175.tar",
                    4396,
                ),
                "zh_caption": (
                    eval_dataset_root_path / "cc12m_zh_caption/cc12m-train-2175.tar",
                    3447,
                ),
            }
        else:
            eval_dataset_paths = {
                "generalist_val_set": (eval_dataset_root_path, 11154),
            }
        eval_datasets = create_eval_datasets(
            eval_dataset_paths,
            transform,
            max_encoder_seq_length,
            max_decoder_seq_length,
            processor,
            insert_lang_token=insert_lang_token,
            no_bos_token=no_bos_token,
        )
    else:
        # For testing the training setup
        print_on_main_rank("Creating dummy dataset ...")
        dataset = DummyDataset(
            training_args.per_device_train_batch_size * training_args.world_size * 10,
            max_encoder_seq_length=max_encoder_seq_length,
            max_decoder_seq_length=max_decoder_seq_length,
            pad_token_id=encoder_pad_token_id,
            image_shape=(3, image_size, image_size),
        )
        eval_datasets = None
        training_args.gradient_accumulation_steps = 1
        training_args.max_steps = 10
        training_args.eval_strategy = "no"
        training_args.save_strategy = "no"

    # Create trainer
    trainer = NoAccelerateTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_datasets,
        data_collator=data_collator,
    )

    # Add BLEU Multi30k task1 evaluation for generalist finetune setting
    if is_generalist_finetune and multi30k_paths is not None:
        dataset_path, flickr30k_images_path, coco_images_path = multi30k_paths.split(",")
        trainer.add_callback(
            CaptionEvalCallback(
                trainer=trainer,
                processor=processor,
                batch_size=8,
                metric_prefix="multi30k_de",
                **{
                    "dataset_path": dataset_path,
                    "flickr30k_images_path": flickr30k_images_path,
                    "coco_images_path": coco_images_path,
                    "task": "task1",
                    "split": "val",
                    "lang": "de",
                },
            )
        )
        trainer.add_callback(
            CaptionEvalCallback(
                trainer=trainer,
                processor=processor,
                batch_size=8,
                metric_prefix="multi30k_fr",
                **{
                    "dataset_path": dataset_path,
                    "flickr30k_images_path": flickr30k_images_path,
                    "coco_images_path": coco_images_path,
                    "task": "task1",
                    "split": "val",
                    "lang": "fr",
                },
            )
        )

    # FIXME this check is required because of our loss implementation
    assert not trainer.model_accepts_loss_kwargs

    # Check if the base folder exists
    auto_resume = False
    if os.path.isdir(trainer.args.output_dir):
        # Use glob to find subfolders matching the pattern "checkpoint-*"
        subfolder_pattern = os.path.join(training_args.output_dir, "checkpoint-*")
        subfolders = glob.glob(subfolder_pattern)

        if subfolders:
            auto_resume = True
            print_on_main_rank("Autoresume is active ...")

    # Start training
    result = trainer.train(resume_from_checkpoint=auto_resume)
    print_on_main_rank(result)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
