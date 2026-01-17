import json
import math
from pathlib import Path

import torch
import transformers
from tqdm import tqdm

from caption_via_translation.models.florence2 import (
    Florence2Config,
    Florence2ForConditionalGeneration,
)
from caption_via_translation.models.gemma2 import Gemma2ForCausalLM
from caption_via_translation.process.processing_florence2 import Florence2Processor


def prepare_florence2(model_name):
    output_folder = Path("data/pretrained/") / model_name / "init_state_dict.pt"
    if output_folder.exists():
        print(f"Model {model_name} is already preprocessed.")
        return

    output_folder.parent.mkdir(parents=True, exist_ok=True)

    print(f"Preprocess {model_name} ...")
    tokenizer_matching = get_tokenizer_matching_from_florence2_to_gemma2()

    vocab_size = 51328  # 51289

    # The encoder vocab site will remain constant
    encoder_vocab_size = vocab_size

    # Make multiple of 128 and set to new vocab size
    vocab_size = math.ceil(len(tokenizer_matching) / 128) * 128
    vocab_size = 257024

    # Load florence2 model
    config = Florence2Config.from_pretrained("microsoft/" + model_name)

    # Load the unmodified model to get the weights
    model = Florence2ForConditionalGeneration.from_pretrained(
        "microsoft/" + model_name, config=config, ignore_mismatched_sizes=True
    )

    # Lets load the default embedding tokens and lm head
    default_embed_tokens = model.language_model.model.shared.weight.data
    default_lm_head = model.language_model.lm_head.weight.data
    del model

    # Update config
    pad_token_id, bos_token_id, eos_token_id = 0, 2, 1
    config_update = dict(
        use_encoder_tokenizer=True,  # Use a separate encoder tokenizer
        vocab_size=vocab_size,
        encoder_vocab_size=encoder_vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        forced_eos_token_id=eos_token_id,
        forced_bos_token_id=bos_token_id,
        decoder_start_token_id=eos_token_id,
    )
    config.update(config_update)
    config.text_config.update(config_update)

    # Load the model agian with updated config
    model = Florence2ForConditionalGeneration.from_pretrained(
        "microsoft/" + model_name, config=config, ignore_mismatched_sizes=True
    )

    # Update generation config
    generation_config_update = dict(
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        decoder_start_token_id=eos_token_id,
        forced_eos_token_id=eos_token_id,
        forced_bos_token_id=bos_token_id,
    )
    model.language_model.generation_config.update(**generation_config_update)
    model.generation_config.update(**generation_config_update)

    # Init weights of new parameters
    std = config.text_config.init_std
    for module in [
        model.language_model.model.encoder.embed_tokens,
        model.language_model.model.decoder.embed_tokens,
        model.language_model.lm_head,
    ]:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    print("Tie decoder embeddings and lm head ...")
    model._tie_or_clone_weights(
        model.language_model.model.decoder.embed_tokens,
        model.language_model.lm_head,
    )

    print("Remapping weights for encoder tokenizer mode ...")

    def log_remapping(weight_name, weight, prev_weight):
        print(
            f"- {weight_name}: Copy from {prev_weight.shape} (mean:{prev_weight.mean().item():.03f}, max:{prev_weight.max().item():.03f}) to {weight.data.shape} (mean:{weight.data.mean().item():.03f}, max:{weight.data.max().item():.03f})"
        )

    # Encoder embeds stay the same
    model.language_model.model.encoder.embed_tokens.weight.data[: default_embed_tokens.shape[0], :] = (
        default_embed_tokens
    )
    log_remapping(
        "language_model.model.encoder.embed_tokens.weight",
        model.language_model.model.encoder.embed_tokens.weight,
        default_embed_tokens,
    )

    # Group by length
    matching_dict_by_length = {}
    for k, v in tokenizer_matching.items():
        if len(v) not in matching_dict_by_length:
            matching_dict_by_length[len(v)] = {}
        matching_dict_by_length[len(v)][int(k)] = v

    # For each matching size
    # This grouping by number of src tokens allows for more efficent processing
    for l, l_matching_dict in matching_dict_by_length.items():
        # Build the index tensors for new and old weights
        num_new_indices = len(l_matching_dict)
        new_indices = torch.zeros(num_new_indices, dtype=torch.long)
        old_indices = torch.zeros((num_new_indices, l), dtype=torch.long)
        for i, (k, v) in enumerate(l_matching_dict.items()):
            new_indices[i] = k
            old_indices[i] = torch.tensor(v, dtype=torch.long)

        old_indices = old_indices.flatten()

        contributing_head_weights = torch.index_select(default_lm_head, 0, old_indices).reshape(
            (num_new_indices, -1, default_lm_head.shape[-1])
        )
        contributing_embedding_weights = torch.index_select(default_embed_tokens, 0, old_indices).reshape(
            (num_new_indices, -1, default_embed_tokens.shape[-1])
        )

        assert contributing_head_weights.shape[1] == l
        assert contributing_embedding_weights.shape[1] == l

        model.language_model.lm_head.weight.data[new_indices] = contributing_head_weights.mean(dim=1)
        model.language_model.model.decoder.embed_tokens.weight.data[new_indices] = contributing_embedding_weights.mean(
            dim=1
        )

    log_remapping(
        "language_model.lm_head.weight",
        model.language_model.lm_head.weight,
        default_lm_head,
    )
    log_remapping(
        "language_model.model.decoder.embed_tokens.weight",
        model.language_model.model.decoder.embed_tokens.weight,
        default_embed_tokens,
    )

    state_dict = model.state_dict()
    torch.save(state_dict, output_folder)
    print(f"Preprocessing results for {model_name} saved to {output_folder}.")


def prepare_gemma2(model_name):
    model_name_map = {
        "FlorenceGemma2-large": "google/gemma-2-2b",
        "FlorenceGemma2-xlarge": "google/gemma-2-9b",
    }
    output_folder = Path("data/pretrained/") / model_name / "init_state_dict.pt"
    if output_folder.exists():
        print(f"Model {model_name} is already preprocessed.")
        return

    output_folder.parent.mkdir(parents=True, exist_ok=True)

    print(f"Preprocess {model_name} ...")

    # Extract relevant keys from gemma2
    gemma2_state_dict = Gemma2ForCausalLM.from_pretrained(model_name_map[model_name]).state_dict()
    updated_state_dict = {}
    for k, v in gemma2_state_dict.items():
        if "lm_head." in k:
            k = k.replace("lm_head.", "language_model.lm_head.")
        else:
            k = k.replace("model.", "language_model.model.decoder.")

        if "post_attention_layernorm." in k:
            updated_state_dict[k.replace("post_attention_layernorm.", "post_encoder_attention_layernorm.")] = v

        updated_state_dict[k] = v
    del gemma2_state_dict

    # Extract relevant keys from florence2
    florence2_state_dict = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-large").state_dict()
    for k, v in florence2_state_dict.items():
        if "language_model.model.decoder." in k or "language_model.lm_head." in k:
            continue
        if "language_model.final_logits_bias" in k:
            continue
        if "language_model.model.shared.weight" in k:
            continue
        updated_state_dict[k.replace("lm_head.", "language_model.lm_head.")] = v
    del florence2_state_dict

    torch.save(updated_state_dict, output_folder)
    print(f"Preprocessing results for {model_name} saved to {output_folder}.")


def get_tokenizer_matching_from_florence2_to_gemma2():
    tokenizer_matching_file = Path("data/florence2_gemma2_tokenizer_matching.json")

    if tokenizer_matching_file.exists():
        with open(tokenizer_matching_file) as f:
            return json.load(f)

    print("Match tokens from gemma2 to tokens from florence2 ...")

    processor = Florence2Processor.from_pretrained("microsoft/Florence-2-base")

    new_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "google/gemma-2-2b",
        add_bos_token=True,
        add_eos_token=True,
        padding_siden="right",
        truncation_siden="right",
    )
    print(new_tokenizer.all_special_tokens)
    print(processor.tokenizer.all_special_tokens[:10])

    matching = {}
    matching[new_tokenizer.convert_tokens_to_ids("<bos>")] = [processor.tokenizer.convert_tokens_to_ids("<s>")]
    matching[new_tokenizer.convert_tokens_to_ids("<eos>")] = [processor.tokenizer.convert_tokens_to_ids("</s>")]
    matching[new_tokenizer.convert_tokens_to_ids("<unk>")] = [processor.tokenizer.convert_tokens_to_ids("<unk>")]
    matching[new_tokenizer.convert_tokens_to_ids("<pad>")] = [processor.tokenizer.convert_tokens_to_ids("<pad>")]

    # gemma2 token -> [florence2 tokens, ...]
    for k in tqdm(new_tokenizer.get_vocab()):
        token_id = new_tokenizer.convert_tokens_to_ids(k)
        if token_id in matching:
            continue
        if k[0] == "▁":
            k = " " + k[1:]
        inputs = processor.tokenizer(k, add_special_tokens=True)
        inputs["input_ids"] = inputs["input_ids"][1:-1]
        matching[token_id] = inputs["input_ids"]

    print(f"Save tokenizer matching to {tokenizer_matching_file}")
    with open(tokenizer_matching_file, "w") as f:
        json.dump(matching, f)
    return matching


def main():
    prepare_florence2(model_name="Florence-2-base")
    prepare_florence2(model_name="Florence-2-large")
    prepare_gemma2(model_name="FlorenceGemma2-large")
    prepare_gemma2(model_name="FlorenceGemma2-xlarge")


if __name__ == "__main__":
    main()
