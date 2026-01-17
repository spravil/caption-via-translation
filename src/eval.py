from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import string
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import DefaultDict, Dict, List, Union

import evaluate
import numpy as np
import open_clip
import sacrebleu
import stanza
import torch
import torch._dynamo as dynamo
import transformers
from comet import download_model, load_from_checkpoint
from multilingual_clip import pt_multilingual_clip
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from caption_via_translation.models.florence2 import Florence2ForConditionalGeneration
from caption_via_translation.models.gemma2 import FlorenceGemma2ForConditionalGeneration
from caption_via_translation.process.processing_florence2 import Florence2Processor

os.environ["TORCHDYNAMO_CACHE_SIZE_LIMIT"] = "99999"
dynamo.config.recompile_limit = 32
torch.set_float32_matmul_precision("medium")
os.environ["TOKENIZERS_PARALLELISM"] = "0"
os.environ["WANDB_MODE"] = "online"


METEOR_JAR = "meteor-1.5.jar"

DEV_RUN = False


def assert_meteor_jar_exists() -> Path:
    here = Path(__file__).resolve().parent
    jar_path = here / METEOR_JAR

    if not jar_path.is_file():
        print(
            "\nERROR: Missing 'meteor-1.5.jar' next to this script.\n\n"
            "Tutorial (run from your project's src folder):\n"
            "  wget https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz\n"
            "  tar -xvf meteor-1.5.tar.gz meteor-1.5/meteor-1.5.jar meteor-1.5/data --strip-components=1\n"
            "  rm meteor-1.5.tar.gz\n\n"
            f"Expected location:\n  {jar_path}\n"
        )
        sys.exit(1)


class CLIPDataset(Dataset):
    """Support class for managing inputs of clipscore"""

    def __init__(self, images, captions, preprocess, tokenizer=None):
        self.images = images
        self.captions = captions
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_path = self.images[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)

        if self.tokenizer is not None:
            cap_token = self.tokenizer(caption)[0]
            return image, cap_token
        else:
            return image, caption


class ClipScore:
    """
    Implementation of ClipScore, introduced in the paper
    CLIPScore: A Reference-free Evaluation Metric for Image Captioning
    https://arxiv.org/pdf/2104.08718

    Basically a scaled cosine similarity between two embeddings (image and text pair) produced by a pretrained CLIP.
    """

    def __init__(
        self,
        clip_name: str = "ViT-B-32",
        pretrained: str = "openai",
        num_workers: int = 8,
        batch_size: int = 8,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if clip_name.startswith("M-CLIP"):
            self.model = MClipWrapper(clip_name, self.device)
            self.preprocess = self.model.img_preprocess
            self.tokenizer = None  # Explicit None since tokenizer is embbeded into mclip's text encoder
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_name, pretrained=pretrained, device=self.device
            )
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(clip_name)

        self.num_workers = num_workers
        self.batch_size = batch_size
        print("Load models")

    def compute_score(self, images, captions):
        ds = CLIPDataset(images, captions, self.preprocess, self.tokenizer)
        dataloader = DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        scores = []
        with torch.no_grad():
            for images, captions in tqdm(dataloader):
                if not isinstance(captions, tuple):
                    captions = captions.to(self.device)

                image_features = self.model.encode_image(images.to(self.device)).cpu().numpy()
                captions_features = self.model.encode_text(captions).cpu().numpy()
                scores += self.cal_clipscore(image_features, captions_features).tolist()

        clip_score = np.mean(scores)
        return clip_score

    @staticmethod
    def cal_clipscore(vec_a, vec_b, w: float = 2.5):
        cos_sim = np.sum(vec_a * vec_b, axis=1) / (np.linalg.norm(vec_a, axis=1) * np.linalg.norm(vec_b, axis=1))
        clipscore = w * np.maximum(cos_sim, 0)
        return clipscore


class MClipWrapper:
    """
    Wrapping M-Clip to keep the interface consistence
    """

    MCLIP_PAIR = {
        "M-CLIP/LABSE-Vit-L-14": ("ViT-L-14", "openai"),
        "M-CLIP/XLM-Roberta-Large-Vit-B-32": ("ViT-B-32", "openai"),
        "M-CLIP/XLM-Roberta-Large-Vit-L-14": ("ViT-L-14", "openai"),
        "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus": (
            "ViT-B-16-plus-240",
            "laion400m_e31",
        ),
    }

    def __init__(self, clip_name: str = "M-CLIP/XLM-Roberta-Large-Vit-L-14", device=None):
        img_clip_name, img_pretrained = self.MCLIP_PAIR[clip_name]

        self.device = device
        self.img_model, _, self.img_preprocess = open_clip.create_model_and_transforms(
            img_clip_name, pretrained=img_pretrained, device=self.device
        )
        self.img_model.eval()

        self.txt_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(clip_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(clip_name)

        self.encode_image = self.img_model.encode_image
        self.encode_text = partial(self.txt_model.forward, tokenizer=self.tokenizer)


CLIPScore = partial(ClipScore, clip_name="ViT-B-32", pretrained="openai")
MCLIPScore = partial(ClipScore, clip_name="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", pretrained=None)


def meteor_eval(references, hypotheses, tgt_lang):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    meteor_jar = os.path.join(current_dir, METEOR_JAR)

    # Create temporary files for references and hypotheses
    with (
        tempfile.NamedTemporaryFile(mode="w+", delete=False) as ref_file,
        tempfile.NamedTemporaryFile(mode="w+", delete=False) as hyp_file,
    ):
        ref_count = None
        if isinstance(references[0], list):
            ref_count = len(references[0])
            for ref in references:
                ref_file.write("\n".join(ref) + "\n")
            hyp_file.write("\n".join(hypotheses) + "\n")
        else:
            # Write the list of references and hypotheses to the temporary files
            ref_file.write("\n".join(references) + "\n")
            hyp_file.write("\n".join(hypotheses) + "\n")

        # Flush the buffers to ensure all data is written
        ref_file.flush()
        hyp_file.flush()

        # Construct the command as a list
        command = ["java", "-Xmx2G", "-jar", meteor_jar, hyp_file.name, ref_file.name]

        if ref_count is not None:
            command.extend(["-r", str(ref_count)])

        if tgt_lang in ["en", "de", "fr", "es"]:
            command.extend(["-l", tgt_lang, "-norm"])
        else:
            command.extend(["-l", tgt_lang])

        result = subprocess.run(command, capture_output=True, text=True)

    pattern = r"Final score:\s+([\d.]+)"
    match = re.search(pattern, result.stdout)
    final_score = float(match.group(1))
    return final_score * 100


@torch.no_grad()
def comet_eval(sources, references, hypotheses):
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    model.eval()

    # Data must be in the following format:
    data = []
    for src, mt, ref in zip(sources, hypotheses, references):
        data.append({"src": src, "mt": mt, "ref": ref})

    # Call predict method:
    model_output = model.predict(data, batch_size=8, gpus=1)
    return model_output.system_score * 100


def pycocoevalcap_eval(references, hypotheses, lang="en"):
    scorers = {
        "BLEU": Bleu(4),
        # "ROUGE_L": Rouge(),
        "CIDEr": Cider(),
        "cocoMETEOR": Meteor(),
        # "SPICE": Spice(),
    }

    # Tokenize ground truth
    gts = defaultdict(list)
    res = defaultdict(list)
    for i, (refs, hyp) in enumerate(zip(references, hypotheses)):
        gts[i] = [{"caption": r.strip()} for r in refs]
        res[i].append({"caption": hyp.strip()})

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    results = dict()
    for name, scorer in scorers.items():
        score, _ = scorer.compute_score(gts, res)
        if name == "BLEU":
            for i, s in enumerate(score):
                results[f"BLEU_{i + 1}"] = s * 100
        else:
            results[name] = score * 100

    return results


def clipscore_eval(img_paths, hypotheses):
    return CLIPScore().compute_score(img_paths, hypotheses) * 100


def mclipscore_eval(img_paths, hypotheses):
    return MCLIPScore().compute_score(img_paths, hypotheses) * 100


PUNCTUATIONS = (
    set(
        [
            ".",
            "...",
            "·",
            ",",
            "。",
            "、",
            ",",
            ":",
            ";",
            "?",
            "!",
            "''",
            "'",
            "``",
            "`",
            '"',
            "-",
            "--",
            "_",
            "/",
            "\\",
            "《",
            "》",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            ">",
            "<",
            "=",
            "+",
            "@",
            "#",
            "%",
            "&",
            "*",
        ]
    )
    | set(string.punctuation)
    | set("，。！？；：“”‘’…„")  # General punctuation
    | set("–—")  # Dashes
    | set("«»")  # French and Russian guillemets
    | set("¿¡")  # Spanish inverted marks
    | set("“”‘’")  # Curly quotes
)


def xm3600_eval(references, hypotheses, lang):
    translator = str.maketrans("", "", "".join(PUNCTUATIONS))
    stanza.download(lang, processors="tokenize")
    nlp = stanza.Pipeline(lang, processors="tokenize")

    def tokenize_stanza(text):
        text = unicodedata.normalize("NFKC", text)
        doc = nlp(text)
        tokens = [
            word.text.translate(translator).lower()
            for sentence in doc.sentences
            for word in sentence.words
            if word.text
        ]
        return " ".join(tokens).strip()

    gts = defaultdict(list)
    res = defaultdict(list)
    for i, (refs, hyp) in enumerate(zip(references, hypotheses)):
        tok_refs = [tokenize_stanza(r) for r in refs]
        gts[i] = tok_refs
        tok_hyp = tokenize_stanza(hyp)
        res[i].append(tok_hyp)

    cider_score = round(Cider().compute_score(gts, res)[0] * 100, 1)

    return {"XM3600_CIDEr": cider_score}


#
# Datasets
#


class Multi30kDataset(torch.utils.data.Dataset):
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
        return Image.open(img_path).convert("RGB"), img_path, captions_en, captions_de


class XM3600Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str = "data/XM3600/", lang: str = "de"):
        assert lang in [
            "ar",
            "bn",
            "cs",
            "da",
            "de",
            "el",
            "en",
            "es",
            "fa",
            "fi",
            "fil",
            "fr",
            "hi",
            "hr",
            "hu",
            "id",
            "it",
            "he",
            "ja",
            "ko",
            "mi",
            "nl",
            "no",
            "pl",
            "pt",
            "quz",
            "ro",
            "ru",
            "sv",
            "sw",
            "te",
            "th",
            "tr",
            "uk",
            "vi",
            "zh",
        ], "Language not supported"
        self.images_folder = Path(dataset_path) / "images"
        self.lang = lang
        self.captions = []
        with open(Path(dataset_path) / "captions.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.captions.append(
                    {
                        "image_key": data["image/key"],
                        "captions": data["en"]["caption"],
                        f"captions_{self.lang}": data[self.lang]["caption"],  # caption/tokenized
                    }
                )

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_key = self.captions[idx]["image_key"]
        img_path = os.path.join(self.images_folder, f"{img_key}.jpg")
        # EN captions self.captions[idx]["captions"],
        return (
            Image.open(img_path).convert("RGB"),
            img_path,
            self.captions[idx][f"captions_{self.lang}"],
        )


class CoMMuTEDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path="data/CoMMuTE/", lang: str = "de"):
        assert lang in ["de", "fr", "ru", "zh"], "Language not supported"
        dataset_path = Path(dataset_path)
        commute_image_order = dataset_path / f"CoMMuTE/en-{lang}/img.order"
        commute_correct = dataset_path / f"CoMMuTE/en-{lang}/correct.{lang}"
        commute_incorrect = dataset_path / f"CoMMuTE/en-{lang}/incorrect.{lang}"
        commute_src = dataset_path / f"CoMMuTE/en-{lang}/src.en"
        with open(commute_image_order) as f:
            commute_images = [dataset_path / "images/" / line.replace("\n", "") for line in f.readlines()]
        with open(commute_correct) as f:
            commute_correct = [line.replace("\n", "") for line in f.readlines()]
        with open(commute_src) as f:
            commute_src = [line.replace("\n", "") for line in f.readlines()]
        with open(commute_incorrect) as f:
            commute_incorrect = [line.replace("\n", "") for line in f.readlines()]
        self.data = list(zip(commute_images, commute_src, commute_correct, commute_incorrect))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, src, correct, incorrect = self.data[idx]
        return Image.open(img_path).convert("RGB"), src, correct, incorrect


class COCOKarpathyDataset(torch.utils.data.Dataset):
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
        return Image.open(img_path).convert("RGB"), img_path, captions


#
# Models
#


class ModelWrapper:
    def process_batch_captioning(self, images, lang):
        raise NotImplementedError()

    def process_batch_translation(self, images, lang, src):
        raise NotImplementedError()

    def process_batch_lexical_ambiguity(self, images, lang, src, correct, incorrect):
        raise NotImplementedError()

    def generate_caption(self, inputs, device, torch_dtype) -> List[str]:
        raise NotImplementedError()

    def generate_translation(self, inputs, device, torch_dtype) -> List[str]:
        raise NotImplementedError()

    def get_perplexity(self, inputs, device, torch_dtype):
        raise NotImplementedError()


class MicrosoftFlorence2ModelWrapper(ModelWrapper):
    def __init__(self, hf_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.model = Florence2ForConditionalGeneration.from_pretrained(hf_path)
        self.model.to(self.device, self.torch_dtype)
        self.model.eval()
        self.processor = Florence2Processor.from_pretrained(hf_path)

    def process_batch_captioning(self, images, lang):
        prompt = ["<CAPTION>"] * len(images)
        inputs = self.processor(
            prompt,
            images=images,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return inputs

    @torch.no_grad()
    def generate_caption(self, inputs, lang):
        inputs.to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            num_beams=4,
            do_sample=False,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(generated_text)
        return [g.strip() for g in generated_text]


class Florence2ModelWrapper(ModelWrapper):
    def __init__(self, path, input_res, use_gemma_decoder=False, use_prefix=False):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.input_res = input_res
        self.use_prefix = use_prefix
        if use_gemma_decoder:
            self.model = FlorenceGemma2ForConditionalGeneration.from_pretrained(path)
        else:
            self.model = Florence2ForConditionalGeneration.from_pretrained(path)
        self.model.to(self.device, self.torch_dtype)
        self.model.eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/gemma-2-2b",
            add_bos_token=True,
            add_eos_token=True,
            padding_side="right",
            truncation_side="right",
        )
        self.processor = Florence2Processor.from_pretrained(
            "microsoft/Florence-2-large",
            new_tokenizer=tokenizer,
            use_encoder_tokenizer=True,
            insert_lang_token=False,
        )

        self.processor.image_processor.size = {
            "height": self.input_res,
            "width": self.input_res,
        }

    def process_batch_captioning(self, images, lang):
        prompt = [f"<LANG_{lang.upper()}><CAPTION>"] * len(images)
        inputs = self.processor(
            prompt,
            images=images,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        if self.use_prefix:
            prefix = {
                "en": "The image shows",
                "de": "Das Bild zeigt",
                "fr": "La photo montre",
                "es": "La imagen muestra",
                "ru": "На картинке изображен",
                "zh": "图为",
            }

            dec_inp_ids = self.processor.tokenizer([prefix[lang]] * len(images), return_tensors="pt")["input_ids"][
                :, :-1
            ]
            inputs["decoder_input_ids"] = dec_inp_ids
        return inputs

    def process_batch_translation(self, images, lang, src):
        prompt = [f"<LANG_{lang.upper()}><TRANSLATE>" + s for s in src]
        inputs = self.processor(
            prompt,
            images=images,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return inputs

    def process_batch_lexical_ambiguity(self, images, lang, src, correct, incorrect):
        prompt = [f"<LANG_{lang.upper()}><TRANSLATE>" + s for s in src]
        inputs = self.processor(prompt, images=images, padding=True, return_tensors="pt")
        bs = len(src)
        correct_decoder_input = self.processor.tokenizer(
            correct,
            return_tensors="pt",
            padding=True,
        )
        incorrect_decoder_input = self.processor.tokenizer(
            incorrect,
            return_tensors="pt",
            padding=True,
        )
        correct_decoder_input_ids = torch.cat(
            [
                torch.ones(bs, 1, dtype=torch.long) * self.processor.tokenizer.eos_token_id,
                correct_decoder_input["input_ids"],
            ],
            dim=-1,
        )
        correct_decoder_attention_mask = torch.cat(
            [
                torch.ones(bs, 1, dtype=torch.long),
                correct_decoder_input["attention_mask"],
            ],
            dim=-1,
        )

        incorrect_decoder_input_ids = torch.cat(
            [
                torch.ones(bs, 1, dtype=torch.long) * self.processor.tokenizer.eos_token_id,
                incorrect_decoder_input["input_ids"],
            ],
            dim=-1,
        )
        incorrect_decoder_attention_mask = torch.cat(
            [
                torch.ones(bs, 1, dtype=torch.long),
                incorrect_decoder_input["attention_mask"],
            ],
            dim=-1,
        )
        return (
            inputs,
            correct_decoder_input_ids,
            correct_decoder_attention_mask,
            incorrect_decoder_input_ids,
            incorrect_decoder_attention_mask,
        )

    @torch.no_grad()
    def generate_caption(self, inputs, lang):
        inputs.to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            decoder_input_ids=None if not self.use_prefix else inputs["decoder_input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            num_beams=4,
            do_sample=False,
            use_cache=False,
        )
        if self.use_prefix:
            generated_ids = generated_ids[:, inputs["decoder_input_ids"].shape[1] + 1 :]
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        def cap_first_letter(t):
            return t[0].upper() + t[1:]

        generated_text = [cap_first_letter(g.strip()) for g in generated_text]
        print(generated_text)
        return generated_text

    @torch.no_grad()
    def generate_translation(self, inputs, lang):
        return self.generate_caption(inputs, lang)

    @torch.no_grad()
    def get_perplexity(
        self,
        inputs,
        correct_decoder_input_ids,
        correct_decoder_attention_mask,
        incorrect_decoder_input_ids,
        incorrect_decoder_attention_mask,
        lang,
    ):
        inputs.to(self.device, self.torch_dtype)
        correct_decoder_input_ids = correct_decoder_input_ids.to(self.device)
        correct_decoder_attention_mask = correct_decoder_attention_mask.to(self.device)
        incorrect_decoder_input_ids = incorrect_decoder_input_ids.to(self.device)
        incorrect_decoder_attention_mask = incorrect_decoder_attention_mask.to(self.device)

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        correct_out_logits = self.model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=correct_decoder_input_ids,
            decoder_attention_mask=correct_decoder_attention_mask,
        ).logits

        shift_logits = correct_out_logits[..., :-1, :].contiguous()
        shift_labels = correct_decoder_input_ids[..., 1:].contiguous()
        shift_attention_mask_batch = correct_decoder_attention_mask[..., 1:].contiguous()
        correct_perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        incorrect_out_logits = self.model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=incorrect_decoder_input_ids,
            decoder_attention_mask=incorrect_decoder_attention_mask,
        ).logits

        shift_logits = incorrect_out_logits[..., :-1, :].contiguous()
        shift_labels = incorrect_decoder_input_ids[..., 1:].contiguous()
        shift_attention_mask_batch = incorrect_decoder_attention_mask[..., 1:].contiguous()

        incorrect_perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        return correct_perplexity_batch, incorrect_perplexity_batch


class Gemma3ModelWrapper(ModelWrapper):
    def __init__(self, model_id="google/gemma-3-4b-it"):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if "cuda" in self.device else torch.float32

        self.model = (
            Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
            )
            .to(self.device, self.torch_dtype)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def process_batch_captioning(self, images, lang):
        lang_to_prompt = {
            "en": "Describe this image in one short sentence. Only answer with the caption.",
            "de": "Describe this image in one short sentence in German. Only answer with the caption.",
            "fr": "Describe this image in one short sentence in French. Only answer with the caption.",
            "es": "Describe this image in one short sentence in Spanish. Only answer with the caption.",
            "ru": "Describe this image in one short sentence in Russian. Only answer with the caption.",
            "zh": "Describe this image in one short sentence in Chinese. Only answer with the caption.",
        }
        inputs = []
        for img in images:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": lang_to_prompt[lang]},
                    ],
                },
            ]
            i = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device, dtype=self.torch_dtype)
            inputs.append(i)
        return inputs

    def process_batch_translation(self, images, lang, src):
        lang_to_prompt = {
            "de": "Translate the following to German. Only answer with the translation.\n",
            "fr": "Translate the following to French. Only answer with the translation.\n",
            "es": "Translate the following to Spanish. Only answer with the translation.\n",
            "ru": "Translate the following to Russian. Only answer with the translation.\n",
            "zh": "Translate the following to Chinese. Only answer with the translation.\n",
        }
        inputs = []
        for img, s in zip(images, src):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": f"{lang_to_prompt[lang]} {s}"},
                    ],
                },
            ]
            i = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.append(i)
        return inputs

    def process_batch_lexical_ambiguity(self, images, lang, src, correct, incorrect):
        lang_to_prompt = {
            "de": "Translate the following to German. Only answer with the translation.\n",
            "fr": "Translate the following to French. Only answer with the translation.\n",
            "es": "Translate the following to Spanish. Only answer with the translation.\n",
            "ru": "Translate the following to Russian. Only answer with the translation.\n",
            "zh": "Translate the following to Chinese. Only answer with the translation.\n",
        }
        correct_inputs = []
        incorrect_inputs = []
        prompt_inputs = []
        for img, s, c, w in zip(images, src, correct, incorrect):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": f"{lang_to_prompt[lang]} {s}"},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_inputs.append(prompt)
            c_i = self.processor.apply_chat_template(
                messages
                + [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": c},
                        ],
                    }
                ],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            correct_inputs.append(c_i)
            w_i = self.processor.apply_chat_template(
                messages
                + [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": w},
                        ],
                    }
                ],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            incorrect_inputs.append(w_i)

        return (
            prompt_inputs,
            correct_inputs,
            None,
            incorrect_inputs,
            None,
        )

    @torch.no_grad()
    def generate_caption(self, inputs, lang):
        generated_text = []
        with torch.inference_mode():
            for i in inputs:
                i.to(self.device, dtype=self.torch_dtype)
                input_len = i["input_ids"].shape[-1]
                generation = self.model.generate(**i, max_new_tokens=128, do_sample=False, top_p=None, top_k=None)
                generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                generated_text.append(decoded)
        return generated_text

    @torch.no_grad()
    def generate_translation(self, inputs, lang):
        return self.generate_caption(inputs, lang)

    @torch.no_grad()
    def get_perplexity(
        self,
        inputs,  # prompt inptus
        correct_decoder_input_ids,  # correct inputs
        correct_decoder_attention_mask,
        incorrect_decoder_input_ids,  # incorrect inputs
        incorrect_decoder_attention_mask,
        lang,
    ):
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        correct_perplexity_batch, incorrect_perplexity_batch = [], []
        for prompt_input, correct_input, incorrect_input in zip(
            inputs, correct_decoder_input_ids, incorrect_decoder_input_ids
        ):
            correct_input.to(self.device, self.torch_dtype)
            incorrect_input.to(self.device, self.torch_dtype)
            prompt_len = prompt_input["input_ids"].shape[-1]

            # Calculate the perplexity for the correct output
            correct_out_logits = self.model(**correct_input).logits
            shift_logits = correct_out_logits[..., prompt_len:-1, :].contiguous().to(torch.float32)
            shift_labels = correct_input["input_ids"][..., prompt_len + 1 :].contiguous()
            shift_attention_mask_batch = torch.ones_like(shift_labels)
            correct_perplexity = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            ).item()
            correct_perplexity_batch.append(correct_perplexity)

            # Calculate the perplexity for the incorrect output
            incorrect_out_logits = self.model(**incorrect_input).logits
            shift_logits = incorrect_out_logits[..., prompt_len:-1, :].contiguous().to(torch.float32)
            shift_labels = incorrect_input["input_ids"][..., prompt_len + 1 :].contiguous()
            shift_attention_mask_batch = torch.ones_like(shift_labels)
            incorrect_perplexity = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            ).item()
            incorrect_perplexity_batch.append(incorrect_perplexity)

        return correct_perplexity_batch, incorrect_perplexity_batch


class PixtralModelWrapper(ModelWrapper):
    def __init__(self):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if "cuda" in self.device else torch.float32

        model_id = "mistral-community/pixtral-12b"
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id).to(self.device, self.torch_dtype).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def process_batch_captioning(self, images, lang):
        lang_to_prompt = {
            "en": "Describe this image in one short sentence. Only answer with the caption.",
            "de": "Describe this image in one short sentence in German. Only answer with the caption.",
            "fr": "Describe this image in one short sentence in French. Only answer with the caption.",
            "es": "Describe this image in one short sentence in Spanish. Only answer with the caption.",
            "ru": "Describe this image in one short sentence in Russian. Only answer with the caption.",
            "zh": "Describe this image in one short sentence in Chinese. Only answer with the caption.",
        }
        inputs = []
        for img in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": lang_to_prompt[lang]},
                    ],
                },
            ]
            i = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device, dtype=self.torch_dtype)
            inputs.append(i)
        return inputs

    def process_batch_translation(self, images, lang, src):
        lang_to_prompt = {
            "de": "Translate the following to German. Only answer with the translation.\n",
            "fr": "Translate the following to French. Only answer with the translation.\n",
            "es": "Translate the following to Spanish. Only answer with the translation.\n",
            "ru": "Translate the following to Russian. Only answer with the translation.\n",
            "zh": "Translate the following to Chinese. Only answer with the translation.\n",
        }
        inputs = []
        for img, s in zip(images, src):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": f"{lang_to_prompt[lang]} {s}"},
                    ],
                },
            ]
            print(messages)
            i = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.append(i)
        return inputs

    def process_batch_lexical_ambiguity(self, images, lang, src, correct, incorrect):
        lang_to_prompt = {
            "de": "Translate the following to German. Only answer with the translation.\n",
            "fr": "Translate the following to French. Only answer with the translation.\n",
            "es": "Translate the following to Spanish. Only answer with the translation.\n",
            "ru": "Translate the following to Russian. Only answer with the translation.\n",
            "zh": "Translate the following to Chinese. Only answer with the translation.\n",
        }
        correct_inputs = []
        incorrect_inputs = []
        prompt_inputs = []
        for img, s, c, w in zip(images, src, correct, incorrect):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": f"{lang_to_prompt[lang]} {s}"},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_inputs.append(prompt)
            c_i = self.processor.apply_chat_template(
                messages
                + [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": c},
                        ],
                    }
                ],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            correct_inputs.append(c_i)
            w_i = self.processor.apply_chat_template(
                messages
                + [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": w},
                        ],
                    }
                ],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            incorrect_inputs.append(w_i)

        return (
            prompt_inputs,
            correct_inputs,
            None,
            incorrect_inputs,
            None,
        )

    @torch.no_grad()
    def generate_caption(self, inputs, lang):
        generated_text = []
        with torch.inference_mode():
            for i in inputs:
                i.to(self.device, dtype=self.torch_dtype)
                input_len = i["input_ids"].shape[-1]
                generation = self.model.generate(**i, max_new_tokens=128, do_sample=False)
                generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                generated_text.append(decoded)
        return generated_text

    @torch.no_grad()
    def generate_translation(self, inputs, lang):
        return self.generate_caption(inputs, lang)

    @torch.no_grad()
    def get_perplexity(
        self,
        inputs,  # prompt inptus
        correct_decoder_input_ids,  # correct inputs
        correct_decoder_attention_mask,
        incorrect_decoder_input_ids,  # incorrect inputs
        incorrect_decoder_attention_mask,
        lang,
    ):
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        correct_perplexity_batch, incorrect_perplexity_batch = [], []
        for prompt_input, correct_input, incorrect_input in zip(
            inputs, correct_decoder_input_ids, incorrect_decoder_input_ids
        ):
            correct_input.to(self.device, self.torch_dtype)
            incorrect_input.to(self.device, self.torch_dtype)
            prompt_len = prompt_input["input_ids"].shape[-1]

            # Calculate the perplexity for the correct output
            correct_out_logits = self.model(**correct_input).logits
            shift_logits = correct_out_logits[..., prompt_len:-1, :].contiguous().to(torch.float32)
            shift_labels = correct_input["input_ids"][..., prompt_len + 1 :].contiguous()
            shift_attention_mask_batch = torch.ones_like(shift_labels)
            correct_perplexity = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            ).item()
            correct_perplexity_batch.append(correct_perplexity)

            # Calculate the perplexity for the incorrect output
            incorrect_out_logits = self.model(**incorrect_input).logits
            shift_logits = incorrect_out_logits[..., prompt_len:-1, :].contiguous().to(torch.float32)
            shift_labels = incorrect_input["input_ids"][..., prompt_len + 1 :].contiguous()
            shift_attention_mask_batch = torch.ones_like(shift_labels)
            incorrect_perplexity = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            ).item()
            incorrect_perplexity_batch.append(incorrect_perplexity)

            print(correct_perplexity, incorrect_perplexity)

        return correct_perplexity_batch, incorrect_perplexity_batch


class NLLBModelWrapper(ModelWrapper):
    lang_map = {
        "en": "eng_Latn",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "zh": "zho_Hans",
        "ru": "rus_Cyrl",
    }

    def __init__(self, model_name):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"Init {model_name} ...")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device, self.torch_dtype)
        self.model.eval()

    def process_batch_captioning(self, images, lang):
        pass

    def process_batch_translation(self, images, lang, src):
        # self.tokenizer.tgt_lang = self.lang_map[lang]
        inputs = self.tokenizer(src, padding=True, return_attention_mask=True, return_tensors="pt")
        return inputs

    @torch.no_grad()
    def generate_caption(self, inputs, lang):
        pass

    @torch.no_grad()
    def generate_translation(self, inputs, lang):
        inputs.to(self.device)
        lang = self.lang_map[lang]
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(lang),
            max_new_tokens=128,
            num_beams=4,
            do_sample=False,
        )

        decoded = [s.strip() for s in self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)]
        print(decoded)
        return decoded

    @torch.no_grad()
    def get_perplexity(
        self,
        inputs,
        correct_decoder_input_ids,
        correct_decoder_attention_mask,
        incorrect_decoder_input_ids,
        incorrect_decoder_attention_mask,
        lang,
    ):
        pass


class BaselineWrapper(ModelWrapper):
    lang_map = {
        "en": "eng_Latn",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "zh": "zho_Hans",
        "ru": "rus_Cyrl",
    }

    def __init__(self, model_name):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device, self.torch_dtype)
        self.model.eval()

        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
        self.blip2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
        self.blip2.to(self.device, self.torch_dtype)
        self.blip2.eval()

    def process_batch_captioning(self, images, lang):
        return self.blip2_processor(images, ["a photo of"] * len(images), return_tensors="pt")

    def process_batch_translation(self, images, lang, src):
        caption_inputs = self.process_batch_captioning(images, "en")
        captions = self.generate_caption(caption_inputs, "en")

        self.tokenizer.tgt_lang = self.lang_map[lang]
        inputs = self.tokenizer(
            [f"{s} ### ({c})" for s, c in zip(src, captions)],
            padding=True,
            return_tensors="pt",
        )
        inputs_fallback = self.tokenizer(src, padding=True, return_tensors="pt")
        return inputs, inputs_fallback

    def process_batch_lexical_ambiguity(self, images, lang, src, correct, incorrect):
        # Generate captions for the images
        caption_inputs = self.process_batch_captioning(images, "en")
        captions = self.generate_caption(caption_inputs, "en")

        # Combine source text with captions
        src_with_captions = [f"{s} {c}" for s, c in zip(src, captions)]

        # Tokenize for each translation
        self.tokenizer.tgt_lang = self.lang_map[lang]
        correct_inputs = self.tokenizer(src_with_captions, text_target=correct, return_tensors="pt", padding=True)
        incorrect_inputs = self.tokenizer(src_with_captions, text_target=incorrect, return_tensors="pt", padding=True)

        correct_decoder_input_ids = correct_inputs["labels"]
        correct_decoder_input_ids = torch.cat(
            [
                (
                    torch.ones(
                        correct_decoder_input_ids.shape[0],
                        1,
                        device=correct_decoder_input_ids.device,
                    )
                    * self.tokenizer.convert_tokens_to_ids("</s>")
                ).to(torch.long),
                correct_decoder_input_ids,
            ],
            dim=-1,
        )

        incorrect_decoder_input_ids = incorrect_inputs["labels"]
        incorrect_decoder_input_ids = torch.cat(
            [
                (
                    torch.ones(
                        incorrect_decoder_input_ids.shape[0],
                        1,
                        device=incorrect_decoder_input_ids.device,
                    )
                    * self.tokenizer.convert_tokens_to_ids("</s>")
                ).to(torch.long),
                incorrect_decoder_input_ids,
            ],
            dim=-1,
        )

        return (
            correct_inputs,
            correct_decoder_input_ids,
            correct_decoder_input_ids != self.tokenizer.pad_token_id,
            incorrect_decoder_input_ids,
            incorrect_decoder_input_ids != self.tokenizer.pad_token_id,
        )

    @torch.no_grad()
    def generate_caption(self, inputs, lang):
        inputs.to(self.device, self.torch_dtype)
        out = self.blip2.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            do_sample=False,
        )
        answers = self.blip2_processor.batch_decode(out, skip_special_tokens=True)
        answers = [s.replace("\n", " ").replace("a photo of ", "").strip() for s in answers]
        if lang == "en":
            return answers
        inputs = self.tokenizer(answers, padding=True, return_tensors="pt").to(self.device)
        lang = self.lang_map[lang]
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(lang),
            max_new_tokens=128,
            num_beams=4,
            do_sample=False,
        )
        captions = [s.strip() for s in self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)]
        captions = [s[0].upper() + s[1:] if len(s) > 1 else s for s in captions]
        return captions

    @torch.no_grad()
    def generate_translation(self, inputs, lang):
        inputs, inputs_fallback = inputs
        inputs.to(self.device)
        lang = self.lang_map[lang]
        translated_tokens = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(lang),
            max_new_tokens=128,
            num_beams=4,
            do_sample=False,
        )

        decoded = [s.strip() for s in self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)]

        # Identify translations that failed with simple rules
        retry = []
        decoded_retry = []
        for i, translation in enumerate(decoded):
            new_translation = translation.split("#")[0].strip()
            model_included_details = len(translation) - len(new_translation) > 0
            model_includes_translation = len(new_translation) > 0 and len(set(new_translation)) > 5
            if not model_included_details or not model_includes_translation:
                retry.append(i)
                decoded_retry.append(None)
            else:
                decoded_retry.append(new_translation)

        if len(retry) > 0:
            inputs_fallback.to(self.device)
            translated_tokens = self.model.generate(
                input_ids=inputs_fallback["input_ids"][retry],
                attention_mask=inputs_fallback["attention_mask"][retry],
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(lang),
                max_new_tokens=128,
                num_beams=4,
                do_sample=False,
            )

            decoded = [s.strip() for s in self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)]

            for i, d in zip(retry, decoded):
                decoded_retry[i] = d

        print(decoded_retry, len(retry) / 64)
        return decoded_retry

    @torch.no_grad()
    def get_perplexity(
        self,
        inputs,
        correct_decoder_input_ids,
        correct_decoder_attention_mask,
        incorrect_decoder_input_ids,
        incorrect_decoder_attention_mask,
        lang,
    ):
        inputs.to(self.device)
        correct_decoder_input_ids = correct_decoder_input_ids.to(self.device)
        correct_decoder_attention_mask = correct_decoder_attention_mask.to(self.device)
        incorrect_decoder_input_ids = incorrect_decoder_input_ids.to(self.device)
        incorrect_decoder_attention_mask = incorrect_decoder_attention_mask.to(self.device)
        lang = self.lang_map[lang]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            correct_out_logits = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=correct_decoder_input_ids,
                decoder_attention_mask=correct_decoder_attention_mask,
            ).logits

        shift_logits = correct_out_logits[..., :-1, :].contiguous()
        shift_labels = correct_decoder_input_ids[..., 1:].contiguous()
        # shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
        shift_attention_mask_batch = correct_decoder_attention_mask[..., 1:].contiguous()
        correct_perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        with torch.no_grad():
            incorrect_out_logits = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=incorrect_decoder_input_ids,
                decoder_attention_mask=incorrect_decoder_attention_mask,
            ).logits

        shift_logits = incorrect_out_logits[..., :-1, :].contiguous()
        shift_labels = incorrect_decoder_input_ids[..., 1:].contiguous()
        # shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
        shift_attention_mask_batch = incorrect_decoder_attention_mask[..., 1:].contiguous()
        incorrect_perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        return correct_perplexity_batch, incorrect_perplexity_batch


class PaliGemmaModelWrapper(ModelWrapper):
    def __init__(
        self, model_id="google/paligemma-3b-ft-coco35l-448"
    ):  # paligemma-3b-mix-224 google/paligemma-3b-ft-coco35l-224
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
        self.model.to(self.device, self.torch_dtype)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def process_batch_captioning(self, images, lang):
        prompt = "<image>caption " + lang + "\n"
        model_inputs = self.processor(text=[prompt] * len(images), images=list(images), return_tensors="pt")
        return model_inputs

    def process_batch_translation(self, images, lang, src):
        pass

    def process_batch_lexical_ambiguity(self, images, lang, src, correct, incorrect):
        pass

    @torch.no_grad()
    def generate_caption(self, inputs, lang):
        inputs.to(self.device, self.torch_dtype)
        input_len = inputs["input_ids"].shape[-1]
        generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[:, input_len:]
        decoded = self.processor.batch_decode(generation, skip_special_tokens=True)
        print(decoded)
        return [
            re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", d) for d in decoded
        ]  # Remove white space between chinese characters

    @torch.no_grad()
    def generate_translation(self, inputs, lang):
        pass

    @torch.no_grad()
    def get_perplexity(
        self,
        inputs,
        correct_decoder_input_ids,
        correct_decoder_attention_mask,
        incorrect_decoder_input_ids,
        incorrect_decoder_attention_mask,
        lang,
    ):
        pass


def get_model(name: str, path: str, model_type: str, input_res: int, use_prefix: bool):
    if model_type == "florence2_src":
        return MicrosoftFlorence2ModelWrapper(
            "microsoft/Florence-2-base" if "base" in path else "microsoft/Florence-2-large"
        )
    elif model_type == "florence2":
        return Florence2ModelWrapper(path, input_res=input_res, use_prefix=use_prefix)
    elif model_type == "florencegemma2":
        return Florence2ModelWrapper(path, input_res=input_res, use_gemma_decoder=True, use_prefix=use_prefix)
    elif model_type == "paligemma":
        return PaliGemmaModelWrapper("google/paligemma-3b-ft-coco35l-448")
    elif model_type == "nllb":
        if "600m" in path:
            model_name = "facebook/nllb-200-distilled-600M"
        elif "1b" in path:
            model_name = "facebook/nllb-200-1.3B"
        else:
            model_name = "facebook/nllb-200-3.3B"
        return NLLBModelWrapper(model_name)
    elif model_type == "base":
        return BaselineWrapper("facebook/nllb-200-3.3B")
    elif model_type == "gemma3":
        return Gemma3ModelWrapper("google/gemma-3-4b-it")
    elif model_type == "gemma3-12b":
        return Gemma3ModelWrapper("google/gemma-3-12b-it")
    elif model_type == "pixtral":
        return PixtralModelWrapper()
    else:
        raise Exception()


@torch.no_grad()
def eval_translation_multi30k(model: ModelWrapper, batch_size, dataset_kwargs, output_folder):
    dataset = Multi30kDataset(**dataset_kwargs)
    split = dataset_kwargs["split"]
    lang = dataset_kwargs["lang"]
    task = dataset_kwargs["task"]

    output_path = output_folder / f"preds_translation_multi30k_{task}_{split}_{lang}.json"

    if not output_path.exists():

        def collate_fn(data):
            images, img_paths, src, tgt = zip(*data)
            src = [s[0].strip() for s in src]
            tgt = [t[0].strip() for t in tgt]
            inputs = model.process_batch_translation(images=images, lang=lang, src=src)
            return inputs, src, tgt

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        sources, predictions, references = [], [], []
        for inputs, src, tgt in tqdm(dataloader, desc=f"Evaluating Multi30k Translation ({lang})"):
            generated_text = model.generate_translation(inputs, lang)
            sources += src
            predictions += generated_text
            references += tgt

            if DEV_RUN:
                break

        with open(output_path, "w") as f:
            json.dump(
                {
                    "sources": sources,
                    "predictions": predictions,
                    "references": references,
                },
                f,
            )

    else:
        with open(output_path, "r") as f:
            data = json.load(f)
            sources, predictions, references = (
                data["sources"],
                data["predictions"],
                data["references"],
            )

    result = {}  # pycocoevalcap_eval([[r] for r in references], predictions)
    result["SacreBLEU"] = sacrebleu.corpus_bleu(
        predictions, [references], tokenize="zh" if lang == "zh" else "13a"
    ).score
    if lang in ["cz", "de", "en", "es", "fr"]:
        result["METEOR"] = meteor_eval(references, predictions, tgt_lang=lang)
    result["COMET"] = comet_eval(sources, references, predictions)
    bleu = evaluate.load("bleu")
    result["HfBLEU"] = bleu.compute(predictions=predictions, references=references)["bleu"] * 100
    return result


@torch.no_grad()
def eval_translation_commute(model: ModelWrapper, batch_size, dataset_kwargs, output_folder):
    dataset = CoMMuTEDataset(**dataset_kwargs)
    lang = dataset_kwargs["lang"]

    output_path = output_folder / f"preds_translation_commute_{lang}.json"
    if not output_path.exists():

        def collate_fn(data):
            images, src, correct, incorrect = zip(*data)
            inputs = model.process_batch_translation(images=images, lang=lang, src=src)
            return inputs, src, correct

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        sources, predictions, references = [], [], []
        for inputs, src, tgt in tqdm(dataloader, desc=f"Evaluating CoMMuTE Translation ({lang})"):
            generated_text = model.generate_translation(inputs, lang)
            sources += src
            predictions += generated_text
            references += tgt

            if DEV_RUN:
                break

        with open(output_path, "w") as f:
            json.dump(
                {
                    "sources": sources,
                    "predictions": predictions,
                    "references": references,
                },
                f,
            )
    else:
        with open(output_path, "r") as f:
            data = json.load(f)
            sources, predictions, references = (
                data["sources"],
                data["predictions"],
                data["references"],
            )

    result = {}
    result["SacreBLEU"] = sacrebleu.corpus_bleu(
        predictions, [references], tokenize="zh" if lang == "zh" else "13a"
    ).score
    if lang in ["cz", "de", "en", "es", "fr"]:
        result["METEOR"] = meteor_eval(references, predictions, tgt_lang=lang)
    result["COMET"] = comet_eval(sources, references, predictions)
    return result


@torch.no_grad()
def eval_caption_coco_karpathy(model: ModelWrapper, batch_size, dataset_kwargs, output_folder):
    dataset = COCOKarpathyDataset(**dataset_kwargs)
    lang = "en"

    output_path = output_folder / f"preds_caption_coco_karpathy_{lang}.json"
    if not output_path.exists():

        def collate_fn(data):
            images, img_paths, tgt = zip(*data)
            tgt = [[t.strip() for t in tt] for tt in tgt]
            inputs = model.process_batch_captioning(images=images, lang=lang)
            return inputs, img_paths, tgt

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        predictions, all_img_paths, references = [], [], []
        for inputs, img_paths, tgt in tqdm(dataloader, desc="Evaluating COCO Karpathy Captioning"):
            generated_text = model.generate_caption(inputs, lang)
            predictions += generated_text
            all_img_paths += img_paths
            references += tgt

            if DEV_RUN:
                break

        with open(output_path, "w") as f:
            json.dump(
                {
                    "predictions": predictions,
                    "all_img_paths": [str(p) for p in all_img_paths],
                    "references": references,
                },
                f,
            )
    else:
        with open(output_path, "r") as f:
            data = json.load(f)
            predictions, all_img_paths, references = (
                data["predictions"],
                data["all_img_paths"],
                data["references"],
            )

    results = pycocoevalcap_eval(references, predictions)
    results["CLIPScore"] = clipscore_eval(all_img_paths, predictions)
    return results


@torch.no_grad()
def eval_caption_multi30k(model: ModelWrapper, batch_size, dataset_kwargs, output_folder):
    dataset = Multi30kDataset(**dataset_kwargs)
    split = dataset_kwargs["split"]
    lang = dataset_kwargs["lang"]
    task = dataset_kwargs["task"]

    output_path = output_folder / f"preds_caption_multi30k_{task}_{split}_{lang}.json"
    if not output_path.exists():

        def collate_fn(data):
            images, img_paths, _, tgt = zip(*data)
            tgt = [[t.strip() for t in tt] for tt in tgt]
            inputs = model.process_batch_captioning(images=images, lang=lang)
            return inputs, img_paths, tgt

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        predictions, all_img_paths, references = [], [], []
        for inputs, img_paths, tgt in tqdm(dataloader, desc=f"Evaluating Multi30k Captioning ({lang})"):
            generated_text = model.generate_caption(inputs, lang)
            predictions += generated_text
            all_img_paths += img_paths
            references += tgt

            if DEV_RUN:
                break

        with open(output_path, "w") as f:
            json.dump(
                {
                    "predictions": predictions,
                    "all_img_paths": [str(p) for p in all_img_paths],
                    "references": references,
                },
                f,
            )
    else:
        with open(output_path, "r") as f:
            data = json.load(f)
            predictions, all_img_paths, references = (
                data["predictions"],
                data["all_img_paths"],
                data["references"],
            )

    results = pycocoevalcap_eval(references, predictions)
    results["CLIPScore"] = clipscore_eval(all_img_paths, predictions)
    results["MCLIPScore"] = mclipscore_eval(all_img_paths, predictions)
    results["SacreBLEU"] = sacrebleu.corpus_bleu(
        predictions, references, tokenize="zh" if lang == "zh" else "13a"
    ).score
    if lang in ["cz", "de", "en", "es", "fr"]:
        results["METEOR"] = meteor_eval(references, predictions, tgt_lang=lang)
    return results


@torch.no_grad()
def eval_caption_xm3600(model: ModelWrapper, batch_size, dataset_kwargs, output_folder):
    dataset = XM3600Dataset(**dataset_kwargs)
    lang = dataset_kwargs["lang"]

    output_path = output_folder / f"preds_caption_xm3600_{lang}.json"
    if not output_path.exists():

        def collate_fn(data):
            images, img_paths, tgt = zip(*data)
            tgt = [[t.strip() for t in tt] for tt in tgt]
            inputs = model.process_batch_captioning(images=images, lang=lang)
            return inputs, img_paths, tgt

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        predictions, all_img_paths, references = [], [], []
        for inputs, img_paths, tgt in tqdm(dataloader, desc=f"Evaluating XM3600 Captioning ({lang})"):
            generated_text = model.generate_caption(inputs, lang)
            predictions += generated_text
            all_img_paths += img_paths
            references += tgt

            if DEV_RUN:
                break

        with open(output_path, "w") as f:
            json.dump(
                {
                    "predictions": predictions,
                    "all_img_paths": [str(p) for p in all_img_paths],
                    "references": references,
                },
                f,
            )
    else:
        with open(output_path, "r") as f:
            data = json.load(f)
            predictions, all_img_paths, references = (
                data["predictions"],
                data["all_img_paths"],
                data["references"],
            )

    results = xm3600_eval(references, predictions, lang)
    results["CLIPScore"] = clipscore_eval(all_img_paths, predictions)
    results["MCLIPScore"] = mclipscore_eval(all_img_paths, predictions)
    return results


@torch.no_grad()
def eval_lexical_ambiguity_commute(model: ModelWrapper, batch_size, dataset_kwargs, _):
    dataset = CoMMuTEDataset(**dataset_kwargs)
    lang = dataset_kwargs["lang"]

    def collate_fn(data):
        images, src, correct, incorrect = zip(*data)
        return model.process_batch_lexical_ambiguity(images, lang, src, correct, incorrect)

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    predictions = []
    for (
        inputs,
        correct_decoder_input_ids,
        correct_decoder_attention_mask,
        incorrect_decoder_input_ids,
        incorrect_decoder_attention_mask,
    ) in tqdm(dataloader, desc=f"Evaluating CoMMuTE Lexical Ambiguity ({lang})"):
        correct_perplexity_batch, incorrect_perplexity_batch = model.get_perplexity(
            inputs,
            correct_decoder_input_ids,
            correct_decoder_attention_mask,
            incorrect_decoder_input_ids,
            incorrect_decoder_attention_mask,
            lang,
        )
        predictions += [int(c < i) for c, i in zip(correct_perplexity_batch, incorrect_perplexity_batch)]

        if DEV_RUN:
            break

    return {"accuracy": sum(predictions) / len(predictions) * 100}


def print_latex_table(
    results,
    dataset_key: str,
    latex_keys: list,
    latex_metrics: list,
    mean_keys=False,
    precision=1,
):
    # Check if the latex keys are in the results
    for k in latex_keys:
        if not any((dataset_key + k) == r["task"] for r in results):
            print(dataset_key + k + " not in dataset")
            return

    # Iterate over results and extract requested information
    latex_results = {}
    for r in results:
        if r["task"].startswith(dataset_key):
            key = r["task"].replace(dataset_key, "")
            if key in latex_keys:
                latex_results[key] = {m: r[m] if m in r else -100 for m in latex_metrics}

    if mean_keys:
        mean_results = {}
        for m in latex_metrics:
            tmp_res = []
            for v in latex_results.values():
                tmp_res.append(v[m])
            mean_results[m] = sum(tmp_res) / len(tmp_res)

        print(" & ".join(m for m in latex_metrics))
        print(" & ".join(str(round(mean_results[m], precision)) for m in latex_metrics))
    else:
        # Print results
        print(" & ".join(k for k in latex_keys for _ in latex_metrics))
        print(" & ".join(m for _ in latex_keys for m in latex_metrics))
        print(" & ".join(str(round(latex_results[k][m], precision)) for k in latex_keys for m in latex_metrics))

    print("")


def compose_latex_results(results):
    print("Full multi30k translation results for de:")
    print_latex_table(
        results,
        dataset_key="multi30k_translation_",
        latex_keys=[
            "test_2016_flickr_de",
            "test_2017_flickr_de",
            "test_2018_flickr_de",
            "test_2017_mscoco_de",
        ],
        latex_metrics=["SacreBLEU", "COMET"],
    )
    print("Full multi30k translation results for fr:")
    print_latex_table(
        results,
        dataset_key="multi30k_translation_",
        latex_keys=[
            "test_2016_flickr_fr",
            "test_2017_flickr_fr",
            "test_2018_flickr_fr",
            "test_2017_mscoco_fr",
        ],
        latex_metrics=["SacreBLEU", "COMET"],
    )
    print("Mean multi30k translation results (2016, 2017, coco) for de:")
    print_latex_table(
        results,
        dataset_key="multi30k_translation_",
        latex_keys=[
            "test_2016_flickr_de",
            "test_2017_flickr_de",
            "test_2017_mscoco_de",
        ],
        latex_metrics=["SacreBLEU", "COMET"],
        mean_keys=True,
    )
    print("Mean multi30k translation results (2016, 2017, coco) for fr:")
    print_latex_table(
        results,
        dataset_key="multi30k_translation_",
        latex_keys=[
            "test_2016_flickr_fr",
            "test_2017_flickr_fr",
            "test_2017_mscoco_fr",
        ],
        latex_metrics=["SacreBLEU", "COMET"],
        mean_keys=True,
    )
    print("Commute translation results for all languages:")
    print_latex_table(
        results,
        dataset_key="commute_translation_",
        latex_keys=["de", "fr", "ru", "zh"],
        latex_metrics=["SacreBLEU", "COMET"],
    )
    print("Mean Commute translation results for X:")
    print_latex_table(
        results,
        dataset_key="commute_translation_",
        latex_keys=["de", "fr", "ru", "zh"],
        latex_metrics=["SacreBLEU"],
        mean_keys=True,
    )

    print("Commute lexical ambiguity for all languages:")
    print_latex_table(
        results,
        dataset_key="commute_lexical_ambiguity_",
        latex_keys=["de", "fr", "ru", "zh"],
        latex_metrics=["accuracy"],
    )
    print("Commute lexical ambiguity for all languages:")
    print_latex_table(
        results,
        dataset_key="commute_lexical_ambiguity_",
        latex_keys=["de", "fr", "ru", "zh"],
        latex_metrics=["accuracy"],
        mean_keys=True,
    )
    print("Multi30k captioning results for en:")
    print_latex_table(
        results,
        dataset_key="multi30k_caption_test_2016_",
        latex_keys=["en"],
        latex_metrics=["BLEU_4", "CIDEr", "CLIPScore"],
    )
    print("Multi30k captioning results for de:")
    print_latex_table(
        results,
        dataset_key="multi30k_caption_test_2016_",
        latex_keys=["de"],
        latex_metrics=["BLEU_4", "CIDEr", "MCLIPScore"],
    )
    print("XM3600 captioning results:")
    print_latex_table(
        results,
        dataset_key="xm3600_caption_",
        latex_keys=["en", "de", "fr", "es", "ru", "zh"],
        latex_metrics=["XM3600_CIDEr"],  # , "CLIPScore", "MCLIPScore"],
    )
    print("MEAN XM3600 captioning results:")
    print_latex_table(
        results,
        dataset_key="xm3600_caption_",
        latex_keys=["fr", "es", "ru", "zh"],
        latex_metrics=["XM3600_CIDEr"],
        mean_keys=True,
    )
    print("COCO karpathy captioning results:")
    print_latex_table(
        results,
        dataset_key="",
        latex_keys=["coco_karpathy_caption"],
        latex_metrics=["BLEU_4", "CIDEr", "CLIPScore"],
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on various datasets.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name.")
    parser.add_argument("--model_type", type=str, required=True, help="Model type.")
    parser.add_argument("--eval_datasets_root", type=str, required=True, help="The path to the eval datasets.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--input_res", type=int, default=768, help="Input resolution")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--eval_task_set", type=str, default="full", help="Eval task set.")
    parser.add_argument("--use_prefix", action="store_true", help="Add a decoder prefx")
    args = parser.parse_args()

    assert_meteor_jar_exists()

    eval_datasets_root = Path(args.eval_datasets_root)
    output_path = Path(args.model_path) / f"eval_results_{args.eval_task_set}.json"
    if args.eval_task_set == "full":
        output_path = Path(args.model_path) / "eval_results.json"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if os.path.exists(output_path):
        print(f"Results for to {args.model_path}")
        print("")
        with open(output_path, "r") as f:
            all_metrics = json.load(f)
        compose_latex_results(all_metrics)
        return

    # Initialize wandb
    wandb.init(
        name=f"eval_{args.model_name}",
        project="huggingface",
    )

    captioning_tasks = [
        # Multi30k Captioning (task2, test_2016)
        *[
            (
                f"multi30k_caption_test_2016_{lang}",
                eval_caption_multi30k,
                {
                    "dataset_path": eval_datasets_root / "multi30k",
                    "flickr30k_images_path": eval_datasets_root / "multi30k/flickr30k-images",
                    "coco_images_path": eval_datasets_root / "multi30k/flickr30k-images",
                    "task": "task2",
                    "split": "test_2016",
                    "lang": lang,
                },
            )
            for lang in ["en", "de"]
        ],
        # XM3600 captioning
        *[
            (
                f"xm3600_caption_{lang}",
                eval_caption_xm3600,
                {
                    "dataset_path": eval_datasets_root / "xm3600",
                    "lang": lang,
                },
            )
            for lang in ["en", "de", "fr", "es", "ru", "zh"]
        ],
        # COCO karpathy
        (
            "coco_karpathy_caption",
            eval_caption_coco_karpathy,
            {"dataset_path": eval_datasets_root / "coco", "split": "test"},
        ),
    ]

    translation_tasks = [
        # Multi30k Translation (task1, test_2016_flickr, test_2017_flickr, test_2018_flickr)
        *[
            (
                f"multi30k_translation_{split}_{lang}",
                eval_translation_multi30k,
                {
                    "dataset_path": eval_datasets_root / "multi30k",
                    "flickr30k_images_path": eval_datasets_root / "multi30k/flickr30k-images",
                    "coco_images_path": eval_datasets_root / "coco",
                    "task": "task1",
                    "split": split,
                    "lang": lang,
                },
            )
            for split in [
                "test_2016_flickr",
                "test_2017_flickr",
                "test_2018_flickr",
                "test_2017_mscoco",
            ]
            for lang in ["de", "fr"]
        ],
        # CoMMuTE Translation
        *[
            (
                f"commute_translation_{lang}",
                eval_translation_commute,
                {"dataset_path": eval_datasets_root / "CoMMuTE", "lang": lang},
            )
            for lang in ["de", "fr", "ru", "zh"]
        ],
    ]

    ambiguity_tasks = [
        # CoMMuTE Lexical Ambiguity
        *[
            (
                f"commute_lexical_ambiguity_{lang}",
                eval_lexical_ambiguity_commute,
                {"dataset_path": eval_datasets_root / "CoMMuTE", "lang": lang},
            )
            for lang in ["de", "fr", "ru", "zh"]
        ],
    ]

    if args.eval_task_set == "full":
        eval_tasks = [*translation_tasks, *captioning_tasks, *ambiguity_tasks]
    elif args.eval_task_set == "ambiguity":
        eval_tasks = [*ambiguity_tasks]
    elif args.eval_task_set == "caption":
        eval_tasks = [*captioning_tasks]
    elif args.eval_task_set == "translation":
        eval_tasks = [*translation_tasks]
    elif args.eval_task_set == "en_caption":
        # en captioning only
        eval_tasks = [  # Multi30k Captioning (task2, test_2016)
            (
                "multi30k_caption_test_2016_en",
                eval_caption_multi30k,
                {
                    "dataset_path": eval_datasets_root / "multi30k",
                    "flickr30k_images_path": eval_datasets_root / "multi30k/flickr30k-images",
                    "coco_images_path": eval_datasets_root / "multi30k/flickr30k-images",
                    "task": "task2",
                    "split": "test_2016",
                    "lang": "en",
                },
            ),
            # XM3600 captioning
            (
                "xm3600_caption_en",
                eval_caption_xm3600,
                {
                    "dataset_path": eval_datasets_root / "xm3600",
                    "lang": "en",
                },
            ),
            # COCO karpathy
            (
                "coco_karpathy_caption",
                eval_caption_coco_karpathy,
                {"dataset_path": eval_datasets_root / "coco", "split": "test"},
            ),
        ]
    else:
        raise Exception()

    print(f"Evaluating {len(eval_tasks)} tasks ...")
    if DEV_RUN:
        print("Dev run is active.")

    print("Load model ...")
    model = get_model(
        args.model_name,
        args.model_path,
        args.model_type,
        args.input_res,
        args.use_prefix,
    )

    print(f"Results will be saved to {output_path}")

    start = time.time()

    all_metrics = []
    for task_name, eval_func, dataset_kwargs in eval_tasks:
        metrics = eval_func(model, args.batch_size, dataset_kwargs, Path(args.model_path))
        metrics["task"] = task_name
        all_metrics.append(metrics)
        print("")
        print(tabulate([metrics], headers="keys"))
        print("")

    stop = time.time()

    # Display all metrics in a single table
    print(tabulate(all_metrics, headers="keys"))

    # Load to wandb
    all_keys = set()
    for metrics in all_metrics:
        all_keys.update(metrics.keys())
    wandb_table = wandb.Table(columns=list(all_keys))
    for metrics in all_metrics:
        wandb_table.add_data(*[metrics[key] if key in metrics else None for key in all_keys])
    wandb.log({"evaluation_metrics": wandb_table})

    print(f"Elapsed time: {stop - start}")

    if not DEV_RUN:
        with open(output_path, "w") as f:
            json.dump(all_metrics, f)


if __name__ == "__main__":
    main()
