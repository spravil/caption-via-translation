# Setup Finetuning Dataset

Download and place the finetuning datasets adhering the following folder structure.

```bash
path/to/finetuning_datasets/
├── CoMMuTE
│   ├── CoMMuTE
│   ├── images
│   └── images_dev
├── coco
│   ├── annotations
│   ├── train2014
│   └── val2014
├── docci
│   ├── docci_descriptions.jsonlines
│   └── images
├── image_paragraphs
│   ├── images
│   ├── paragraphs_v1.json
│   ├── test_split.json
│   ├── train_split.json
│   └── val_split.json
├── multi30k
│   ├── data
│   ├── flickr30k-images
│   ├── task1_test_2017
│   ├── task1_test_2018
└── xm3600
    ├── captions.jsonl
    ├── dataset.py
    └── images
```

Note: CoMMuTE and xm3600 are not used for training, only for evaluation.

Execute the following scripts in the given order:

```bash
python scripts/finetuning_dataset/datasets_to_csv.py
python scripts/finetuning_dataset/recaption.py
python scripts/finetuning_dataset/translate.py
python scripts/finetuning_dataset/translate_without_context.py
python scripts/finetuning_dataset/bake.py
`````
