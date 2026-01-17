# Setup Continous Pretraining Dataset

Download cc12m and the custom cc12m annotations form huggingface.

```shell
python scripts/download_cc12m.py
```

Update the paths in the file.
This will download the cc12m images `pixparse/cc12m-wds` and custom annotations `spravil/cc12m_ccmatrix_captions_and_translations`.

Next, prepare the annotations.

```shell
python scripts/prepare_cc12m_annotations.py -i data/cc12m_annotations/train -o data/cc12m_annotations_preprocessed/train
python scripts/prepare_cc12m_eval_annotations.py -i data/cc12m_annotations/eval -o data/cc12m_annotations_preprocessed/eval
```

Finally we can "bake" the annotations into the `tar` files.

```bash
bash bake_cc12m.sh
```
