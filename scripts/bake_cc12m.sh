#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --export=ALL

source .venv/bin/activate
python --version

SOURCE_FOLDER="data/cc12m"
VAL_ANNOTATION_FOLDER="data/cc12m_annotations_preprocessed/eval"
VAL_OUTPUT_FOLDER="data/cc12m_baked/eval"
TRAIN_ANNOTATION_FOLDER="data/cc12m_annotations_preprocessed/train"
TRAIN_OUTPUT_FOLDER="data/cc12m_baked/train"
NUM_WORKERS=16

# Process val
python scripts/bake_cc12m.py -e "${VAL_ANNOTATION_FOLDER}/cc12m_de_caption/" -o ${SOURCE_FOLDER} -n "${VAL_OUTPUT_FOLDER}/cc12m_de_caption/" -w ${NUM_WORKERS}
python scripts/bake_cc12m.py -e "${VAL_ANNOTATION_FOLDER}/cc12m_en_caption/" -o ${SOURCE_FOLDER} -n "${VAL_OUTPUT_FOLDER}/cc12m_en_caption/" -w ${NUM_WORKERS}
python scripts/bake_cc12m.py -e "${VAL_ANNOTATION_FOLDER}/cc12m_es_caption/" -o ${SOURCE_FOLDER} -n "${VAL_OUTPUT_FOLDER}/cc12m_es_caption/" -w ${NUM_WORKERS}
python scripts/bake_cc12m.py -e "${VAL_ANNOTATION_FOLDER}/cc12m_zh_caption/" -o ${SOURCE_FOLDER} -n "${VAL_OUTPUT_FOLDER}/cc12m_zh_caption/" -w ${NUM_WORKERS}
python scripts/bake_cc12m.py -e "${VAL_ANNOTATION_FOLDER}/cc12m_es_translation/" -o ${SOURCE_FOLDER} -n "${VAL_OUTPUT_FOLDER}/cc12m_es_translation/" -w ${NUM_WORKERS}
python scripts/bake_cc12m.py -e "${VAL_ANNOTATION_FOLDER}/cc12m_zh_translation/" -o ${SOURCE_FOLDER} -n "${VAL_OUTPUT_FOLDER}/cc12m_zh_translation/" -w ${NUM_WORKERS}

# Process train
python scripts/bake_cc12m.py -e ${TRAIN_ANNOTATION_FOLDER} -o ${SOURCE_FOLDER} -n ${TRAIN_OUTPUT_FOLDER} -w ${NUM_WORKERS}