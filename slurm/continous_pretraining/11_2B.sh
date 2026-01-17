#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=1000G
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --export=ALL

export TRAIN_DATASET_PATH="data/cc12m_baked/train/cc12m-train-{0000..2174}.tar"
export EVAL_DATASET_ROOT_PATH="data/cc12m_baked/eval"
export TRAINER_OUTPUT_FOLDER="data/models/11_2B-500steps"
export MODEL_NAME="Florence-Gemma-2-xlarge"
export USE_PRETRAINED_DECODER=true
export MAX_ENCODER_SEQ_LENGTH=128
export MAX_DECODER_SEQ_LENGTH=128
export IMAGE_SIZE=224
export USE_DUMMY_DATASET=false
export BALANCING_TYPE=task-lang-balancing
export STEPS=500

# Calculate wolrd size
export NODES=1
export GPUS_PER_NODE=8
WORLD_SIZE=$(( NODES * GPUS_PER_NODE ))
echo "NODES=$NODES GPUS_PER_NODE=$GPUS_PER_NODE WORLD_SIZE=$WORLD_SIZE"

# Calculate grad seteps
ALL_BATCH_SIZE=1024
LOCAL_BATCH_SIZE=32
let GRAD_ACCUM_STEPS=ALL_BATCH_SIZE/GPUS_PER_NODE/LOCAL_BATCH_SIZE
echo GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS

source .venv/bin/activate
python --version

OUTPUT_FILE="$SLURM_SUBMIT_DIR/R-$SLURM_JOB_NAME.$SLURM_JOB_ID.out"
echo "Submit Directory: $OUTPUT_FILE"

accelerate launch --config_file "configs/fsdp_config.yaml" src/train.py \
    --per_device_train_batch_size=${LOCAL_BATCH_SIZE} \
    --per_device_eval_batch_size=${LOCAL_BATCH_SIZE} \
    --gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
    --output_dir=${TRAINER_OUTPUT_FOLDER} \
    --eval_strategy=steps \
    --eval_steps=${STEPS} \
    --num_train_epochs=1 \
    --max_steps=${STEPS} \
    --dataloader_num_workers=15 \
    --fp16=True \
    --gradient_checkpointing=False \
    --logging_steps=25 \
    --fp16_full_eval=True \
    --save_total_limit=1 \
    --save_steps=${STEPS} \
    --lr_scheduler_type=cosine \
    --warmup_steps=100 \
    --learning_rate=1e-4 \
    --weight_decay=0.01 \
    --dispatch_batches=False \
    --report_to=wandb \
    --dataloader_pin_memory=False \
    --ddp_find_unused_parameters=False \
    --dataloader_persistent_workers=False \
    --ignore_data_skip=True \
    --ddp_timeout=3600 \
    --seed=44595
