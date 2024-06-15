# dataset path
DATA_PATH='../../dataset/IndianPines'

## Setup
'''
pip install -r requirements.txt
'''

## Run
1. pretrain model
```bash
# Set the path to save checkpoints
OUTPUT_DIR='HSI_IP/pretrain/NeiCoT_IP'
# path to Indianpines train set
DATA_PATH='../../dataset/IndianPines'

OMP_NUM_THREADS=1 python -u run_neicot_pretraining.py \
        --mask_ratio 0.5 \
        --model pretrain_mae_3_1_patch \
        --batch_size 128 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 20 \
        --epochs 100 \
        --output_dir ${OUTPUT_DIR} \
        --dataset IndianPines \
        --save_ckpt_freq 100 \
        --patch_size 11 \
        --depth 7 \
        --cp_loss 0.25 \
        --sigma 1 \
        --use_model neicot_pre
```

2. class_finetune
```bash
# Set the path to save checkpoints
OUTPUT_DIR='./HSI_IP/NeiCoT_IP_test'
# path to imagenet-1k set
DATA_PATH='../../dataset/IndianPines'
# path to pretrain model
MODEL_PATH='HSI_IP/pretrain/NeiCoT_IP/checkpoint.pth'

OMP_NUM_THREADS=1 python -u run_class_finetuning.py \
        --model vit_3_patch \
        --finetune ${MODEL_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 64 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 200 \
        --dist_eval \
        --run 1 \
        --dataset IndianPines \
        --patch_size 11 \
        --depth 7 \
        --cp_loss 0.25 \
        --sigma 1 \
        --model_name neicot_liner \
        --load_data 0.03
'''

# See method.sh for more information about the code