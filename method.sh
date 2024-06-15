# Indianpines_pretrain
# OMP_NUM_THREADS=1 python -u run_neicot_pretraining.py \
#                     --mask_ratio 0.5 \
#                     --model pretrain_mae_3_1_patch \
#                     --batch_size 128 \
#                     --opt adamw \
#                     --opt_betas 0.9 0.95 \
#                     --warmup_epochs 20 \
#                     --epochs 100 \
#                     --output_dir ./HSI_IP/pretrain/NeiCoT_IP \
#                     --dataset IndianPines \
#                     --save_ckpt_freq 100 \
#                     --patch_size 11 \
#                     --depth 7 \
#                     --cp_loss 0.25 \
#                     --sigma 1 \
#                     --use_model neicot_pre

# liner_classification
# OMP_NUM_THREADS=1 python -u run_class_finetuning.py \
#                     --model vit_3_patch \
#                     --finetune ./HSI_IP/pretrain/NeiCoT_IP/checkpoint-99.pth \
#                     --output_dir ./HSI_IP/NeiCoT_IP_test \
#                     --batch_size 64 \
#                     --opt adamw \
#                     --opt_betas 0.9 0.999 \
#                     --weight_decay 0.05 \
#                     --epochs 200 \
#                     --run 1 \
#                     --dist_eval \
#                     --dataset IndianPines \
#                     --patch_size 11 \
#                     --depth 7 \
#                     --cp_loss 0.25 \
#                     --sigma 1 \
#                     --model_name neicot_liner \
#                     --load_data 0.03