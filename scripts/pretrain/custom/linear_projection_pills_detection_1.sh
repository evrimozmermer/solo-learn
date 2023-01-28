# 1.32 it/s
python main_pretrain.py \
    --dataset custom \
    --backbone resnet18 \
    --train_data_path D:/workspace/datasets/pill_detection_1_ssl/train \
    --val_data_path D:/workspace/datasets/pill_detection_1_ssl/eval \
    --max_epochs 200 \
    --devices 0 \
    --num_workers 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer adamw \
    --scheduler warmup_cosine \
    --min_lr 0.0001 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --brightness 0.2 \
    --contrast 0.2 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.8 \
    --solarization_prob 0.0 0.0 \
    --num_crops_per_aug 1 1 \
    --crop_size 512 \
    --knn_eval \
    --wandb \
    --name lp_uie_pills_det1 \
    --project ssl-lp-whitepaper \
    --entity evrimozmermer \
    --save_checkpoint \
    --method linear_projection \
    --proj_hidden_dim 512 \
    --proj_output_dim 32