# 1.20 it/s
python main_pretrain.py \
    --dataset custom \
    --backbone swin_tiny \
    --train_data_path ./datasets/universal_image_embeddings/train \
    --val_data_path ./datasets/universal_image_embeddings/eval \
    --max_epochs 100 \
    --devices 0 \
    --sync_batchnorm \
    --num_workers 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm_lars \
    --scheduler warmup_cosine \
    --lr 0.1 \
    --weight_decay 1.5e-6 \
    --batch_size 32 \
    --brightness 0.2 \
    --contrast 0.2 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.2 \
    --solarization_prob 0.0 0.0 \
    --crop_size 224 \
    --num_crops_per_aug 1 1 \
    --knn_eval \
    --name bt_uie++ \
    --project ssl-lp-whitepaper \
    --entity evrimozmermer \
    --wandb \
    --save_checkpoint \
    --method barlow_twins \
    --proj_hidden_dim 2048 \
    --proj_output_dim 16 \
    --auto_resume \
    --auto_resumer_max_hours 120