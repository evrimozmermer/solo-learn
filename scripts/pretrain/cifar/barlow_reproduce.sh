# 
python main_pretrain.py \
    --dataset cifar10 \
    --backbone resnet18 \
    --train_data_path ./datasets \
    --val_data_path ./datasets \
    --max_epochs 1000 \
    --devices 0 \
    --num_workers 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer adamw \
    --scheduler warmup_cosine \
    --min_lr 0.00008 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --brightness 0.2 \
    --contrast 0.2 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.8 \
    --solarization_prob 0.0 0.0 \
    --num_crops_per_aug 1 1 \
    --crop_size 32 \
    --knn_eval \
    --wandb \
    --name barlow_cifar10 \
    --project ssl-lp-whitepaper \
    --entity evrimozmermer \
    --save_checkpoint \
    --auto_resume \
    --method barlow_twins \
    --proj_hidden_dim 2048 \
    --proj_output_dim 64 