rm -rf logs/ssc_dino_2
# 64k batchsize for 2.048e-3 lr
CUDA_VISIBLE_DEVICES=2,3 TORCH_CUDNN_V8_API_ENABLED=1 torchrun --nproc_per_node 1 -m open_clip_train.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --dataset-config "configs/datasets/dataset_config.yaml" \
    --lr "1e-4" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 782 \
    --wd 0.2 \
    --batch-size 64 \
    --accum-freq 256 \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --epochs=3 \
    --workers=6 \
    --model Dino+ViT-SO400M-14-SigLIP-LM \
    --pretrained WebLI \
    --image-mean 0.485 0.456 0.406 \
    --image-std 0.229 0.224 0.225 \
    --force-image-size 224 \
    --train-num-samples 12000000 \
    --precision 'amp_bf16' \
    --local-loss \
    --gather-with-grad \
    --log-every-n-steps 32 \
    --seed 0 \
    --name 'ssc_dino_2' \
    --report-to "wandb" \
    --wandb-project-name "train_ssc" \
    --ssc \
    --siglip_weight 1.0 \
    --ssc_weight 1.0 \
    --debug
    # --grad-checkpointing \
    # --imagenet-val '/path/to/ImageNet/val' \

