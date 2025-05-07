rm -rf logs/ssc_aimv2_phi35_recon_var_diffpool
# 64k batchsize for 2.048e-3 lr
CUDA_VISIBLE_DEVICES=0,3 TORCH_CUDNN_V8_API_ENABLED=1 torchrun  --standalone --nproc_per_node 1 -m open_clip_train.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --dataset-config "configs/datasets/dataset_config.yaml" \
    --lr "1e-4" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 0 \
    --wd 0.2 \
    --batch-size 16 \
    --accum-freq 96 \
    --log-every-n-steps 1 \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --image-mean 0.48145466 0.4578275 0.40821073 \
    --image-std 0.26862954 0.26130258 0.27577711 \
    --epochs=300 \
    --workers=6 \
    --model aimv2-large-patch14+llama1B \
    --force-image-size 224 \
    --train-num-samples 12000000 \
    --precision 'bf16' \
    --local-loss \
    --gather-with-grad \
    --seed 0 \
    --name 'ssc_aimv2_phi35_recon_var_diffpool' \
    --ssc \
    --siglip_weight 1.0 \
    --ssc_weight 1.0 \
    --debug
    # --resume '/home/terran/projects/vlmfrontier/open_clip/logs/ssc_aimv2_phi35_recon_var/checkpoints/epoch_1.pt'
        # --grad-checkpointing \
    #  --wandb-project-name "train_ssc_siglip_4_resume_1" \
    # --report-to "wandb" \
    # --imagenet-val '/path/to/ImageNet/val' \

