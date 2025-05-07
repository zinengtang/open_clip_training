rm -rf /home/terran/logs/siglip2_base_224_ii_tt_it_hnm_new_new_4
# 64k batchsize for 2.048e-3 lr
CUDA_VISIBLE_DEVICES=5 TORCH_CUDNN_V8_API_ENABLED=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun  --standalone --nproc_per_node 1 -m open_clip_train.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --dataset-config "configs/datasets/dataset_config_2.yaml" \
    --lr "1e-5" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 0 \
    --wd 0.2 \
    --batch-size 64 \
    --accum-freq 32 \
    --log-every-n-steps 1 \
    --aug-cfg scale='(0.8, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --image-mean 0.5 0.5 0.5 \
    --image-std 0.5 0.5 0.5 \
    --epochs=300 \
    --workers=0 \
    --model ViT-B-16-SigLIP2 \
    --pretrained WebLI \
    --force-image-size 224 \
    --train-num-samples 1200000000 \
    --precision 'bf16' \
    --local-loss \
    --gather-with-grad \
    --seed 100 \
    --logs-path /home/terran/logs \
    --name 'siglip2_base_224_ii_tt_it_hnm_new_new_4' \
    --ssc \
    --siglip_weight 1.0 \
    --ssc_weight 1.0 \
    --grad-checkpointing \
    --resume '/home/terran/logs/siglip2_base_224_ii_tt_it_hnm_new_new_3/checkpoints/epoch_latest.pt'
    # --resume '/home/terran/logs/epoch_7.pt'
    #  --wandb-project-name "train_ssc_siglip_4_resume_1" \
    # --report-to "wandb" \
        # --pretrained WebLI \
        # --resume '/home/terran/projects/vlmfrontier/open_clip/logs/ssc_siglip_4/checkpoints/epoch_2.pt' \
    # --imagenet-val '/path/to/ImageNet/val' \
    # --batch-size 64 \
    # --accum-freq 192 \

