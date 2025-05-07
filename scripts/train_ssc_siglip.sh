rm -rf /scratch/partial_datasets/mvbench/logs/ssc_siglip_image_imagetext
# 64k batchsize for 2.048e-3 lr
CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_CUDNN_V8_API_ENABLED=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun  --standalone --nproc_per_node 4 -m open_clip_train.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --dataset-config "configs/datasets/dataset_config.yaml" \
    --lr "1e-4" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 32 \
    --wd 0.2 \
    --batch-size 64 \
    --accum-freq 96 \
    --log-every-n-steps 1 \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --image-mean 0.5 0.5 0.5 \
    --image-std 0.5 0.5 0.5 \
    --epochs=300 \
    --workers=6 \
    --model ViT-SO400M-14-SigLIP \
    --pretrained WebLI \
    --force-image-size 224 \
    --train-num-samples 12000000 \
    --precision 'bf16' \
    --local-loss \
    --gather-with-grad \
    --seed 0 \
    --name 'ssc_siglip_image_imagetext' \
    --ssc \
    --siglip_weight 1.0 \
    --ssc_weight 1.0 \
    --debug \
    --grad-checkpointing
    # --resume '/scratch/partial_datasets/mvbench/logs/ssc_siglip_recon_diffpool_probebottleneck_resume2/checkpoints/epoch_latest.pt'
    #  --wandb-project-name "train_ssc_siglip_4_resume_1" \
    # --report-to "wandb" \
        # --pretrained WebLI \
        # --resume '/home/terran/projects/vlmfrontier/open_clip/logs/ssc_siglip_4/checkpoints/epoch_2.pt' \
    # --imagenet-val '/path/to/ImageNet/val' \

