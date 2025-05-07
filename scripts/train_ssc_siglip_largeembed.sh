rm -rf /scratch/partial_datasets/mvbench/logs/test
# 64k batchsize for 2.048e-3 lr
CUDA_VISIBLE_DEVICES=6,7 TORCH_CUDNN_V8_API_ENABLED=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun  --standalone --nproc_per_node 1 -m open_clip_train.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --dataset-config "configs/datasets/dataset_config.yaml" \
    --lr "1e-5" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 0 \
    --wd 0.2 \
    --batch-size 2 \
    --accum-freq 128 \
    --log-every-n-steps 1 \
    --aug-cfg scale='(0.8, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --image-mean 0.5 0.5 0.5 \
    --image-std 0.5 0.5 0.5 \
    --epochs=300 \
    --workers=16 \
    --model ViT-SO400M-14-SigLIP2-384 \
    --pretrained WebLI \
    --force-image-size 384 \
    --train-num-samples 1200000000 \
    --precision 'bf16' \
    --local-loss \
    --gather-with-grad \
    --seed 4 \
    --name 'test' \
    --ssc \
    --siglip_weight 1.0 \
    --ssc_weight 1.0 \
    --grad-checkpointing
    # --resume '/scratch/partial_datasets/mvbench/logs/siglip2_so400m_384_ii_tt_it_probe/checkpoints/epoch_latest.pt'
    # --resume '/scratch/partial_datasets/mvbench/logs/siglip2_embedsize1_so400m_decoders_resume0/checkpoints/epoch_latest.pt'
    # --resume '~/logs/embedsize1_so400m_resume0/checkpoints/epoch_latest.pt'
# --batch-size 12 \
#     --accum-freq 768 \
