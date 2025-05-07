rm -rf /home/terran/logs/siglip2_so400m_384_reproduction_4
# 64k batchsize for 2.048e-3 lr
python -m spacy download en_core_web_sm
CUDA_VISIBLE_DEVICES=2,3,4,5 TORCH_CUDNN_V8_API_ENABLED=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun  --standalone --nproc_per_node 1 -m open_clip_train.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --dataset-config "configs/datasets/dataset_config.yaml" \
    --lr "1e-5" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 0 \
    --wd 0.2 \
    --batch-size 6 \
    --accum-freq 256 \
    --log-every-n-steps 1 \
    --aug-cfg scale='(0.8, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --image-mean 0.5 0.5 0.5 \
    --image-std 0.5 0.5 0.5 \
    --epochs=300 \
    --epochs-steps=10 \
    --workers=6 \
    --model ViT-SO400M-14-SigLIP2-384 \
    --pretrained WebLI \
    --force-image-size 384 \
    --train-num-samples 1200000000 \
    --precision 'bf16' \
    --local-loss \
    --gather-with-grad \
    --seed 1000 \
    --logs-path /home/terran/logs \
    --name 'siglip2_so400m_384_reproduction_4' \
    --ssc \
    --siglip_weight 1.0 \
    --ssc_weight 1.0 \
    --grad-checkpointing \
    --resume '/home/terran/logs/siglip2_so400m_384_reproduction/checkpoints/epoch_latest.pt'
