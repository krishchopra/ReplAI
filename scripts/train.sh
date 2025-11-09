export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Generate timestamp for output directory
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path NousResearch/Hermes-4-14B \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template qwen3 \
    --flash_attn auto \
    --use_unsloth True \
    --dataset_dir train/LLaMA-Factory/data \
    --dataset replai_fixed \
    --cutoff_len 131072 \
    --learning_rate 1e-05 \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 200 \
    --warmup_steps 20 \
    --packing False \
    --enable_thinking False \
    --report_to wandb \
    --output_dir saves/Custom/lora/train_${TIMESTAMP} \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch 