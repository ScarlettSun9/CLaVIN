torchrun --nproc_per_node 1 train.py \
    --llm_model 7B\
    --llama_model_path ./llama/llama-2-7b/ \
    --tokenizer_path ./tokenizer/merged_tokenizer_hf \
    --data_path ./data/PT_Dataset/ \
    --max_seq_len 512 \
    --batch_size 1 \
    --accum_iter 1 \
    --epochs 1 \
    --warmup_epochs 0.1 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./LaVIN-7B-VLIT/\
    --adapter_type attn\
    --adapter_dim 8\
    --adapter_scale 1\
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 5.\
    --visual_adapter_type router \
    --do_pretrain