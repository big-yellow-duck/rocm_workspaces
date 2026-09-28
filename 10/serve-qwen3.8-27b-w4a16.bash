    vllm serve dbirks/Qwen3.8-27B-W4A16-AutoRound \
        --max-num-seqs 8 \
        --gpu-memory-utilization 0.8 \
        --max-num-batched-tokens 8192 \
        --max-model-len auto \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --reasoning-parser qwen3 \
        --mm-encoder-tp-mode data \
        --load-format fastsafetensors \
        --attention-backend ROCM_SEGMENTED_ATTN

