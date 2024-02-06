bash run.sh \
    --task text \
    --model-dir ./models/llama-2-7b-1layer \
    --tokenizer ./models/llama-2-7b-1layer/tokenizer.model \
    --device cpu \
    --prompt-file prompts/text_completion_example.yml \
    --do-profile 0 \
    --profile-output ./models/llama-2-7b-1layer/ \
    --init-method random \
    --data-type float32

# bash run.sh \
#     --task text \
#     --model-dir ./models/llama-2-7b-1layer \
#     --tokenizer ./models/llama-2-7b-1layer/tokenizer.model \
#     --device cuda \
#     --prompt-file prompts/text_completion_example.yml \
#     --do-profile 0 \
#     --profile-output ./models/llama-2-7b-1layer/ \
#     --init-method random \
#     --data-type cuda.FloatTensor
