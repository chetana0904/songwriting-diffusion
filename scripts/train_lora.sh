export PYTHONPATH=$PYTHONPATH:$PWD

accelerate launch --config-file config/accelerate_config.yaml \
    train/train_lora.py \
    --model-config config/diffrhythm-1b.json \
    --batch-size 4 \
    --max-frames 2048 \
    --min-frames 512 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --learning-rate 2e-4 \
    --num-warmup-updates 100 \
    --epochs 10 \
    --save-per-updates 500 \
    --last-per-steps 500 \
    --audio-drop-prob 0.3 \
    --style-drop-prob 0.1 \
    --lrc-drop-prob 0.1 \
    --grad-accumulation-steps 1 \
    --exp-name diffrhythm-lora-test \
    --file-path "dataset/train.scp" \
    --resumable-with-seed 666
