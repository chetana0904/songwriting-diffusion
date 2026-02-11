cd "$(dirname "$0")"
cd ../

export PYTHONPATH=$PYTHONPATH:$PWD

if [[ "$OSTYPE" =~ ^darwin ]]; then
    # Prefer ARM Homebrew path on Apple Silicon so the dylib matches Python's arch (arm64)
    if [[ "$(uname -m)" == "arm64" ]] && [[ -f /opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib ]]; then
        export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib
    else
        export PHONEMIZER_ESPEAK_LIBRARY=$(brew --prefix espeak-ng)/lib/libespeak-ng.dylib
    fi
fi

python3 infer/infer.py \
    --lrc-path infer/example/eg_en_full.lrc \
    --ref-audio-path infer/example/eg_en.mp3 \
    --audio-length 177 \
    --output-dir infer/example/output \
    --chunked \
    --batch-infer-num 5