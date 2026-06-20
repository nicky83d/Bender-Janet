#!/usr/bin/env bash
set -euo pipefail
source ~/JanetEnv/bin/activate || true
python3 -m edge_tts --voice en-GB-RyanNeural --text "Hello, I am Janet." --write-media /tmp/janet_edge_en.mp3
python3 -m edge_tts --voice zh-CN-XiaoxiaoNeural --text "你好，我是 Janet。" --write-media /tmp/janet_edge_zh.mp3
ffmpeg -hide_banner -loglevel error -y -i /tmp/janet_edge_en.mp3 -ar 16000 -ac 1 /tmp/janet_edge_en.wav
ffmpeg -hide_banner -loglevel error -y -i /tmp/janet_edge_zh.mp3 -ar 16000 -ac 1 /tmp/janet_edge_zh.wav
aplay -D plughw:CARD=Y02,DEV=0 /tmp/janet_edge_en.wav
aplay -D plughw:CARD=Y02,DEV=0 /tmp/janet_edge_zh.wav
