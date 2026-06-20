#!/usr/bin/env bash
set -e
echo "Testing Janet Y02 speaker..."
amixer -c Y02 sset Speaker 85% unmute || true
amixer -c Y02 sset PCM 85% unmute || true
espeak-ng -w /tmp/janet_hi.wav "Hi Nico, I am Janet" || espeak -w /tmp/janet_hi.wav "Hi Nico, I am Janet"
aplay -D plughw:CARD=Y02,DEV=0 /tmp/janet_hi.wav
