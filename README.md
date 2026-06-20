# Janet V14 Project

Janet V14.11 is a modular rebuild of the V13 robot controller.  The aim is to keep the working features from Janet V13.30, but make the code much easier to read, maintain, and extend.

## Run

```bash
cd JanetV14_project
source ~/JanetEnv/bin/activate
python3 run_janet.py
```

Then open:

```text
http://<janet-pi-ip>:5000
```

## Folder layout

```text
JanetV14_project/
├── run_janet.py                  # Launch Janet
├── janet_core/                   # Robot logic modules
│   ├── config.py                 # All major constants and paths
│   ├── state.py                  # Shared state object
│   ├── motors.py                 # Arduino/I2C motor control and routines
│   ├── sonars.py                 # HC-SR04 distance sensors
│   ├── speech.py                 # Speaker output / espeak-ng / aplay
│   ├── voice.py                  # Microphone / arecord / speech recognition
│   ├── faces.py                  # Known faces and face recognition
│   ├── objects.py                # Known object memory and object thumbnails
│   ├── vision.py                 # OAK-D camera, detections, overlays
│   ├── hermes.py                 # Hermes API client
│   └── controller.py             # Wires all modules together
├── janet_web/                    # Flask web application
│   ├── app.py                    # Routes/API endpoints
│   ├── templates/index.html      # Web page
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── data/                         # Janet memory and captured media
│   ├── known_faces/
│   ├── known_objects/
│   ├── screenshots/
│   ├── speech_samples/
│   └── voice_samples/
├── arduino/                      # Arduino reference firmware
└── legacy/                       # The source V13.30 file for reference
```

## Main retained features

- OAK-D front camera with YOLO object detection.
- Object memory with saved thumbnails and 30 minute object acknowledgement cooldown.
- Face recognition with known face memory and 10 minute face greeting cooldown.
- Speech output through the working Y02 USB speaker using `espeak-ng` and `aplay`.
- Voice input through `arecord` and `SpeechRecognition`.
- Motor control over Arduino I2C.
- HC-SR04 sonar distance sensors.
- Web UI split into HTML, CSS, and JavaScript.
- Hermes AI tab and OpenAI-compatible API client.

## Notes

- The object images included in this zip are placeholder thumbnails when the original object crops were not uploaded. Janet will replace/add real object thumbnails as she sees objects again.
- The face thumbnails were regenerated from the uploaded `known_faces.npz` templates.
- Janet V14 keeps V13.30 in `legacy/` for reference only. The app runs from the modular V14 code.

## V14.1 Hermes notes

Hermes/BenderJanet said Janet should send a simple OpenAI-compatible request:

```http
POST http://192.168.50.186:8642/v1/chat/completions
Content-Type: application/json
```

```json
{
  "model": "gemma4:31b-cloud",
  "messages": [
    {"role": "user", "content": "What is the weather forecast for today?"}
  ]
}
```

V14.1 now uses that exact payload shape by default and includes a Hermes → Network Diagnostics button.

If Janet shows a TCP timeout, Hermes is not reachable from the Raspberry Pi on port 8642. On the Hermes machine, check:

```bash
ss -ltnp | grep 8642
cat ~/.hermes/.env | grep API_SERVER
```

The API server must bind to LAN, normally:

```bash
API_SERVER_ENABLED=true
API_SERVER_HOST=0.0.0.0
API_SERVER_PORT=8642
API_SERVER_KEY=change-me-local-dev
```

Then restart Hermes gateway.

## Janet V14.2 Hermes notes

V14.2 keeps the OpenAI-compatible Hermes request format:

```http
POST http://192.168.50.186:8642/v1/chat/completions
Content-Type: application/json
Authorization: Bearer change-me-local-dev
```

```json
{
  "model": "gemma4:31b-cloud",
  "messages": [{"role": "user", "content": "Hello Hermes"}],
  "stream": false,
  "max_tokens": 300
}
```

The Hermes client disables environment proxy use for LAN requests (`requests.Session().trust_env = False`). This stops proxy variables accidentally affecting `192.168.x.x` calls.

The Hermes tab now has **Discover Hermes**, which scans common local ports on the configured host, including 8642, 8000, 9119, 3000, 8080, 5000, 5173, 11434, 7860, and 9000. It tests TCP first, then HTTP health paths, then chat endpoints.

If Janet reports TCP timeout, Hermes has not received the request. In that case check on the Hermes machine:

```bash
ss -ltnp | grep -E '8642|8000|9119'
cat ~/.hermes/.env | grep API_SERVER
```

The API server must listen on `0.0.0.0:8642` or the LAN IP, not just `127.0.0.1:8642`.


## Hermes / Bender API

Janet V14.3 is preconfigured to use the working OpenAI-compatible Hermes endpoint:

```text
http://192.168.50.186:8642/v1/chat/completions
```

Default Hermes settings are stored in `data/hermes_settings.json`:

```json
{
  "base_url": "http://192.168.50.186:8642",
  "api_key": "change-me-local-dev",
  "model": "gemma4:31b-cloud",
  "endpoint": "/v1/chat/completions"
}
```

Use **Hermes → Discover / Repair Hermes** only if the IP, port, or endpoint changes. General questions send only the user question to Hermes. The **What can Janet see?** and **Robot status** buttons include Janet context.


## Natural TTS / optional ElevenLabs

Janet V14.7 uses **Edge TTS** by default for natural English + Chinese speech. ElevenLabs is no longer required for Janet's voice.

Install the Edge TTS voice support on the Pi:

```bash
source ~/JanetEnv/bin/activate
pip install edge-tts
sudo apt install -y ffmpeg
```

Default natural voices:

- English: `en-GB-RyanNeural`
- Chinese: `zh-CN-XiaoxiaoNeural`

Speech behaviour:

1. Janet speaks the English text.
2. Janet repeats it in Chinese.
3. WAV files are cached locally so repeated names, objects, and common vocabulary do not regenerate every time.

Cached speech is stored in:

```text
data/edge_tts_cache/
data/elevenlabs_cache/        # only used if you enable ElevenLabs manually
```

ElevenLabs is now optional. Janet no longer reads `ElevenLabs.txt`, `API-ElevenLabs.txt`, or other API key files automatically. To use ElevenLabs later, open:

```text
Skills -> Natural TTS
```

Then paste the `sk_...` key into the ElevenLabs API key field, enable ElevenLabs, and choose the ElevenLabs engine. The key is saved locally in:

```text
data/elevenlabs_settings.json
```

On free ElevenLabs accounts, library voices such as Bren/Sage may return `paid_plan_required` / HTTP 402. In that case, keep the engine set to **Edge TTS natural English + Chinese**.

## Startup greeting

Janet V14.7 says a bilingual boot message a couple of seconds after startup:

English: `Hello, I am Bender-Janet. I have just woken up, and my speaker is working.`

Chinese: `你好，我是 Bender-Janet。我刚刚醒来，我的扬声器正在工作。`

The message uses the active Natural TTS engine, normally Edge TTS, and is cached after the first generation.

## V14.8 boot routine

Janet now runs a controlled boot sequence before normal face/object acknowledgement starts:

1. speaks the wake-up/speaker check in English and Chinese;
2. performs a very small forward/back motor command test without checking sonar values;
3. starts the OAK-D camera and announces the active detection model;
4. checks all sonar sensors but only announces the front sensor value;
5. discovers/checks Hermes with short success/issue feedback, then asks for tomorrow's Bournemouth weather forecast in one short sentence;
6. announces boot complete and then enables normal listening/recognition.

During boot, the camera stream may show a placeholder and face/object greetings are paused so Janet does not start acknowledging detections before the wake-up routine has finished.


## V14.10 boot tweaks

- Step 2 now confirms the forward/back motor commands only; it no longer uses sonar readings to verify movement.
- Step 4 still checks the sonar system, but Janet only speaks the front sensor value so the boot speech is shorter.
- Step 5 gives shorter Hermes success/issue feedback and asks for a one-sentence Bournemouth weather forecast.
- Default ALSA speaker volume target increased from 85% to 95%.


## V14.10 boot streamlining

V14.11 starts the camera immediately at boot, removes the English-to-Chinese pause, sets ALSA volume to 100%, says only short boot confirmations, checks sonars without announcing values, and changes the Settings tab into a left-hand Skills menu.


## V14.11 quick boot and Skills UI

- Camera starts first at boot so the OAK-D warms up while Janet speaks and tests motors.
- English and Chinese speech now play back-to-back with no configured pause.
- ALSA speaker volume target is now 100%.
- Sonar boot check says whether sonars work, without reading all values aloud.
- Hermes now runs a boot-time connect/repair check and retries discovery automatically on any failed ask.
- The old Settings tab is now Skills, with the skill list on the left and the selected skill options on the right.
