import os
import time
from flask import Flask, Response, jsonify, render_template, request, send_file
from janet_core import config


def create_app(janet):
    app = Flask(__name__, template_folder='templates', static_folder='static')

    @app.route('/')
    def index():
        return render_template('index.html', version=config.APP_VERSION)

    @app.route('/video_front')
    def video_front():
        return Response(janet.vision.generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/video_rear')
    def video_rear():
        return Response(janet.vision.generate_rear_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/readings')
    @app.route('/readings_basic')
    def readings():
        data = janet.state.get()
        data['motor_settings'] = janet.motors.get_settings()
        data['routines'] = janet.motors.get_routine_status()
        return jsonify(data)

    @app.route('/move/<direction>')
    def move(direction):
        ok, msg = janet.motors.execute_move_async(direction, duration=request.args.get('duration'), acceleration=request.args.get('acceleration'))
        return jsonify({'status': 'ok' if ok else 'error', 'message': msg, 'direction': direction}), (200 if ok else 500)

    @app.route('/motor_settings', methods=['GET','POST'])
    def motor_settings():
        if request.method == 'POST':
            p=request.get_json(silent=True) or {}
            s=janet.motors.set_settings(duration=p.get('duration'), acceleration=p.get('acceleration'), preset='custom')
            return jsonify({**s, 'motors_available': janet.motors.motors is not None, 'message': 'Motor settings saved'})
        return jsonify({**janet.motors.get_settings(), 'motors_available': janet.motors.motors is not None, 'presets': config.MOTOR_PRESETS})

    @app.route('/motor_preset/<preset>', methods=['POST'])
    def motor_preset(preset):
        ok,msg=janet.motors.apply_preset(preset)
        return jsonify({'status':'ok' if ok else 'error', 'message':msg, **janet.motors.get_settings()}), (200 if ok else 404)

    @app.route('/routine_info')
    def routine_info(): return jsonify({'status':'ok', 'routines': janet.motors.get_routine_status()})

    @app.route('/routine_start/<routine_id>', methods=['POST'])
    def routine_start(routine_id):
        ok,msg=janet.motors.start_routine(routine_id)
        return jsonify({'status':'ok' if ok else 'error','message':msg,'routines':janet.motors.get_routine_status()}), (200 if ok else 409)

    @app.route('/routine_stop', methods=['POST'])
    def routine_stop():
        janet.motors.stop_routine(); return jsonify({'status':'ok','message':'Routine stop requested','routines':janet.motors.get_routine_status()})


    @app.route('/boot_status')
    def boot_status():
        return jsonify({'status': 'ok', 'boot': janet.state.section('boot')})

    @app.route('/boot_start', methods=['POST'])
    def boot_start():
        started = janet.boot.start()
        return jsonify({'status': 'ok' if started else 'busy', 'message': 'Boot routine started' if started else 'Boot routine already running', 'boot': janet.state.section('boot')})

    @app.route('/speech_settings', methods=['GET','POST'])
    def speech_settings():
        if request.method=='POST':
            p=request.get_json(silent=True) or {}
            s=janet.speech.set_config(device=p.get('device'), rate=p.get('rate'), channels=p.get('channels'))
            return jsonify({'status':'ok','message':f"Speaker set to {s['device']}", **s})
        return jsonify(janet.speech.get_config())

    @app.route('/speech_devices')
    def speech_devices(): return jsonify({'status':'ok','devices':janet.speech.list_speakers(), 'suggested_device': janet.speech.get_config()['device']})

    @app.route('/speech_test_beep', methods=['POST'])
    def speech_test_beep():
        janet.speech.test_beep(); return jsonify({'status':'queued','message':'Speaker beep queued'})

    @app.route('/speech_set_volume', methods=['POST'])
    def speech_set_volume():
        ok,msg=janet.speech.set_volume(config.SPEECH_VOLUME_PERCENT)
        return jsonify({'status':'ok' if ok else 'error','message':msg})

    @app.route('/speech_say', methods=['POST'])
    def speech_say():
        p=request.get_json(silent=True) or {}
        janet.speech.say_async(p.get('text') or config.SPEECH_TEST_PHRASE)
        return jsonify({'status':'queued','message':'Speech queued'})


    @app.route('/elevenlabs_settings', methods=['GET','POST'])
    def elevenlabs_settings():
        if request.method == 'POST':
            p = request.get_json(silent=True) or {}
            return jsonify({'status': 'ok', 'settings': janet.speech.set_elevenlabs_config(
                engine=p.get('engine'),
                api_key=p.get('api_key'),
                clear_api_key=p.get('clear_api_key'),
                enabled=p.get('enabled'),
                bilingual=p.get('bilingual'),
                translate_with_hermes=p.get('translate_with_hermes'),
                english_voice_id=p.get('english_voice_id'),
                chinese_voice_id=p.get('chinese_voice_id'),
                model_id=p.get('model_id'),
                output_format=p.get('output_format'),
                cache_enabled=p.get('cache_enabled'),
                edge_enabled=p.get('edge_enabled'),
                edge_english_voice=p.get('edge_english_voice'),
                edge_chinese_voice=p.get('edge_chinese_voice'),
            )})
        return jsonify(janet.speech.elevenlabs_info(include_key=False))

    @app.route('/elevenlabs_test', methods=['POST'])
    def elevenlabs_test():
        p = request.get_json(silent=True) or {}
        ok, msg = janet.speech.test_elevenlabs(p.get('text') or config.SPEECH_TEST_PHRASE)
        return jsonify({'status': 'queued' if ok else 'error', 'message': msg, 'info': janet.speech.elevenlabs_info(include_key=False)})

    @app.route('/elevenlabs_precache', methods=['POST'])
    def elevenlabs_precache():
        face_names = []
        try:
            for item in janet.faces.summary():
                if item.get('name'):
                    face_names.append(item.get('name'))
        except Exception:
            pass
        object_labels = []
        try:
            for item in janet.objects.summary():
                if item.get('label'):
                    object_labels.append(item.get('label'))
        except Exception:
            pass
        return jsonify(janet.speech.precache_vocabulary(names=face_names, objects=object_labels))

    @app.route('/elevenlabs_cache_info')
    def elevenlabs_cache_info():
        return jsonify(janet.speech.elevenlabs_info(include_key=False))

    @app.route('/voice_settings', methods=['GET','POST'])
    def voice_settings():
        if request.method=='POST':
            p=request.get_json(silent=True) or {}
            return jsonify({'status':'ok', **janet.voice.set_config(device=p.get('device'), rate=p.get('rate'), channels=p.get('channels'))})
        return jsonify(janet.voice.get_config())

    @app.route('/voice_devices')
    def voice_devices(): return jsonify({'status':'ok','devices':janet.voice.list_devices(), 'suggested_device':janet.voice.get_config()['device']})

    @app.route('/face_info')
    def face_info(): return jsonify({'status':'ok','face':janet.state.section('face')})

    @app.route('/face_settings', methods=['POST'])
    def face_settings():
        p=request.get_json(silent=True) or {}; s=janet.faces.set_settings(enabled=p.get('enabled'), threshold=p.get('threshold'))
        return jsonify({'status':'ok','settings':s,'message':'Face settings saved'})

    @app.route('/face_add', methods=['POST'])
    def face_add():
        p=request.get_json(silent=True) or {}; frame=janet.vision.latest_frame
        ok,msg=janet.faces.add_from_frame(p.get('name'), frame) if frame is not None else (False,'No camera frame available yet')
        return jsonify({'status':'ok' if ok else 'error','message':msg,'known_faces':janet.faces.summary()}), (200 if ok else 400)

    @app.route('/face_upload', methods=['POST'])
    def face_upload():
        name=request.form.get('name','').strip(); file=request.files.get('photo')
        if not file: return jsonify({'status':'error','message':'No photo uploaded'}), 400
        ok,msg=janet.faces.add_from_uploaded_file(name, file)
        return jsonify({'status':'ok' if ok else 'error','message':msg,'known_faces':janet.faces.summary()}), (200 if ok else 400)

    @app.route('/face_remove', methods=['POST'])
    def face_remove():
        p=request.get_json(silent=True) or {}; ok,msg=janet.faces.remove_name(p.get('name',''))
        return jsonify({'status':'ok' if ok else 'error','message':msg,'known_faces':janet.faces.summary()}), (200 if ok else 404)

    @app.route('/face_remove_sample', methods=['POST'])
    def face_remove_sample():
        p=request.get_json(silent=True) or {}; ok,msg=janet.faces.remove_sample(p.get('sample_index'))
        return jsonify({'status':'ok' if ok else 'error','message':msg,'known_faces':janet.faces.summary()}), (200 if ok else 404)

    @app.route('/face_image/<path:filename>')
    def face_image(filename):
        path=config.FACE_IMAGE_DIR/os.path.basename(filename)
        return send_file(path, mimetype='image/jpeg') if path.exists() else ('',404)

    @app.route('/object_info')
    def object_info(): return jsonify({'status':'ok','object':janet.state.section('object')})

    @app.route('/object_remove', methods=['POST'])
    def object_remove():
        p=request.get_json(silent=True) or {}; ok,msg=janet.objects.remove_label(p.get('label',''))
        return jsonify({'status':'ok' if ok else 'error','message':msg,'known_objects':janet.objects.summary()}), (200 if ok else 404)

    @app.route('/object_remove_sample', methods=['POST'])
    def object_remove_sample():
        p=request.get_json(silent=True) or {}; ok,msg=janet.objects.remove_sample(sample_index=p.get('sample_index'), sample_id=p.get('sample_id'))
        return jsonify({'status':'ok' if ok else 'error','message':msg,'known_objects':janet.objects.summary()}), (200 if ok else 404)

    @app.route('/object_image/<path:filename>')
    def object_image(filename):
        path=config.OBJECT_IMAGE_DIR/os.path.basename(filename)
        return send_file(path, mimetype='image/jpeg') if path.exists() else ('',404)

    @app.route('/detection_info')
    def detection_info():
        data=janet.state.get()
        safe_candidates = [c for c in config.DETECTION_MODEL_CANDIDATES if c.get('type') == 'known-good']
        return jsonify({'status':'ok','model':data.get('detection_model',{}),'fps':data.get('fps',0),'detections':data.get('detections',[]),'candidates':safe_candidates})

    @app.route('/detection_scan', methods=['POST'])
    def detection_scan():
        results=[c for c in config.DETECTION_MODEL_CANDIDATES if c.get('type') == 'known-good']
        janet.state.update('detection_model', scan_results=results, last_scan_message=f'{len(results)} configured candidate model(s) listed')
        return jsonify({'status':'ok','message':'Stable model candidates listed. Risky model switching is disabled to prevent camera crashes.','results':results})

    @app.route('/detection_use_model', methods=['POST'])
    def detection_use_model():
        p=request.get_json(silent=True) or {}
        requested = str(p.get('model') or '').strip()
        safe_models = {c.get('name') for c in config.DETECTION_MODEL_CANDIDATES if c.get('type') == 'known-good'}
        if requested not in safe_models:
            return jsonify({'status':'error','message':f'Model {requested} is disabled for stability. Use OAK-D showcase modes for advanced visuals.'}), 400
        ok,msg=janet.vision.set_model(requested)
        return jsonify({'status':'ok' if ok else 'error','message':msg})

    @app.route('/hermes_settings', methods=['GET','POST'])
    def hermes_settings():
        if request.method=='POST':
            p=request.get_json(silent=True) or {}
            return jsonify({'status':'ok','settings':janet.hermes.set_config(base_url=p.get('base_url'), api_key=p.get('api_key'), model=p.get('model'), endpoint=p.get('endpoint'))})
        return jsonify(janet.hermes.info())

    @app.route('/hermes_button_test')
    def hermes_button_test(): return jsonify({'status':'ok','message':'Janet button route is alive'})

    @app.route('/hermes_quick_check')
    def hermes_quick_check(): return jsonify(janet.hermes.quick_check())

    @app.route('/hermes_repair', methods=['POST'])
    def hermes_repair(): return jsonify(janet.hermes.connect_or_repair(do_chat=True, save=True))

    @app.route('/hermes_probe', methods=['POST'])
    def hermes_probe(): return jsonify(janet.hermes.probe())

    @app.route('/hermes_diagnostics')
    def hermes_diagnostics(): return jsonify(janet.hermes.diagnostics(do_chat=False))

    @app.route('/hermes_discover', methods=['GET','POST'])
    def hermes_discover(): return jsonify(janet.hermes.discover(do_chat=True))

    @app.route('/hermes_curl_example')
    def hermes_curl_example():
        question=request.args.get('q') or 'Reply with OK if Hermes can hear Janet.'
        return Response(janet.hermes.curl_example(question), mimetype='text/plain; charset=utf-8')

    @app.route('/hermes_raw_chat_test', methods=['GET','POST'])
    def hermes_raw_chat_test(): return jsonify(janet.hermes.ask('Reply with OK if Hermes can hear Janet.', max_tokens=40))

    @app.route('/hermes_ask', methods=['POST'])
    def hermes_ask():
        p=request.get_json(silent=True) or {}
        question=p.get('question') or 'Hello Hermes'
        # General questions should go to Hermes exactly like Hermes requested:
        # model + messages + stream=false. Robot context is included only for
        # Janet status / vision helper buttons, otherwise it can bloat the prompt.
        include_context = bool(p.get('include_context', False))
        context = janet.build_service_report() if include_context else None
        result=janet.hermes.ask(question, context=context)
        if result.get('ok') and p.get('speak', True): janet.speech.say_async(result.get('answer',''))
        return jsonify(result)

    @app.route('/service_report')
    def service_report(): return Response(janet.build_service_report(), mimetype='text/plain; charset=utf-8')

    @app.route('/service_report.json')
    def service_report_json(): return jsonify({'status':'ok','report':janet.build_service_report(),'readings':janet.state.get()})

    @app.route('/screenshot_front')
    def screenshot_front():
        jpeg=janet.vision.get_latest_jpeg()
        if not jpeg: return jsonify({'status':'error','message':'No screenshot frame available yet'}),503
        config.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        fn=f"{config.SCREENSHOT_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"; path=config.SCREENSHOT_DIR/fn
        path.write_bytes(jpeg)
        return send_file(path, mimetype='image/jpeg', as_attachment=True, download_name=fn)

    @app.route('/favicon.ico')
    def favicon(): return ('',204)

    

    # OAK-D Examples routes
    @app.route('/oak_d_list')
    def oak_d_list():
        """List curated OAK-D showcase modes that are stable on Janet."""
        examples = [
            {'id': 'smart-detection', 'name': 'Smart Detection', 'description': 'Stable object detection mode'},
            {'id': 'objectron-3d', 'name': '3D Object Detection', 'description': 'Pseudo 3D bounding boxes inspired by Objectron demos'},
            {'id': 'object-distance', 'name': 'Distance To Objects', 'description': 'Approximate object distance overlay'},
            {'id': 'gaze-detection', 'name': 'Gaze Detection', 'description': 'Face/head gaze-style direction overlay'},
            {'id': 'human-pose', 'name': 'Human Pose', 'description': 'Pose-like stick figure overlay for people detections'},
            {'id': 'people-focus', 'name': 'People Focus', 'description': 'Filter detections to people only'},
            {'id': 'edge-vision', 'name': 'Edge Vision', 'description': 'Highlight contours and scene edges'},
            {'id': 'depth-style', 'name': 'Depth Style Heatmap', 'description': 'Pseudo depth-style color map overlay'},
            {'id': 'night-boost', 'name': 'Low Light Boost', 'description': 'Enhance contrast for darker scenes'},
            {'id': 'tracking-reticle', 'name': 'Tracking Reticle', 'description': 'Show lock-on reticle for strongest target'},
            {'id': 'social-distance', 'name': 'Social Distance', 'description': 'People spacing warnings inspired by spatial demos'},
        ]
        
        oak_state = janet.state.section('oak_d') or {}
        selected_modes = oak_state.get('selected_modes')
        if not isinstance(selected_modes, list):
            legacy_mode = str(oak_state.get('mode', 'normal') or 'normal')
            selected_modes = [] if legacy_mode == 'normal' else [legacy_mode]
        selected_modes = [str(m).strip() for m in selected_modes if str(m).strip() and str(m).strip() != 'normal']

        mode_to_example = {
            'objectron-3d': 'objectron-3d',
            'object-distance': 'object-distance',
            'gaze-detection': 'gaze-detection',
            'human-pose': 'human-pose',
            'people-focus': 'people-focus',
            'edge-vision': 'edge-vision',
            'depth-style': 'depth-style',
            'night-boost': 'night-boost',
            'tracking-reticle': 'tracking-reticle',
            'social-distance': 'social-distance',
        }
        selected_examples = [mode_to_example[m] for m in selected_modes if m in mode_to_example]
        current = selected_examples[0] if selected_examples else 'bender-janet'
        return jsonify({'status': 'ok', 'examples': examples, 'current': current, 'selected_modes': selected_modes, 'selected_examples': selected_examples})

    @app.route('/oak_d_example', methods=['POST'])
    def oak_d_example():
        """Apply a curated OAK-D showcase mode to the camera feed."""
        p = request.get_json(silent=True) or {}
        example_id = p.get('example', 'bender-janet')
        toggle = bool(p.get('toggle', True))
        mode_map = {
            'bender-janet': 'normal',
            'smart-detection': 'normal',
            'objectron-3d': 'objectron-3d',
            'object-distance': 'object-distance',
            'gaze-detection': 'gaze-detection',
            'human-pose': 'human-pose',
            'people-focus': 'people-focus',
            'edge-vision': 'edge-vision',
            'depth-style': 'depth-style',
            'night-boost': 'night-boost',
            'tracking-reticle': 'tracking-reticle',
            'social-distance': 'social-distance',
        }
        if example_id not in mode_map:
            return jsonify({'status': 'error', 'message': f'Unknown OAK-D example: {example_id}'}), 400

        # Do not restart the DepthAI pipeline for showcase mode changes.
        # We keep the detector model steady and only switch overlay/behavior mode.
        target_model = (janet.state.section('detection_model') or {}).get('name', config.DETECTION_MODEL_NAME)
        mode = mode_map[example_id]

        oak_state = janet.state.section('oak_d') or {}
        selected_modes = oak_state.get('selected_modes')
        if not isinstance(selected_modes, list):
            legacy_mode = str(oak_state.get('mode', 'normal') or 'normal')
            selected_modes = [] if legacy_mode == 'normal' else [legacy_mode]
        selected_modes = [str(m).strip() for m in selected_modes if str(m).strip() and str(m).strip() != 'normal']

        if mode == 'normal':
            selected_modes = []
            message = 'Returned to normal operating mode'
        elif toggle:
            if mode in selected_modes:
                selected_modes = [m for m in selected_modes if m != mode]
                message = f'Disabled OAK-D showcase: {example_id}'
            else:
                selected_modes.append(mode)
                message = f'Enabled OAK-D showcase: {example_id}'
        else:
            selected_modes = [mode]
            message = f'Applied OAK-D showcase: {example_id}'

        mode_to_example = {
            'objectron-3d': 'objectron-3d',
            'object-distance': 'object-distance',
            'gaze-detection': 'gaze-detection',
            'human-pose': 'human-pose',
            'people-focus': 'people-focus',
            'edge-vision': 'edge-vision',
            'depth-style': 'depth-style',
            'night-boost': 'night-boost',
            'tracking-reticle': 'tracking-reticle',
            'social-distance': 'social-distance',
        }
        selected_examples = [mode_to_example[m] for m in selected_modes if m in mode_to_example]
        current_example = selected_examples[0] if selected_examples else 'bender-janet'
        primary_mode = selected_modes[0] if selected_modes else 'normal'
        full_message = f'{message}. Active modes: {", ".join(selected_examples) if selected_examples else "none"}'
        janet.state.update('oak_d', current_example=current_example, selected_examples=selected_examples, selected_modes=selected_modes, mode=primary_mode, message=full_message, model=target_model)
        return jsonify({'status': 'ok', 'message': full_message, 'model': target_model, 'selected_examples': selected_examples, 'selected_modes': selected_modes})

    @app.route('/ai_detection_toggle', methods=['POST'])
    def ai_detection_toggle():
        """Toggle AI detection on/off"""
        p = request.get_json(silent=True) or {}
        enabled = p.get('enabled', True)
        janet.state.update('detection', ai_enabled=enabled, message=f'AI detection {"enabled" if enabled else "disabled"}')
        return jsonify({'status': 'ok', 'enabled': enabled, 'message': f'AI detection {"enabled" if enabled else "disabled"}'})

    return app
