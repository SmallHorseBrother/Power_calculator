import os
import uuid
import threading
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# job_id -> {'status', 'progress', 'results', 'output_video', 'error'}
jobs = {}


def update_job_progress(job_id, total_frames):
    total = max(int(total_frames or 0), 1)

    def _callback(processed_frames):
        pct = min(15 + int((processed_frames / total) * 80), 95)
        jobs[job_id]['progress'] = pct

    return _callback


def run_analysis(job_id, video_path, height_cm, weight_kg, mode, output_video_path):
    try:
        jobs[job_id]['status'] = 'processing'

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()

        if mode == 'muscleup':
            from power_calculator_mu import process_muscle_up_video
            analysis_kwargs = {}
        else:
            from power_calculator_pullup_with_report import process_muscle_up_video
            analysis_kwargs = {
                'report_path': os.path.join(app.config['OUTPUT_FOLDER'], f'{job_id}_report.html')
            }

        results = process_muscle_up_video(
            video_path=video_path,
            height_cm=height_cm,
            weight_kg=weight_kg,
            output_video_path=output_video_path,
            progress_callback=update_job_progress(job_id, total_frames),
            **analysis_kwargs,
        )

        jobs[job_id]['status'] = 'done'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['results'] = results or []
        jobs[job_id]['output_video'] = output_video_path
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    file = request.files['video']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    ext = os.path.splitext(secure_filename(file.filename))[1].lower()
    allowed_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    if ext not in allowed_exts:
        return jsonify({'error': 'Unsupported video format'}), 400

    try:
        height_cm = float(request.form.get('height', 175))
        weight_kg = float(request.form.get('weight', 70))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid height or weight'}), 400

    if not 100 <= height_cm <= 250:
        return jsonify({'error': 'Height must be between 100 and 250 cm'}), 400
    if not 30 <= weight_kg <= 250:
        return jsonify({'error': 'Weight must be between 30 and 250 kg'}), 400

    mode = request.form.get('mode', 'pullup').lower()
    if mode not in {'pullup', 'muscleup'}:
        return jsonify({'error': 'Invalid mode'}), 400

    job_id = str(uuid.uuid4())
    ext = ext or '.mp4'
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{job_id}{ext}')
    output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{job_id}_output.mp4')

    file.save(video_path)
    jobs[job_id] = {'status': 'queued', 'progress': 5, 'results': [], 'output_video': None, 'error': None}

    t = threading.Thread(
        target=run_analysis,
        args=(job_id, video_path, height_cm, weight_kg, mode, output_video_path),
        daemon=True
    )
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'results': job['results'],
        'error': job['error'],
    })


@app.route('/download/<job_id>')
def download(job_id):
    job = jobs.get(job_id)
    if not job or not job['output_video'] or not os.path.exists(job['output_video']):
        return jsonify({'error': 'Video not found'}), 404
    return send_file(job['output_video'], as_attachment=True, download_name='analysis_output.mp4')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
