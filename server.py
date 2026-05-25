from flask import Flask, request, Response, send_file
import cv2
import numpy as np
import io
from pathlib import Path

from crack_segment import remove_illumination, gaussian_tophat, shape_filter
from skimage.morphology import remove_small_holes, closing, disk
from skimage.filters import apply_hysteresis_threshold

app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
BASE = Path(__file__).parent


def run_pipeline(gray: np.ndarray, high_mult: float = 0.30) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    norm = remove_illumination(denoised, sigma=100)
    saliency = np.maximum(
        gaussian_tophat(norm, sigma=8),
        gaussian_tophat(norm, sigma=16),
    )
    bright = (norm > 175).astype(np.uint8)
    bright_dil = cv2.dilate(bright, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
    saliency[bright_dil > 0] = 0
    peak        = int(saliency.max())
    thresh_high = max(8, int(peak * high_mult))
    thresh_low  = max(3, int(peak * high_mult / 3.0))
    binary = apply_hysteresis_threshold(saliency.astype(np.float64), thresh_low, thresh_high)
    closed = closing(binary, disk(5))
    filled = remove_small_holes(closed, max_size=500)
    return (shape_filter(filled, gray) * 255).astype(np.uint8)


@app.route('/')
def index():
    return send_file(BASE / 'annotator.html')


@app.route('/annotator.css')
def serve_css():
    return send_file(BASE / 'annotator.css', mimetype='text/css')


@app.route('/annotator.js')
def serve_js():
    return send_file(BASE / 'annotator.js', mimetype='application/javascript')


@app.route('/ping')
def ping():
    return Response('ok', mimetype='text/plain')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return Response('No image field', status=400)
    raw  = request.files['image'].read()
    data = np.frombuffer(raw, np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        try:
            from PIL import Image
            gray = np.array(Image.open(io.BytesIO(raw)).convert('L'))
        except Exception as e:
            return Response(f'Cannot decode image: {e}', status=400)

    high_mult = float(request.form.get('sensitivity', 0.30))
    high_mult = max(0.05, min(0.70, high_mult))

    mask = run_pipeline(gray, high_mult)
    _, png = cv2.imencode('.png', mask)
    return Response(png.tobytes(), mimetype='image/png')


if __name__ == '__main__':
    print()
    print('  CrackMark — open http://localhost:5000 in Chrome')
    print()
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
