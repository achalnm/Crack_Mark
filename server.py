import os
import io
import threading
from pathlib import Path

from flask import Flask, request, Response, send_file, jsonify
import cv2
import numpy as np

from crack_segment import remove_illumination, gaussian_tophat, shape_filter
from skimage.morphology import remove_small_holes, closing, disk
from skimage.filters import apply_hysteresis_threshold

BACKEND    = os.environ.get('CRACKMARK_BACKEND', 'dl').lower()
MODEL_NAME = os.environ.get('CRACKMARK_MODEL',   'default')
CKPT_ENV   = os.environ.get('CRACKMARK_CKPT',    '')
PORT       = int(os.environ.get('CRACKMARK_PORT', 5001))

BASE           = Path(__file__).parent
FINETUNED_CKPT = BASE / 'crackmark_finetuned.pt'

from PIL import Image as _PILImg
try:
    _PIL_BILINEAR = _PILImg.Resampling.BILINEAR
    _PIL_NEAREST  = _PILImg.Resampling.NEAREST
except AttributeError:
    _PIL_BILINEAR = _PILImg.BILINEAR
    _PIL_NEAREST  = _PILImg.NEAREST

app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


def run_pipeline(gray: np.ndarray, high_mult: float = 0.30) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    norm = remove_illumination(denoised, sigma=100)
    saliency = np.maximum(
        gaussian_tophat(norm, sigma=8),
        gaussian_tophat(norm, sigma=16),
    )
    bright     = (norm > 175).astype(np.uint8)
    bright_dil = cv2.dilate(bright, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
    saliency[bright_dil > 0] = 0
    peak        = int(saliency.max())
    thresh_high = max(8, int(peak * high_mult))
    thresh_low  = max(3, int(peak * high_mult / 3.0))
    binary = apply_hysteresis_threshold(saliency.astype(np.float64), thresh_low, thresh_high)
    closed = closing(binary, disk(5))
    filled = remove_small_holes(closed, max_size=500)
    return (shape_filter(filled, gray) * 255).astype(np.uint8)


_model = None
_device = None
_model_lock = threading.Lock()
_active_backend = 'classical'

if BACKEND == 'dl':
    try:
        import torch
        from model import build_model, load_model

        if torch.backends.mps.is_available():
            _device = torch.device('mps')
        elif torch.cuda.is_available():
            _device = torch.device('cuda')
        else:
            _device = torch.device('cpu')

        _model = build_model(MODEL_NAME).to(_device)
        _model.eval()

        _ckpt_path = None
        _weights_label = 'ImageNet init (no crack checkpoint)'

        if FINETUNED_CKPT.exists():
            _ckpt_path = FINETUNED_CKPT
            _weights_label = FINETUNED_CKPT.name
        elif CKPT_ENV:
            p = Path(CKPT_ENV)
            _ckpt_path = p if p.is_absolute() else BASE / p
            _weights_label = _ckpt_path.name

        if _ckpt_path:
            load_model(_model, _ckpt_path, _device)

        _active_backend = 'dl'
        print(f"\n  CrackMark  |  backend: DL  |  device: {_device}  |  weights: {_weights_label}")

    except Exception as err:
        print(f"\n  CrackMark  |  DL init failed ({err}), falling back to classical")
        _model = None
        _device = None
        _active_backend = 'classical'

if _active_backend == 'classical':
    print(f"\n  CrackMark  |  backend: classical")


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

    raw = request.files['image'].read()
    high_mult = float(request.form.get('sensitivity', 0.30))
    high_mult = max(0.05, min(0.70, high_mult))

    if _active_backend == 'dl':
        return _analyze_dl(raw, high_mult)
    return _analyze_classical(raw, high_mult)


def _analyze_classical(raw: bytes, high_mult: float) -> Response:
    data = np.frombuffer(raw, np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        try:
            from PIL import Image
            gray = np.array(Image.open(io.BytesIO(raw)).convert('L'))
        except Exception as e:
            return Response(f'Cannot decode image: {e}', status=400)

    mask = run_pipeline(gray, high_mult)
    _, png = cv2.imencode('.png', mask)
    return Response(png.tobytes(), mimetype='image/png')


def _analyze_dl(raw: bytes, high_mult: float) -> Response:
    if _model is None:
        return _analyze_classical(raw, high_mult)

    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image as PILImage

    try:
        pil_img = PILImage.open(io.BytesIO(raw)).convert('RGB')
    except Exception as e:
        return Response(f'Cannot decode image: {e}', status=400)

    orig_w, orig_h = pil_img.size
    img_512 = pil_img.resize((512, 512), _PIL_BILINEAR)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_t = normalize(transforms.ToTensor()(img_512)).unsqueeze(0).to(_device)

    with _model_lock:
        with torch.no_grad():
            logits = _model(img_t)

    prob = torch.softmax(logits, dim=1)[:, 1:2]
    thresh = max(0.05, min(0.95, high_mult))
    binary = (prob > thresh).float()
    binary_up = F.interpolate(binary, size=(orig_h, orig_w), mode='nearest')
    mask_np = (binary_up[0, 0].cpu().numpy() * 255).astype(np.uint8)

    _, png = cv2.imencode('.png', mask_np)
    return Response(png.tobytes(), mimetype='image/png')


@app.route('/finetune', methods=['POST'])
def finetune():
    if _active_backend != 'dl' or _model is None:
        return jsonify({'loss': None, 'steps': 0, 'skipped': 'classical backend'})

    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image as PILImage

    if 'image' not in request.files or 'mask' not in request.files:
        return Response('Missing image or mask fields', status=400)

    steps = max(1, int(request.form.get('steps', 5)))
    lr    = float(request.form.get('lr', 1e-4))

    try:
        pil_img = PILImage.open(io.BytesIO(request.files['image'].read())).convert('RGB')
    except Exception as e:
        return Response(f'Cannot decode image: {e}', status=400)

    try:
        pil_mask = PILImage.open(io.BytesIO(request.files['mask'].read())).convert('L')
    except Exception as e:
        return Response(f'Cannot decode mask: {e}', status=400)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    img_t = normalize(
        transforms.ToTensor()(pil_img.resize((512, 512), _PIL_BILINEAR))
    ).unsqueeze(0).to(_device)

    mask_np = np.array(pil_mask.resize((512, 512), _PIL_NEAREST))
    mask_t = torch.from_numpy((mask_np > 127).astype(np.int64)).unsqueeze(0).to(_device)

    try:
        with _model_lock:
            has_split = hasattr(_model, 'encoder_params') and hasattr(_model, 'decoder_params')

            if has_split:
                dec_params = list(_model.decoder_params())
                enc_params = list(_model.encoder_params())
                optimizer = torch.optim.Adam([
                    {'params': dec_params, 'lr': lr},
                    {'params': enc_params, 'lr': lr * 0.1},
                ])
            else:
                optimizer = torch.optim.Adam(_model.parameters(), lr=lr)

            ce_weight = torch.tensor([1.0, 10.0], device=_device)
            total_loss = 0.0

            _model.train()
            if has_split and hasattr(_model, 'encoder'):
                _model.encoder.eval()

            try:
                for _ in range(steps):
                    optimizer.zero_grad()
                    logits = _model(img_t)

                    ce_loss = F.cross_entropy(logits, mask_t, weight=ce_weight)

                    prob = torch.softmax(logits, dim=1)[:, 1]
                    target = mask_t.float()
                    inter = (prob * target).sum()
                    dice = 1.0 - (2.0 * inter + 1.0) / (prob.sum() + target.sum() + 1.0)

                    loss = 0.5 * ce_loss + 0.5 * dice
                    loss.backward()

                    clip_params = dec_params if has_split else list(_model.parameters())
                    torch.nn.utils.clip_grad_norm_(clip_params, 1.0)

                    optimizer.step()
                    total_loss += loss.item()
            finally:
                _model.eval()

            torch.save(_model.state_dict(), FINETUNED_CKPT)

    except Exception as e:
        return Response(f'Fine-tune error: {e}', status=500)

    return jsonify({'loss': total_loss / steps, 'steps': steps})


if __name__ == '__main__':
    print(f"  Open http://localhost:{PORT} in Chrome\n")
    app.run(host='127.0.0.1', port=PORT, debug=False, threaded=True)
