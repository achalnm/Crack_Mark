import io
import os
import sys
import time
import threading

import numpy as np
import cv2
import requests


def dummy_image(w=256, h=256):
    img = np.tile(np.linspace(80, 200, w, dtype=np.uint8), (h, 1))
    img[h // 2 - 2 : h // 2 + 2, :] = 30
    _, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def dummy_mask(w=256, h=256):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 2 - 2 : h // 2 + 2, :] = 255
    _, buf = cv2.imencode('.png', mask)
    return buf.tobytes()


BASE_URL = f"http://127.0.0.1:{os.environ.get('CRACKMARK_PORT', 5001)}"
results  = []


def check(name, passed, detail=''):
    mark = 'PASS' if passed else 'FAIL'
    print(f"  {mark}  {name}" + (f"\n        {detail}" if detail else ''))
    results.append((name, passed))


_ready  = threading.Event()
_errors = []


def _start_server():
    try:
        import server as srv
        original = srv.app.run
        def _run(*a, **kw):
            _ready.set()
            original(*a, **kw)
        srv.app.run = _run
        srv.app.run(host='127.0.0.1', port=int(os.environ.get('CRACKMARK_PORT', 5001)),
                    debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        _errors.append(e)
        _ready.set()


threading.Thread(target=_start_server, daemon=True).start()
print('\nStarting server...')
_ready.wait(timeout=60)

if _errors:
    print(f'Server failed: {_errors[0]}')
    sys.exit(1)

time.sleep(1.5)

print('\n-- /ping --')
try:
    r = requests.get(BASE_URL + '/ping', timeout=5)
    check('ping', r.status_code == 200 and r.text == 'ok')
except Exception as e:
    check('ping', False, str(e))

print('\n-- /analyze --')
img_bytes = dummy_image()
try:
    r = requests.post(
        BASE_URL + '/analyze',
        files={'image': ('test.png', io.BytesIO(img_bytes), 'image/png')},
        data={'sensitivity': '0.30'},
        timeout=60,
    )
    if r.status_code == 200 and r.headers.get('Content-Type', '').startswith('image/png'):
        arr  = np.frombuffer(r.content, np.uint8)
        mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        check('returns PNG mask', mask is not None, f'shape={getattr(mask, "shape", None)}')
        check('mask same size as input', mask is not None and mask.shape == (256, 256))
    else:
        check('returns PNG mask', False, f'status={r.status_code}')
        check('mask same size as input', False)
except Exception as e:
    check('returns PNG mask', False, str(e))
    check('mask same size as input', False)

print('\n-- /finetune --')
mask_bytes = dummy_mask()
try:
    r = requests.post(
        BASE_URL + '/finetune',
        files={
            'image': ('test.png', io.BytesIO(img_bytes),  'image/png'),
            'mask':  ('mask.png', io.BytesIO(mask_bytes), 'image/png'),
        },
        data={'steps': '3'},
        timeout=120,
    )
    if r.status_code == 200:
        data = r.json()
        if data.get('skipped'):
            check('classical skip response', True, str(data))
            check('no checkpoint (classical, expected)', True)
        else:
            check('returns loss', isinstance(data.get('loss'), float), str(data))
            import pathlib
            check('checkpoint written', pathlib.Path('crackmark_finetuned.pt').exists())
    else:
        check('returns 200', False, f'status={r.status_code}  {r.text[:200]}')
        check('checkpoint written', False)
except Exception as e:
    check('returns 200', False, str(e))
    check('checkpoint written', False)

print('\n-- load_model key tolerance --')
try:
    import torch
    import pathlib
    from model import build_model, load_model

    m  = build_model('default')
    sd = m.state_dict()
    keys = list(sd.keys())

    partial = {k: sd[k] for k in keys[:len(keys) // 2]}
    partial['criterion.weight'] = torch.tensor([1.0])
    lightning = {'state_dict': {'model.' + k: v for k, v in partial.items()}}

    tmp = pathlib.Path('_smoke_ckpt.pt')
    torch.save(lightning, tmp)
    load_model(m, tmp, torch.device('cpu'))
    tmp.unlink()
    check('survives partial Lightning checkpoint', True)
except ImportError:
    check('survives partial Lightning checkpoint', True, '(no torch, classical only)')
except Exception as e:
    check('survives partial Lightning checkpoint', False, str(e))

print('\n' + '-' * 50)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f'  {passed}/{total} passed')
if passed < total:
    for name, ok in results:
        if not ok:
            print(f'    FAIL: {name}')
print()
sys.exit(0 if passed == total else 1)
