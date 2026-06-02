'use strict';

let images       = [];
let imageIndex   = 0;
let savedCount   = 0;
let outDirHandle = null;

let imgW = 0, imgH = 0;
let currentImage   = null;
let currentFileObj = null;
let objectURL      = null;

let rawIR  = { x: 0, y: 0, w: 0, h: 0 };
let maskIR = { x: 0, y: 0, w: 0, h: 0 };

let mlBinary      = null;
let mlFetchCtrl   = null;
let mlSensitivity = 30;

let strokes   = [];
let curStroke = null;
let isDrawing = false;
let activeIR  = null;
let drawMode  = 'draw';
let brushSize = 10;

let rawCtx = null, strokeCtx = null, maskCtx = null;

const get = id => document.getElementById(id);

const sImport = get('s-import');
const sLoaded = get('s-loaded');
const sEditor = get('s-editor');
const sDone   = get('s-done');

const fi         = get('fi');
const impErr     = get('imp-err');
const btnPickSrc = get('btn-pick-src');
const btnPickOut = get('btn-pick-out');
const btnStart   = get('btn-start');
const srcStatus  = get('src-status');
const outStatus  = get('out-status');
const impSrcStep = get('imp-src-step');
const impOutStep = get('imp-out-step');
const ldCount    = get('ld-count');
const ldFolder   = get('ld-folder');

const hdrFile  = get('hdr-file');
const hdrProg  = get('hdr-prog');
const mlChip   = get('ml-status');
const mlSlider = get('ml-sl');
const mlSlVal  = get('ml-sl-val');

const rawCw   = get('raw-cw');
const rawC    = get('raw-c');
const strokeC = get('stroke-c');
const maskCw  = get('mask-cw');
const maskC   = get('mask-c');

const btnDraw  = get('btn-draw');
const btnErase = get('btn-erase');
const brushSl  = get('brush-sl');
const brushNum = get('brush-num');
const btnUndo  = get('btn-undo');
const btnClr   = get('btn-clr');
const btnRmAI  = get('btn-rm-ai');

const btnSave  = get('btn-save');
const btnSkip  = get('btn-skip');
const saveProg = get('save-prog');

const lpEl     = get('lp');
const splitter = get('spl');
const bc       = get('bc');

const doneTitle = get('done-title');
const doneSub   = get('done-sub');
const kbdBox    = get('kbd-box');
const kbdToggle = get('kbd-toggle');

function showScreen(el) {
  [sImport, sLoaded, sEditor, sDone].forEach(s => s.classList.remove('active'));
  el.classList.add('active');
}

const IMG_EXTS = new Set(['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']);
const fileExt  = f => { const i = f.name.lastIndexOf('.'); return i >= 0 ? f.name.slice(i).toLowerCase() : ''; };

btnPickSrc.addEventListener('click', () => fi.click());

fi.addEventListener('change', e => {
  impErr.classList.add('hidden');
  const sorted = [...e.target.files]
    .filter(f => IMG_EXTS.has(fileExt(f)))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  fi.value = '';

  if (!sorted.length) { impErr.classList.remove('hidden'); return; }

  images = sorted;
  const folderName = sorted[0].webkitRelativePath
    ? sorted[0].webkitRelativePath.split('/')[0]
    : 'Selected Folder';

  srcStatus.textContent = `${sorted.length} image${sorted.length !== 1 ? 's' : ''} (${folderName})`;
  srcStatus.classList.add('ok');
  impSrcStep.classList.add('done');
  btnStart.disabled = false;
});

btnPickOut.addEventListener('click', async () => {
  if (!('showDirectoryPicker' in window)) {
    outDirHandle = null;
    outStatus.textContent = 'Will download to browser (no picker available)';
    outStatus.classList.add('ok');
    impOutStep.classList.add('done');
    return;
  }
  try {
    const handle = await window.showDirectoryPicker({ mode: 'readwrite' });
    outDirHandle = handle;
    outStatus.textContent = handle.name;
    outStatus.classList.add('ok');
    impOutStep.classList.add('done');
  } catch (e) {
    if (e.name !== 'AbortError') toast('Could not open folder: ' + e.message, 'err');
  }
});

btnStart.addEventListener('click', () => {
  if (!images.length) return;
  imageIndex = 0;
  savedCount = 0;
  strokes    = [];

  const folderName = images[0].webkitRelativePath
    ? images[0].webkitRelativePath.split('/')[0]
    : 'Images';
  ldCount.textContent  = `${images.length} image${images.length !== 1 ? 's' : ''} loaded`;
  ldFolder.textContent = folderName;

  if (!outDirHandle) toast('No output folder - masks will download to browser', 'err', 5000);
  showScreen(sLoaded);
  setTimeout(startEditor, 1400);
});

function startEditor() {
  showScreen(sEditor);
  requestAnimationFrame(() => {
    lpEl.style.width = Math.floor(get('ed-body').offsetWidth * 0.50) + 'px';
    rawCtx    = rawC.getContext('2d');
    strokeCtx = strokeC.getContext('2d');
    maskCtx   = maskC.getContext('2d');
    loadImage();
  });
}

function updateHeader() {
  hdrProg.textContent  = `${imageIndex + 1} / ${images.length}`;
  saveProg.textContent = `${savedCount} saved`;
}

function loadImage() {
  if (mlFetchCtrl) { mlFetchCtrl.abort(); mlFetchCtrl = null; }

  strokes = []; curStroke = null; isDrawing = false; mlBinary = null;
  updateUndoButton();

  const file = images[imageIndex];
  currentFileObj      = file;
  hdrFile.textContent = file.name;
  hdrFile.title       = file.name;
  updateHeader();
  setMLChip('-', '');

  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = URL.createObjectURL(file);

  const img = new Image();
  img.onload = () => {
    currentImage = img;
    imgW = img.naturalWidth;
    imgH = img.naturalHeight;
    resizeAll();
    paintRawImage();
    clearStrokeCanvas();
    renderMaskFull();
    setTimeout(() => fetchML(file), 80);
  };
  img.onerror = () => { toast(`Could not load ${file.name}`, 'err'); advance(); };
  img.src = objectURL;
}

function computeImageRect(canvasW, canvasH) {
  if (!currentImage) return { x: 0, y: 0, w: canvasW, h: canvasH };
  const imgAspect    = imgW / imgH;
  const canvasAspect = canvasW / canvasH;
  let dw, dh, dx, dy;
  if (imgAspect > canvasAspect) {
    dw = canvasW; dh = canvasW / imgAspect;
    dx = 0;       dy = (canvasH - dh) / 2;
  } else {
    dh = canvasH; dw = canvasH * imgAspect;
    dy = 0;       dx = (canvasW - dw) / 2;
  }
  return { x: dx, y: dy, w: dw, h: dh };
}

function syncCanvasSize(canvas, wrap) {
  const r = wrap.getBoundingClientRect();
  const w = Math.floor(r.width);
  const h = Math.floor(r.height);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return { w, h };
}

function resizeAll() {
  if (!currentImage) return;
  const { w: rw, h: rh } = syncCanvasSize(rawC, rawCw);
  strokeC.width = rawC.width; strokeC.height = rawC.height;
  rawIR = computeImageRect(rw, rh);
  const { w: mw, h: mh } = syncCanvasSize(maskC, maskCw);
  maskIR = computeImageRect(mw, mh);
}

function paintRawImage() {
  if (!rawCtx || !currentImage) return;
  rawCtx.clearRect(0, 0, rawC.width, rawC.height);
  rawCtx.drawImage(currentImage, rawIR.x, rawIR.y, rawIR.w, rawIR.h);
}

function clearStrokeCanvas() {
  if (strokeCtx) strokeCtx.clearRect(0, 0, strokeC.width, strokeC.height);
}

function redrawStrokeCanvas() {
  clearStrokeCanvas();
  for (const s of strokes) {
    if (s.mode === 'draw' && s.src !== 'raw') continue;
    drawStrokeOnCanvas(strokeCtx, s, rawIR);
  }
}

const resizeObserver = new ResizeObserver(() => {
  if (!currentImage) return;
  resizeAll();
  paintRawImage();
  redrawStrokeCanvas();
  renderMaskFull();
});
resizeObserver.observe(rawCw);
resizeObserver.observe(maskCw);

function renderMaskFull() {
  if (!maskCtx) return;
  const W = maskC.width, H = maskC.height;

  maskCtx.clearRect(0, 0, W, H);
  if (!currentImage) return;

  maskCtx.fillStyle = '#000';
  maskCtx.fillRect(maskIR.x, maskIR.y, maskIR.w, maskIR.h);

  if (mlBinary) {
    const imageData = maskCtx.createImageData(W, H);
    const scX = maskIR.w / imgW;
    const scY = maskIR.h / imgH;
    for (let iy = 0; iy < imgH; iy++) {
      const cy = Math.round(iy * scY + maskIR.y);
      if (cy < 0 || cy >= H) continue;
      for (let ix = 0; ix < imgW; ix++) {
        if (!mlBinary[iy * imgW + ix]) continue;
        const cx = Math.round(ix * scX + maskIR.x);
        if (cx < 0 || cx >= W) continue;
        const p = (cy * W + cx) * 4;
        imageData.data[p] = imageData.data[p + 1] = imageData.data[p + 2] = 255;
        imageData.data[p + 3] = 255;
      }
    }
    maskCtx.putImageData(imageData, 0, 0);
  }

  for (const s of strokes) drawStrokeOnCanvas(maskCtx, s, maskIR);
}

function drawStrokeOnCanvas(ctx, stroke, ir) {
  if (!stroke.pts.length) return;

  const scale = ir.w / imgW;
  const ox    = ir.x, oy = ir.y;
  const isRaw = (ir === rawIR);

  ctx.save();
  ctx.lineCap   = 'round';
  ctx.lineJoin  = 'round';
  ctx.lineWidth = stroke.sz * scale;

  if (stroke.mode === 'erase') {
    if (isRaw) {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.strokeStyle = ctx.fillStyle = 'rgba(0,0,0,1)';
    } else {
      ctx.strokeStyle = ctx.fillStyle = '#000';
    }
  } else {
    ctx.strokeStyle = ctx.fillStyle = isRaw ? 'rgba(0,221,184,0.78)' : '#fff';
  }

  const pts = stroke.pts;
  if (pts.length === 1) {
    ctx.beginPath();
    ctx.arc(pts[0].x * scale + ox, pts[0].y * scale + oy, stroke.sz * scale / 2, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.moveTo(pts[0].x * scale + ox, pts[0].y * scale + oy);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i].x * scale + ox, pts[i].y * scale + oy);
    }
    ctx.stroke();
  }

  ctx.restore();
}

function canvasToImage(cx, cy, ir) {
  return { x: (cx - ir.x) * (imgW / ir.w), y: (cy - ir.y) * (imgH / ir.h) };
}

function pointInRect(cx, cy, ir) {
  return cx >= ir.x && cx <= ir.x + ir.w && cy >= ir.y && cy <= ir.y + ir.h;
}

function clampToRect(cx, cy, ir) {
  return {
    cx: Math.max(ir.x, Math.min(ir.x + ir.w, cx)),
    cy: Math.max(ir.y, Math.min(ir.y + ir.h, cy)),
  };
}

function brushInImagePx(ir) { return brushSize * (imgW / ir.w); }

function getEventPos(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const src  = e.touches ? e.touches[0] : e;
  return { cx: src.clientX - rect.left, cy: src.clientY - rect.top, clientX: src.clientX, clientY: src.clientY };
}

strokeC.addEventListener('mousedown',  e => beginStroke(e, strokeC, rawIR));
maskC.addEventListener('mousedown',    e => beginStroke(e, maskC,   maskIR));
strokeC.addEventListener('touchstart', e => { e.preventDefault(); beginStroke(e, strokeC, rawIR); }, { passive: false });
maskC.addEventListener('touchstart',   e => { e.preventDefault(); beginStroke(e, maskC,   maskIR); }, { passive: false });
window.addEventListener('mousemove',   moveStroke);
window.addEventListener('touchmove',   e => { if (isDrawing) e.preventDefault(); moveStroke(e); }, { passive: false });
window.addEventListener('mouseup',     endStroke);
window.addEventListener('touchend',    endStroke);

strokeC.addEventListener('mouseenter', () => updateBrushCursor(true, 'raw'));
strokeC.addEventListener('mouseleave', () => { if (!isDrawing) hideBrushCursor(); });
maskC.addEventListener('mouseenter',   () => updateBrushCursor(true, 'mask'));
maskC.addEventListener('mouseleave',   () => { if (!isDrawing) hideBrushCursor(); });

function beginStroke(e, canvas, ir) {
  if (!currentImage) return;
  const { cx, cy, clientX, clientY } = getEventPos(e, canvas);
  if (!pointInRect(cx, cy, ir)) return;

  isDrawing = true;
  activeIR  = ir;
  bc.style.left = clientX + 'px';
  bc.style.top  = clientY + 'px';

  const pt = canvasToImage(cx, cy, ir);
  curStroke = {
    pts:  [pt],
    sz:   brushInImagePx(ir),
    mode: drawMode,
    src:  ir === rawIR ? 'raw' : 'mask',
  };

  drawDot(maskCtx, pt, maskIR, curStroke.sz);

  if (ir === rawIR || drawMode === 'erase') {
    drawDot(strokeCtx, pt, rawIR, curStroke.sz);
  }
}

function drawDot(ctx, pt, ir, sz) {
  const scale = ir.w / imgW;
  const isRaw = (ir === rawIR);
  ctx.save();
  if (drawMode === 'erase' && isRaw) {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = 'rgba(0,0,0,1)';
  } else {
    ctx.fillStyle = drawMode === 'erase' ? '#000' : (isRaw ? 'rgba(0,221,184,0.78)' : '#fff');
  }
  ctx.beginPath();
  ctx.arc(pt.x * scale + ir.x, pt.y * scale + ir.y, sz * scale / 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function moveStroke(e) {
  if (!currentImage) return;
  const src = e.touches ? e.touches[0] : e;
  bc.style.left = src.clientX + 'px';
  bc.style.top  = src.clientY + 'px';
  if (!isDrawing || !activeIR) return;

  const canvas = activeIR === rawIR ? strokeC : maskC;
  const { cx, cy } = getEventPos(e, canvas);
  const { cx: ccx, cy: ccy } = clampToRect(cx, cy, activeIR);
  const pt   = canvasToImage(ccx, ccy, activeIR);
  const prev = curStroke.pts[curStroke.pts.length - 1];
  curStroke.pts.push(pt);

  drawSegment(maskCtx, prev, pt, maskIR, curStroke.sz);

  if (activeIR === rawIR || drawMode === 'erase') {
    drawSegment(strokeCtx, prev, pt, rawIR, curStroke.sz);
  }
}

function drawSegment(ctx, from, to, ir, sz) {
  const scale = ir.w / imgW;
  const isRaw = (ir === rawIR);
  ctx.save();
  ctx.lineCap   = 'round';
  ctx.lineJoin  = 'round';
  ctx.lineWidth = sz * scale;
  if (drawMode === 'erase' && isRaw) {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.strokeStyle = 'rgba(0,0,0,1)';
  } else {
    ctx.strokeStyle = drawMode === 'erase' ? '#000' : (isRaw ? 'rgba(0,221,184,0.78)' : '#fff');
  }
  ctx.beginPath();
  ctx.moveTo(from.x * scale + ir.x, from.y * scale + ir.y);
  ctx.lineTo(to.x   * scale + ir.x, to.y   * scale + ir.y);
  ctx.stroke();
  ctx.restore();
}

function endStroke() {
  if (!isDrawing) return;
  isDrawing = false;
  activeIR  = null;
  if (curStroke?.pts.length) strokes.push(curStroke);
  curStroke = null;
  updateUndoButton();
}

function updateBrushCursor(show, side) {
  bc.style.display = show ? 'block' : 'none';
  if (!show) return;
  const isDraw = drawMode === 'draw';
  if (side === 'raw') {
    bc.style.border     = isDraw ? '1.5px solid rgba(0,221,184,0.8)' : '1.5px solid rgba(224,88,88,0.8)';
    bc.style.background = isDraw ? 'rgba(0,221,184,0.05)' : 'rgba(224,88,88,0.05)';
  } else {
    bc.style.border     = isDraw ? '1.5px solid rgba(255,255,255,0.7)' : '1.5px solid rgba(224,88,88,0.8)';
    bc.style.background = 'rgba(255,255,255,0.04)';
  }
}

function hideBrushCursor() { bc.style.display = 'none'; }

function setMode(mode) {
  drawMode = mode;
  btnDraw.classList.toggle('active-draw',   mode === 'draw');
  btnErase.classList.toggle('active-erase', mode === 'erase');
}
btnDraw.addEventListener('click',  () => setMode('draw'));
btnErase.addEventListener('click', () => setMode('erase'));
setMode('draw');

function setBrush(v) {
  brushSize      = Math.max(2, Math.min(60, v));
  brushSl.value  = brushSize;
  brushNum.value = brushSize;
  bc.style.width  = brushSize + 'px';
  bc.style.height = brushSize + 'px';
}

brushSl.addEventListener('input',   () => setBrush(+brushSl.value));
brushNum.addEventListener('input',  () => setBrush(+brushNum.value));
brushNum.addEventListener('change', () => setBrush(+brushNum.value));
setBrush(10);

function updateUndoButton() { btnUndo.disabled = strokes.length === 0; }

btnUndo.addEventListener('click', () => {
  if (!strokes.length) return;
  strokes.pop();
  updateUndoButton();
  redrawStrokeCanvas();
  renderMaskFull();
});

btnClr.addEventListener('click', () => {
  if (!strokes.length) { toast('No edits to reset'); return; }
  if (!confirm('Remove all your edits and return to the AI prediction?')) return;
  strokes = [];
  updateUndoButton();
  redrawStrokeCanvas();
  renderMaskFull();
});

btnRmAI.addEventListener('click', () => {
  mlBinary = null;
  setMLChip('AI cleared', '');
  renderMaskFull();
  toast('AI detections removed - draw manually on either side', 'ok', 4000);
});

function setMLChip(text, state) {
  mlChip.textContent = text;
  mlChip.className   = 'ml-chip' + (state ? ' ' + state : '');
}

mlSlider.addEventListener('input', () => {
  mlSensitivity = +mlSlider.value;
  mlSlVal.textContent = mlSensitivity + '%';
  if (currentFileObj) fetchML(currentFileObj);
});

async function fetchML(file) {
  if (mlFetchCtrl) mlFetchCtrl.abort();
  mlFetchCtrl = new AbortController();
  setMLChip('Analyzing…', 'running');

  try {
    const form = new FormData();
    form.append('image', file);
    form.append('sensitivity', (mlSensitivity / 100).toFixed(2));

    const res = await fetch('/analyze', { method: 'POST', body: form, signal: mlFetchCtrl.signal });
    if (!res.ok) throw new Error('HTTP ' + res.status);

    const blob = await res.blob();
    const mask = await decodeMaskBlob(blob, imgW, imgH);

    mlFetchCtrl = null;
    mlBinary    = mask;
    setMLChip('AI Ready', 'ready');
    renderMaskFull();

  } catch (e) {
    mlFetchCtrl = null;
    if (e.name === 'AbortError') return;
    mlBinary = null;
    setMLChip('No server', 'err');
    toast('server.py not running - draw manually', 'err', 5000);
  }
}

async function decodeMaskBlob(blob, W, H) {
  const url = URL.createObjectURL(blob);
  const img = new Image();
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url; });
  URL.revokeObjectURL(url);

  const oc  = new OffscreenCanvas(W, H);
  const ctx = oc.getContext('2d');
  ctx.drawImage(img, 0, 0, W, H);

  const data = ctx.getImageData(0, 0, W, H).data;
  const mask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) mask[i] = data[i * 4] > 127 ? 1 : 0;
  return mask;
}

function buildFinalMask() {
  const canvas  = document.createElement('canvas');
  canvas.width  = imgW;
  canvas.height = imgH;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, imgW, imgH);

  if (mlBinary) {
    const id = ctx.createImageData(imgW, imgH);
    for (let i = 0; i < imgW * imgH; i++) {
      const p = i * 4;
      id.data[p + 3] = 255;
      if (mlBinary[i]) id.data[p] = id.data[p + 1] = id.data[p + 2] = 255;
    }
    ctx.putImageData(id, 0, 0);
  }

  for (const s of strokes) {
    ctx.save();
    ctx.lineCap   = 'round';
    ctx.lineJoin  = 'round';
    ctx.lineWidth = s.sz;
    ctx.strokeStyle = ctx.fillStyle = s.mode === 'erase' ? '#000' : '#fff';

    const pts = s.pts;
    if (pts.length === 1) {
      ctx.beginPath();
      ctx.arc(pts[0].x, pts[0].y, s.sz / 2, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.stroke();
    }
    ctx.restore();
  }

  return canvas;
}

async function saveMask() {
  const canvas   = buildFinalMask();
  const file     = images[imageIndex];
  const baseName = file.name.slice(0, file.name.length - fileExt(file).length);
  const filename = baseName + '_mask.png';

  if (outDirHandle) {
    try {
      const blob = await new Promise(res => canvas.toBlob(res, 'image/png'));
      const fh   = await outDirHandle.getFileHandle(filename, { create: true });
      const wr   = await fh.createWritable();
      await wr.write(blob);
      await wr.close();
      return true;
    } catch (err) {
      if (err.name === 'AbortError') return false;
      toast('Save error: ' + err.message, 'err');
    }
  }

  await new Promise(resolve => {
    canvas.toBlob(blob => {
      const url = URL.createObjectURL(blob);
      const a   = Object.assign(document.createElement('a'), { href: url, download: filename });
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => { URL.revokeObjectURL(url); resolve(); }, 400);
    }, 'image/png');
  });
  return true;
}

btnSave.addEventListener('click', doSave);
btnSkip.addEventListener('click', advance);

async function doSave() {
  btnSave.disabled = btnSkip.disabled = true;
  const originalHTML  = btnSave.innerHTML;
  btnSave.textContent = 'Saving…';

  const ok = await saveMask();

  btnSave.innerHTML = originalHTML;
  btnSave.disabled  = btnSkip.disabled = false;

  if (!ok) return;
  savedCount++;
  const fileForTrain = currentFileObj;
  const maskForTrain = buildFinalMask();
  submitFinetune(fileForTrain, maskForTrain);
  advance();
}

function advance() {
  imageIndex++;
  if (imageIndex >= images.length) showDone();
  else { updateHeader(); loadImage(); }
}

function showDone() {
  doneTitle.textContent = 'All done!';
  doneSub.textContent   = `${savedCount} mask${savedCount !== 1 ? 's' : ''} saved`;
  showScreen(sDone);
}

get('btn-restart').addEventListener('click', () => {
  images = []; imageIndex = 0; savedCount = 0; strokes = [];
  mlBinary = null; outDirHandle = null;
  srcStatus.textContent = 'No folder selected'; srcStatus.classList.remove('ok');
  outStatus.textContent = 'No folder selected'; outStatus.classList.remove('ok');
  impSrcStep.classList.remove('done'); impOutStep.classList.remove('done');
  btnStart.disabled = true; impErr.classList.add('hidden');
  if (objectURL) { URL.revokeObjectURL(objectURL); objectURL = null; }
  showScreen(sImport);
});

let splDragging = false, splStartX = 0, splStartW = 0;

splitter.addEventListener('mousedown', e => {
  splDragging = true;
  splStartX   = e.clientX;
  splStartW   = lpEl.offsetWidth;
  splitter.classList.add('drag');
  document.body.style.cursor     = 'col-resize';
  document.body.style.userSelect = 'none';
  e.preventDefault();
});

window.addEventListener('mousemove', e => {
  if (!splDragging) return;
  const totalW = get('ed-body').offsetWidth;
  const newW   = Math.max(180, Math.min(totalW - 180 - 5, splStartW + (e.clientX - splStartX)));
  lpEl.style.width = newW + 'px';
  if (currentImage) { resizeAll(); paintRawImage(); redrawStrokeCanvas(); renderMaskFull(); }
});

window.addEventListener('mouseup', () => {
  if (!splDragging) return;
  splDragging = false;
  splitter.classList.remove('drag');
  document.body.style.cursor = document.body.style.userSelect = '';
});

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const k = e.key.toLowerCase();

  if (k === 'd' && !e.ctrlKey) { setMode('draw');  e.preventDefault(); return; }
  if (k === 'e' && !e.ctrlKey) { setMode('erase'); e.preventDefault(); return; }

  if (k === 'z' && !e.shiftKey && !e.ctrlKey) { btnUndo.click(); e.preventDefault(); return; }
  if (k === 'z' && e.shiftKey)                { btnClr.click();  e.preventDefault(); return; }

  if (k === 'enter' || k === ' ') { doSave();  e.preventDefault(); return; }
  if (k === 's' && !e.ctrlKey)   { advance(); e.preventDefault(); return; }

  if (k === '[') { setBrush(brushSize - 3); e.preventDefault(); }
  if (k === ']') { setBrush(brushSize + 3); e.preventDefault(); }
});

kbdToggle.addEventListener('click', () => kbdBox.classList.toggle('visible'));

async function submitFinetune(file, maskCanvas) {
  try {
    const maskBlob = await new Promise(res => maskCanvas.toBlob(res, 'image/png'));
    const form = new FormData();
    form.append('image', file);
    form.append('mask', maskBlob, 'mask.png');
    form.append('steps', '5');

    const res = await fetch('/finetune', { method: 'POST', body: form });
    if (!res.ok) throw new Error('HTTP ' + res.status);

    const data = await res.json();
    if (data.skipped) {
      toast('Saved (classical mode)', 'ok', 3000);
    } else {
      toast('Fine-tuned - loss: ' + data.loss.toFixed(4), 'ok', 5000);
    }
  } catch (e) {
    toast('Fine-tune error: ' + e.message, 'err', 4000);
  }
}

function toast(message, type = '', duration = 3500) {
  const el = document.createElement('div');
  el.className   = 'toast' + (type ? ' ' + type : '');
  el.textContent = message;
  get('toasts').appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity 300ms';
    el.style.opacity    = '0';
    setTimeout(() => el.remove(), 320);
  }, duration);
}
