"""FastAPI app for the AlexNet Open Day demo."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from src.captions import CAPTIONS
from src.demo_images import DEMO_IMAGES_DIR, discover_images, format_image_label
from src.model import ModelUnavailableError, run_alexnet_analysis

APP_TITLE = "How Does a Neural Network See?"

app = FastAPI(
    title="AlexNet Vision Demo",
    description="Local-first Open Day demo showing AlexNet layer responses and predictions.",
    version="0.2.0",
)


class RunRequest(BaseModel):
    """Request body for running AlexNet on a curated image."""

    image_name: str = Field(..., min_length=1)
    fallback: bool = False


class CameraRunRequest(BaseModel):
    """Request body for running AlexNet on a locally captured camera frame."""

    image_data: str = Field(..., min_length=1)
    fallback: bool = False
    include_visualisations: bool = True
    visualisation_keys: list[str] | None = None


def _image_lookup() -> dict[str, Path]:
    """Return curated images keyed by filename."""
    return {path.name: path for path in discover_images(DEMO_IMAGES_DIR)}


def _find_demo_image(image_name: str) -> Path:
    """Return a validated curated image path or raise a 404."""
    image_path = _image_lookup().get(Path(image_name).name)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Curated image not found.")
    return image_path


def _decode_camera_image(image_data: str) -> Image.Image:
    """Decode a browser camera data URL without saving it to disk."""
    if "," in image_data:
        header, encoded = image_data.split(",", 1)
        if not header.startswith("data:image/"):
            raise ValueError("Camera frame must be an image data URL.")
    else:
        encoded = image_data

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Camera frame was not valid base64 image data.") from exc

    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Camera frame is too large for this local demo.")

    try:
        return Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Camera frame could not be opened as an image.") from exc


def _run_live_analysis_response(
    image: Image.Image,
    *,
    source: str,
    include_visualisations: bool = True,
    visualisation_keys: set[str] | None = None,
) -> JSONResponse:
    """Run AlexNet analysis and return a consistent JSON response."""
    try:
        analysis = run_alexnet_analysis(
            image,
            include_visualisations=include_visualisations,
            visualisation_keys=visualisation_keys,
        )
    except ModelUnavailableError as exc:
        return JSONResponse(
            {
                "ok": False,
                "mode": "live",
                "source": source,
                "message": str(exc),
                "help": "Run the setup script and pre-download AlexNet weights, or use fallback replay once assets have been precomputed.",
                "predictions": [],
                "visualisations": [],
                "visualisations_included": include_visualisations,
            }
        )

    return JSONResponse(
        {
            "ok": True,
            "mode": "live",
            "source": source,
            "message": "AlexNet returned likely ImageNet labels. These can be wrong.",
            "predictions": [prediction.to_dict() for prediction in analysis.predictions],
            "visualisations": [visualisation.to_dict() for visualisation in analysis.visualisations],
            "visualisations_included": include_visualisations,
        }
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-screen demo UI."""
    return _render_index_html()


@app.get("/api/images")
def list_images() -> dict[str, Any]:
    """List curated demo images for the selector."""
    images = [
        {
            "name": path.name,
            "label": format_image_label(path),
            "url": f"/api/images/{path.name}",
        }
        for path in discover_images(DEMO_IMAGES_DIR)
    ]
    return {
        "images": images,
        "empty_message": (
            "No curated images were found. Add .jpg, .jpeg, .png, or .webp files to "
            f"{DEMO_IMAGES_DIR} and refresh the page."
        ),
    }


@app.get("/api/images/{image_name}")
def get_image(image_name: str) -> FileResponse:
    """Serve a validated curated image file."""
    image_path = _find_demo_image(image_name)
    return FileResponse(image_path)


@app.get("/api/captions")
def get_captions() -> dict[str, str]:
    """Return editable public captions for the selectable layer diagram."""
    return CAPTIONS


@app.post("/api/run")
def run_demo(request: RunRequest) -> JSONResponse:
    """Run AlexNet top-5 inference for a curated image.

    Fallback replay is kept as a visible mode, but full fallback asset playback
    will be implemented in the next phase. Live inference failures are returned
    as safe UI messages instead of crashing the app.
    """
    image_path = _find_demo_image(request.image_name)

    if request.fallback:
        return JSONResponse(
            {
                "ok": False,
                "mode": "fallback",
                "source": "curated",
                "message": "Fallback replay assets will be wired in Phase 5. Turn replay mode off for live AlexNet inference.",
                "predictions": [],
                "visualisations": [],
            }
        )

    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return JSONResponse(
            {
                "ok": False,
                "mode": "live",
                "source": "curated",
                "message": f"Could not open the curated image: {exc}",
                "predictions": [],
                "visualisations": [],
            },
            status_code=400,
        )

    return _run_live_analysis_response(image, source="curated")


@app.post("/api/run-camera")
def run_camera_demo(request: CameraRunRequest) -> JSONResponse:
    """Run AlexNet on a browser-captured camera frame without saving it."""
    if request.fallback:
        return JSONResponse(
            {
                "ok": False,
                "mode": "fallback",
                "source": "camera",
                "message": "Fallback replay uses curated precomputed assets, not live camera frames. Turn replay mode off for camera inference.",
                "predictions": [],
                "visualisations": [],
            }
        )

    try:
        image = _decode_camera_image(request.image_data)
    except ValueError as exc:
        return JSONResponse(
            {
                "ok": False,
                "mode": "live",
                "source": "camera",
                "message": str(exc),
                "predictions": [],
                "visualisations": [],
            },
            status_code=400,
        )

    return _run_live_analysis_response(
        image,
        source="camera",
        include_visualisations=request.include_visualisations,
        visualisation_keys=set(request.visualisation_keys or ()) or None,
    )


def _render_index_html() -> str:
    """Return dependency-free HTML, CSS, and JavaScript for the booth UI."""
    return f"""
<!doctype html>
<html lang="en-AU">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{APP_TITLE}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #080814;
      --bg-2: #10142a;
      --panel: rgba(15, 23, 42, 0.82);
      --panel-2: #172033;
      --panel-3: #0d1324;
      --text: #f8fbff;
      --muted: #b9c4d8;
      --accent: #22d3ee;
      --accent-2: #8b5cf6;
      --accent-3: #f59e0b;
      --danger: #fecaca;
      --ok: #bbf7d0;
      --border: rgba(148, 163, 184, 0.22);
      --glow: rgba(34, 211, 238, 0.22);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at 12% 6%, rgba(34,211,238,0.22) 0, transparent 28%), radial-gradient(circle at 86% 12%, rgba(139,92,246,0.24) 0, transparent 30%), linear-gradient(135deg, var(--bg) 0%, var(--bg-2) 56%, #070711 100%); color: var(--text); }}
    body::before {{ content: ""; position: fixed; inset: 0; pointer-events: none; background-image: linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px); background-size: 42px 42px; mask-image: radial-gradient(circle at top, black, transparent 72%); }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 28px 0 44px; position: relative; }}
    header {{ margin-bottom: 22px; }}
    h1 {{ font-size: clamp(2rem, 4vw, 4.5rem); line-height: 1; margin: 0 0 14px; letter-spacing: -0.05em; background: linear-gradient(90deg, #f8fbff, #67e8f9 45%, #c4b5fd 76%, #fbbf24); -webkit-background-clip: text; background-clip: text; color: transparent; }}
    h2 {{ margin: 0 0 14px; font-size: 1.25rem; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .grid {{ display: grid; grid-template-columns: 360px 1fr; gap: 20px; align-items: start; }}
    .card {{ background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(10,15,30,0.88)); border: 1px solid var(--border); border-radius: 24px; padding: 20px; box-shadow: 0 24px 70px rgba(0,0,0,0.38), 0 0 38px rgba(34,211,238,0.07); backdrop-filter: blur(14px); }}
    .stack {{ display: grid; gap: 16px; }}
    label {{ display: block; margin-bottom: 8px; font-weight: 700; }}
    select, button {{ width: 100%; border-radius: 14px; border: 1px solid var(--border); padding: 12px 14px; font: inherit; }}
    select {{ background: linear-gradient(180deg, #1b2540, #121a2f); color: var(--text); }}
    button {{ background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #03111d; font-weight: 900; cursor: pointer; box-shadow: 0 12px 30px rgba(34,211,238,0.22); transition: transform 120ms ease, filter 120ms ease, box-shadow 120ms ease; }}
    button:hover:not(:disabled) {{ transform: translateY(-1px); filter: brightness(1.08); box-shadow: 0 16px 38px rgba(139,92,246,0.26); }}
    button.secondary {{ background: linear-gradient(180deg, #27324d, #1a2338); color: var(--text); box-shadow: none; }}
    button:disabled {{ opacity: 0.5; cursor: not-allowed; box-shadow: none; }}
    .toggle {{ display: flex; gap: 10px; align-items: center; color: var(--muted); }}
    .toggle input {{ width: auto; transform: scale(1.2); accent-color: var(--accent); }}
    .camera-actions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .camera-actions .wide {{ grid-column: 1 / -1; }}
    .camera-note {{ font-size: 0.88rem; margin: 8px 0 0; color: var(--muted); }}
    .message {{ padding: 12px 14px; border-radius: 14px; background: linear-gradient(180deg, rgba(30,41,59,0.98), rgba(17,24,39,0.98)); color: var(--muted); border: 1px solid rgba(148,163,184,0.16); }}
    .message.error {{ background: linear-gradient(180deg, rgba(127,29,29,0.46), rgba(69,10,10,0.34)); color: var(--danger); border-color: rgba(248,113,113,0.3); }}
    .message.ok {{ background: linear-gradient(180deg, rgba(20,83,45,0.45), rgba(6,78,59,0.34)); color: var(--ok); border-color: rgba(74,222,128,0.28); }}
    .bar {{ grid-column: 1 / -1; height: 9px; background: #1f2941; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: linear-gradient(90deg, #22d3ee, #a78bfa, #f59e0b); box-shadow: 0 0 18px rgba(34,211,238,0.55); }}
    .muted {{ color: var(--muted); }}
    .network {{ margin-bottom: 16px; padding: 16px; border-radius: 18px; background: radial-gradient(circle at 20% 45%, rgba(34,211,238,0.18), transparent 32%), radial-gradient(circle at 78% 50%, rgba(245,158,11,0.13), transparent 35%), #030611; border: 1px solid var(--border); overflow: hidden; }}
    .network svg {{ width: 100%; height: auto; display: block; filter: drop-shadow(0 0 16px rgba(34,211,238,0.12)); }}
    .network .layer {{ fill: rgba(34, 211, 238, 0.13); stroke: rgba(226, 232, 240, 0.78); stroke-width: 2; }}
    .network .layer.active, .network .layer:focus {{ fill: rgba(245, 158, 11, 0.38); stroke: #fbbf24; outline: none; }}
    .network .connector {{ stroke: rgba(148, 163, 184, 0.5); stroke-width: 3; stroke-linecap: round; }}
    .network text {{ fill: #f8fbff; font-size: 13px; font-weight: 900; text-anchor: middle; dominant-baseline: middle; pointer-events: none; }}
    .network .stage-note {{ fill: #b9c4d8; font-size: 11px; font-weight: 700; }}
    .layer-detail {{ margin: 0 0 16px; display: grid; grid-template-columns: minmax(260px, 1.4fr) minmax(220px, 0.8fr); gap: 16px; align-items: start; background: linear-gradient(180deg, rgba(8,16,36,0.98), rgba(4,7,16,0.98)); border: 1px solid rgba(34,211,238,0.2); border-radius: 18px; padding: 16px; }}
    .layer-detail.placeholder {{ display: block; }}
    .layer-detail img, .layer-detail video {{ width: 100%; display: block; border-radius: 14px; background: #020208; border: 1px solid rgba(255,255,255,0.09); box-shadow: 0 24px 60px rgba(0,0,0,0.32); }}
    .layer-detail video {{ transform: scaleX(-1); }}
    .camera-capture-source {{ position: fixed; width: 1px; height: 1px; left: -10px; top: -10px; opacity: 0; pointer-events: none; }}
    .detail-copy h3 {{ margin: 0 0 8px; font-size: 1.18rem; color: #fef3c7; }}
    .detail-copy p {{ margin: 0 0 10px; }}
    .detail-pill {{ display: inline-flex; align-items: center; margin: 4px 6px 4px 0; padding: 6px 9px; border-radius: 999px; background: rgba(34,211,238,0.12); border: 1px solid rgba(34,211,238,0.22); color: #cffafe; font-size: 0.82rem; font-weight: 800; }}
    .prediction-detail {{ list-style: none; padding: 0; margin: 8px 0 0; display: grid; gap: 8px; }}
    .prediction-detail li {{ display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; padding: 10px; border-radius: 12px; background: rgba(15,23,42,0.74); border: 1px solid rgba(148,163,184,0.14); }}
    @media (max-width: 980px) {{ .layer-detail {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>{APP_TITLE}</h1>
    <p>This local demo shows how a trained vision model responds at different layers. Early layers often respond to simple visual patterns, while deeper layers combine those patterns into features useful for classification. The final prediction is a likely label, not guaranteed truth.</p>
  </header>

  <section class="grid">
    <aside class="card stack">
      <div>
        <h2>Curated image</h2>
        <label for="imageSelect">Select a booth image</label>
        <select id="imageSelect"><option value="">Loading images…</option></select>
      </div>

      <div>
        <h2>Live camera</h2>
        <div class="camera-actions">
          <button id="startCameraButton" class="secondary" type="button">Start camera</button>
          <button id="cameraRunButton" type="button" disabled>Capture + run</button>
          <button id="liveRunButton" class="wide" type="button" disabled>Start continuous AlexNet</button>
        </div>
        <p class="camera-note">Opt-in local camera mode. Frames are sent only to this local app for analysis and are not saved.</p>
      </div>

      <label class="toggle"><input id="fallbackToggle" type="checkbox" /> Fallback / replay mode</label>

      <button id="runButton" disabled>Run AlexNet</button>
      <button id="resetButton" class="secondary">Reset demo</button>
    </aside>

    <section class="stack">
      <div class="card">
        <h2>AlexNet layer explorer</h2>
        <div id="status" class="message">Run AlexNet, then choose any layer. Predictions are shown by selecting the Prediction stage.</div>
        <p id="caption" class="message">Choose any layer in the diagram, including the input.</p>
        <div class="network" aria-label="Selectable AlexNet layer diagram">
          <svg viewBox="0 0 1100 220" role="img">
            <title>Selectable AlexNet path from input through all demo layers</title>
            <line class="connector" x1="80" y1="110" x2="1050" y2="110" />
            <rect class="layer" data-layer="Input" tabindex="0" x="20" y="48" width="90" height="124" rx="16" />
            <rect class="layer" data-layer="Conv 1" tabindex="0" x="124" y="62" width="76" height="96" rx="14" />
            <rect class="layer" data-layer="Pool 1" tabindex="0" x="214" y="70" width="72" height="80" rx="14" />
            <rect class="layer" data-layer="Conv 2" tabindex="0" x="300" y="62" width="76" height="96" rx="14" />
            <rect class="layer" data-layer="Pool 2" tabindex="0" x="390" y="70" width="72" height="80" rx="14" />
            <rect class="layer" data-layer="Conv 3" tabindex="0" x="476" y="62" width="76" height="96" rx="14" />
            <rect class="layer" data-layer="Conv 4" tabindex="0" x="566" y="62" width="76" height="96" rx="14" />
            <rect class="layer" data-layer="Conv 5" tabindex="0" x="656" y="62" width="76" height="96" rx="14" />
            <rect class="layer" data-layer="Pool 5" tabindex="0" x="746" y="70" width="72" height="80" rx="14" />
            <rect class="layer" data-layer="Avg pool" tabindex="0" x="832" y="74" width="78" height="72" rx="14" />
            <rect class="layer" data-layer="Classifier" tabindex="0" x="924" y="66" width="88" height="88" rx="14" />
            <rect class="layer" data-layer="Prediction" tabindex="0" x="1026" y="68" width="54" height="84" rx="14" />
            <text x="65" y="110">Input</text>
            <text x="162" y="110">Conv 1</text>
            <text x="250" y="110">Pool 1</text>
            <text x="338" y="110">Conv 2</text>
            <text x="426" y="110">Pool 2</text>
            <text x="514" y="110">Conv 3</text>
            <text x="604" y="110">Conv 4</text>
            <text x="694" y="110">Conv 5</text>
            <text x="782" y="110">Pool 5</text>
            <text x="871" y="104">Avg</text>
            <text x="871" y="122" class="stage-note">pool</text>
            <text x="968" y="104">Classifier</text>
            <text x="968" y="122" class="stage-note">scores</text>
            <text x="1053" y="104">Pred</text>
            <text x="1053" y="122" class="stage-note">iction</text>
          </svg>
        </div>
        <div id="layerDetail" class="layer-detail placeholder">
          <p class="message">Choose a curated image or start the camera, then select any AlexNet layer in the diagram.</p>
        </div>
      </div>
    </section>
  </section>
</main>

<script>
const imageSelect = document.getElementById('imageSelect');
const runButton = document.getElementById('runButton');
const startCameraButton = document.getElementById('startCameraButton');
const cameraRunButton = document.getElementById('cameraRunButton');
const liveRunButton = document.getElementById('liveRunButton');
const resetButton = document.getElementById('resetButton');
const fallbackToggle = document.getElementById('fallbackToggle');
const statusBox = document.getElementById('status');
const layerDetail = document.getElementById('layerDetail');
const caption = document.getElementById('caption');
const DIAGRAM_LAYERS = ['Input', 'Conv 1', 'Pool 1', 'Conv 2', 'Pool 2', 'Conv 3', 'Conv 4', 'Conv 5', 'Pool 5', 'Avg pool', 'Classifier', 'Prediction'];
const DIAGRAM_LAYER_KEYS = {{
  'Conv 1': 'conv1',
  'Pool 1': 'pool1',
  'Conv 2': 'conv2',
  'Pool 2': 'pool2',
  'Conv 3': 'conv3',
  'Conv 4': 'conv4',
  'Conv 5': 'conv5',
  'Pool 5': 'pool5',
  'Avg pool': 'avgpool'
}};
let captions = {{}};
let visualisationsByLabel = new Map();
let currentStage = 'Input';
let inputState = {{kind: 'empty'}};
let lastPredictions = [];
let cameraStream = null;
let cameraVideo = null;
let liveRunActive = false;
let liveFrameIndex = 0;

function escapeHtml(value) {{
  return String(value).replace(/[&<>"']/g, character => ({{
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }}[character]));
}}

function setStatus(text, kind = '') {{
  statusBox.className = `message ${{kind}}`;
  statusBox.textContent = text;
}}

function clearResults() {{
  visualisationsByLabel = new Map();
  lastPredictions = [];
  renderSelectedLayerDetail();
  setStatus('Run AlexNet, then choose any layer. Predictions are shown by selecting the Prediction stage.');
}}

function resetDemo() {{
  stopCamera({{clearInput: true}});
  imageSelect.value = '';
  inputState = {{kind: 'empty'}};
  fallbackToggle.checked = false;
  runButton.disabled = true;
  clearResults();
  selectStage('Input');
}}

function selectStage(stage, options = {{}}) {{
  currentStage = DIAGRAM_LAYERS.includes(stage) ? stage : 'Input';
  document.querySelectorAll('.network .layer').forEach(layer => layer.classList.toggle('active', layer.dataset.layer === currentStage));
  caption.textContent = captions[currentStage] || 'This shows how a trained vision model responds at this stage.';
  if (options.render !== false) {{
    renderSelectedLayerDetail({{scroll: options.scroll === true}});
  }}
}}

function renderSelectedLayerDetail(options = {{}}) {{
  if (currentStage === 'Input') {{
    renderInputDetail(options);
  }} else if (currentStage === 'Prediction') {{
    renderPredictionDetail(options);
  }} else if (currentStage === 'Classifier') {{
    renderClassifierDetail(options);
  }} else {{
    const item = visualisationsByLabel.get(currentStage);
    if (item) {{
      renderActivationDetail(item, options);
    }} else {{
      renderLayerPlaceholder(currentStage, options);
    }}
  }}
}}

function scrollLayerDetailIfNeeded(options = {{}}) {{
  if (options.scroll === true) {{
    layerDetail.scrollIntoView({{behavior: 'smooth', block: 'nearest'}});
  }}
}}

function renderInputDetail(options = {{}}) {{
  const captionText = captions.Input || 'The image is resized and normalised before entering the network.';
  layerDetail.className = inputState.kind === 'empty' ? 'layer-detail placeholder' : 'layer-detail';
  if (cameraStream && cameraVideo) {{
    layerDetail.innerHTML = `
      <div class="input-media" aria-label="Local camera preview">
        <video autoplay playsinline muted aria-label="Live local camera input"></video>
      </div>
      <div class="detail-copy">
        <h3>Input</h3>
        <p>${{escapeHtml(captionText)}}</p>
        <p>This is the live local camera frame before AlexNet preprocessing. Frames are analysed in memory and are not saved.</p>
        <span class="detail-pill">Opt-in camera</span>
        <span class="detail-pill">Local only</span>
        <span class="detail-pill">Not saved</span>
      </div>`;
    const visibleVideo = layerDetail.querySelector('.input-media video');
    visibleVideo.srcObject = cameraStream;
    visibleVideo.play().catch(() => {{}});
  }} else if (inputState.kind === 'image') {{
    layerDetail.innerHTML = `
      <img src="${{escapeHtml(inputState.url)}}" alt="Selected curated input image" />
      <div class="detail-copy">
        <h3>Input</h3>
        <p>${{escapeHtml(captionText)}}</p>
        <p>${{escapeHtml(inputState.label || 'Selected curated image')}}</p>
        <span class="detail-pill">Curated image</span>
        <span class="detail-pill">224 × 224 model input after preprocessing</span>
      </div>`;
  }} else {{
    layerDetail.innerHTML = '<p class="message">Choose a curated image or start the camera. The selected input will appear here as the first selectable AlexNet stage.</p>';
  }}
  scrollLayerDetailIfNeeded(options);
}}

function renderActivationDetail(item, options = {{}}) {{
  const captionText = captions[item.caption_key] || item.note || 'This shows fixed channels from this layer response.';
  const tensorShape = (item.tensor_shape || []).join(' × ');
  layerDetail.className = 'layer-detail';
  layerDetail.innerHTML = `
    <img class="activation-detail-image" data-layer="${{escapeHtml(item.label)}}" src="${{item.image_data}}" alt="Large ${{escapeHtml(item.label)}} activation grid" />
    <div class="detail-copy">
      <h3>${{escapeHtml(item.label)}}</h3>
      <p>${{escapeHtml(captionText)}}</p>
      <p>${{escapeHtml(item.note || 'Each square is one fixed channel from this layer, so the tile position stays stable across frames. Cyan, yellow, and white regions indicate stronger responses after normalising that channel for display.')}}</p>
      <span class="detail-pill">${{escapeHtml(tensorShape)}}</span>
      <span class="detail-pill">Fixed channel positions</span>
      <span class="detail-pill">Cyan/yellow/white = stronger</span>
      <span class="detail-pill">Normalised for display</span>
      <span class="detail-pill">Updates with each live frame</span>
    </div>`;
  scrollLayerDetailIfNeeded(options);
}}

function renderLayerPlaceholder(stage, options = {{}}) {{
  const captionText = captions[stage] || 'This shows how a trained vision model responds at this stage.';
  layerDetail.className = 'layer-detail placeholder';
  layerDetail.innerHTML = `<p class="message"><strong>${{escapeHtml(stage)}}:</strong> ${{escapeHtml(captionText)}} Run AlexNet to show this layer from the selected input.</p>`;
  scrollLayerDetailIfNeeded(options);
}}

function renderClassifierDetail(options = {{}}) {{
  const captionText = captions.Classifier || 'The classifier turns compact feature values into scores for ImageNet training labels.';
  const avgPool = visualisationsByLabel.get('Avg pool');
  const shape = avgPool ? avgPool.tensor_shape.join(' × ') : 'Run AlexNet to show incoming feature shape';
  layerDetail.className = 'layer-detail placeholder';
  layerDetail.innerHTML = `
    <div class="detail-copy">
      <h3>Classifier</h3>
      <p>${{escapeHtml(captionText)}}</p>
      <p>AlexNet flattens the compact feature maps and uses fully connected layers to produce class scores. This demo shows those scores through the prediction list rather than claiming the model is certain.</p>
      <span class="detail-pill">Input: ${{escapeHtml(shape)}}</span>
      <span class="detail-pill">Fully connected layers</span>
      <span class="detail-pill">Scores, not certainty</span>
    </div>`;
  scrollLayerDetailIfNeeded(options);
}}

function predictionListHtml() {{
  if (!lastPredictions.length) {{
    return '<p class="message">Run AlexNet to show the top likely ImageNet labels for this input.</p>';
  }}
  const items = lastPredictions.map(item => {{
    const pct = Math.round(item.probability * 1000) / 10;
    return `<li><strong>${{escapeHtml(item.label)}}</strong><span>${{pct}}%</span><div class="bar"><span style="width: ${{pct}}%"></span></div></li>`;
  }}).join('');
  return `<ol class="prediction-detail">${{items}}</ol>`;
}}

function renderPredictionDetail(options = {{}}) {{
  const captionText = captions.Prediction || 'The final prediction is a likely class from the model’s training labels. It can be wrong.';
  layerDetail.className = lastPredictions.length ? 'layer-detail' : 'layer-detail placeholder';
  layerDetail.innerHTML = `
    <div data-role="prediction-list">
      ${{predictionListHtml()}}
    </div>
    <div class="detail-copy">
      <h3>Prediction</h3>
      <p>${{escapeHtml(captionText)}}</p>
      <p>These labels come from the model’s training categories and can be wrong, especially for unusual, ambiguous, or out-of-distribution images.</p>
      <span class="detail-pill">Top-5 labels</span>
      <span class="detail-pill">Likely, not guaranteed</span>
    </div>`;
  scrollLayerDetailIfNeeded(options);
}}

function updateLiveLayerDetail() {{
  if (currentStage === 'Input' || currentStage === 'Classifier') {{
    return;
  }}
  if (currentStage === 'Prediction') {{
    const predictionList = layerDetail.querySelector('[data-role="prediction-list"]');
    if (predictionList) {{
      predictionList.innerHTML = predictionListHtml();
    }} else {{
      renderPredictionDetail({{scroll: false}});
    }}
    return;
  }}

  const item = visualisationsByLabel.get(currentStage);
  if (!item) {{
    return;
  }}
  const image = layerDetail.querySelector('.activation-detail-image');
  if (image && image.dataset.layer === currentStage) {{
    image.src = item.image_data;
  }} else {{
    renderActivationDetail(item, {{scroll: false}});
  }}
}}

function renderAnalysisResult(data, options = {{}}) {{
  setStatus(data.message || 'Run complete.', data.ok ? 'ok' : 'error');
  if (data.help) {{
    setStatus(`${{data.message}} ${{data.help}}`, data.ok ? 'ok' : 'error');
  }}
  lastPredictions = data.predictions || [];
  if (options.live !== true) {{
    visualisationsByLabel = new Map();
  }}
  const visualisationsIncluded = data.visualisations_included !== false;
  if (visualisationsIncluded && data.visualisations) {{
    data.visualisations.forEach(item => visualisationsByLabel.set(item.label, item));
  }}
  if (options.live === true) {{
    updateLiveLayerDetail();
  }} else {{
    renderSelectedLayerDetail();
  }}
}}

function stopLiveRun() {{
  liveRunActive = false;
  liveRunButton.textContent = 'Start continuous AlexNet';
  liveRunButton.classList.remove('secondary');
}}

function stopCamera(options = {{}}) {{
  stopLiveRun();
  if (cameraStream) {{
    cameraStream.getTracks().forEach(track => track.stop());
  }}
  if (cameraVideo) {{
    cameraVideo.srcObject = null;
    cameraVideo.remove();
  }}
  cameraStream = null;
  cameraVideo = null;
  cameraRunButton.disabled = true;
  liveRunButton.disabled = true;
  startCameraButton.textContent = 'Start camera';
  if (options.clearInput === true) {{
    inputState = {{kind: 'empty'}};
  }}
  if (currentStage === 'Input') {{
    renderSelectedLayerDetail();
  }}
}}

async function startCamera() {{
  if (cameraStream) {{
    stopCamera({{clearInput: true}});
    clearResults();
    selectStage('Input');
    return;
  }}
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
    setStatus('This browser does not support local camera access.', 'error');
    return;
  }}
  clearResults();
  imageSelect.value = '';
  inputState = {{kind: 'empty'}};
  runButton.disabled = true;
  try {{
    cameraStream = await navigator.mediaDevices.getUserMedia({{video: {{width: {{ideal: 960}}, height: {{ideal: 720}}, facingMode: 'user'}}, audio: false}});
    cameraVideo = document.createElement('video');
    cameraVideo.autoplay = true;
    cameraVideo.playsInline = true;
    cameraVideo.muted = true;
    cameraVideo.className = 'camera-capture-source';
    cameraVideo.setAttribute('aria-hidden', 'true');
    cameraVideo.tabIndex = -1;
    cameraVideo.srcObject = cameraStream;
    document.body.appendChild(cameraVideo);
    inputState = {{kind: 'camera'}};
    selectStage('Input');
    await waitForCameraReady();
    cameraRunButton.disabled = false;
    liveRunButton.disabled = false;
    startCameraButton.textContent = 'Stop camera';
    setStatus('Camera preview is local. Capture one frame or start continuous AlexNet.', 'ok');
  }} catch (error) {{
    setStatus(`Camera access was not available: ${{error}}`, 'error');
    stopCamera({{clearInput: true}});
  }}
}}

function waitForCameraReady() {{
  return new Promise((resolve, reject) => {{
    if (!cameraVideo) {{
      reject(new Error('Camera preview was not created.'));
      return;
    }}
    const done = () => {{
      cleanup();
      resolve();
    }};
    const fail = () => {{
      cleanup();
      reject(new Error('Camera preview did not become ready in time.'));
    }};
    const cleanup = () => {{
      window.clearTimeout(timeout);
      cameraVideo.removeEventListener('loadedmetadata', done);
      cameraVideo.removeEventListener('canplay', done);
    }};
    const timeout = window.setTimeout(fail, 4000);
    cameraVideo.addEventListener('loadedmetadata', done, {{once: true}});
    cameraVideo.addEventListener('canplay', done, {{once: true}});
    cameraVideo.play().catch(() => {{}});
    if (cameraVideo.videoWidth && cameraVideo.videoHeight) {{
      done();
    }}
  }});
}}

function captureCameraFrame() {{
  if (!cameraVideo || !cameraVideo.videoWidth || !cameraVideo.videoHeight) {{
    throw new Error('Camera preview is not ready yet.');
  }}
  const canvas = document.createElement('canvas');
  const maxSide = 900;
  const scale = Math.min(1, maxSide / Math.max(cameraVideo.videoWidth, cameraVideo.videoHeight));
  canvas.width = Math.round(cameraVideo.videoWidth * scale);
  canvas.height = Math.round(cameraVideo.videoHeight * scale);
  const ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.9);
}}

async function analyseCameraFrame({{includeVisualisations = true, live = false, visualisationKeys = []}} = {{}}) {{
  const imageData = captureCameraFrame();
  const response = await fetch('/api/run-camera', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{
      image_data: imageData,
      fallback: fallbackToggle.checked,
      include_visualisations: includeVisualisations,
      visualisation_keys: visualisationKeys
    }})
  }});
  const data = await response.json();
  renderAnalysisResult(data, {{live}});
  if (live && data.ok) {{
    if (liveFrameIndex === 1 || liveFrameIndex % 5 === 0) {{
      setStatus(`Continuous AlexNet is running locally. Analysed frame ${{liveFrameIndex}}.`, 'ok');
    }}
  }}
  return data;
}}

async function runCameraFrame() {{
  cameraRunButton.disabled = true;
  setStatus('Capturing one local camera frame and selected AlexNet layers…');
  try {{
    await analyseCameraFrame({{includeVisualisations: true, live: false}});
  }} catch (error) {{
    setStatus(`The local app could not analyse the camera frame: ${{error}}`, 'error');
  }} finally {{
    cameraRunButton.disabled = !cameraStream;
  }}
}}

async function liveRunLoop() {{
  if (!liveRunActive || !cameraStream) return;
  liveFrameIndex += 1;
  const selectedVisualisationKey = DIAGRAM_LAYER_KEYS[currentStage];
  try {{
    await analyseCameraFrame({{
      includeVisualisations: Boolean(selectedVisualisationKey),
      live: true,
      visualisationKeys: selectedVisualisationKey ? [selectedVisualisationKey] : []
    }});
  }} catch (error) {{
    setStatus(`Continuous AlexNet stopped: ${{error}}`, 'error');
    stopLiveRun();
    return;
  }}
  if (liveRunActive) {{
    window.setTimeout(liveRunLoop, 150);
  }}
}}

function toggleLiveRun() {{
  if (liveRunActive) {{
    stopLiveRun();
    setStatus('Continuous AlexNet stopped. Camera preview is still local.', 'ok');
    return;
  }}
  if (!cameraStream) {{
    setStatus('Start the camera before continuous AlexNet.', 'error');
    return;
  }}
  liveRunActive = true;
  liveFrameIndex = 0;
  liveRunButton.textContent = 'Stop continuous AlexNet';
  liveRunButton.classList.add('secondary');
  setStatus('Continuous AlexNet is starting. Predictions and the selected diagram layer update on each analysed frame.', 'ok');
  liveRunLoop();
}}

async function loadCaptions() {{
  captions = await fetch('/api/captions').then(response => response.json());
  selectStage('Input');
}}

async function loadImages() {{
  const data = await fetch('/api/images').then(response => response.json());
  imageSelect.innerHTML = '<option value="">Choose a curated image</option>';
  if (!data.images.length) {{
    const option = document.createElement('option');
    option.value = '';
    option.textContent = data.empty_message;
    imageSelect.appendChild(option);
    imageSelect.disabled = true;
    setStatus(data.empty_message, 'error');
    return;
  }}
  data.images.forEach(image => {{
    const option = document.createElement('option');
    option.value = image.name;
    option.textContent = image.label;
    option.dataset.url = image.url;
    imageSelect.appendChild(option);
  }});
}}

imageSelect.addEventListener('change', () => {{
  stopCamera({{clearInput: false}});
  clearResults();
  const selected = imageSelect.selectedOptions[0];
  runButton.disabled = !imageSelect.value;
  if (!imageSelect.value) {{
    inputState = {{kind: 'empty'}};
    selectStage('Input');
    return;
  }}
  inputState = {{kind: 'image', url: selected.dataset.url, label: selected.textContent}};
  selectStage('Input');
  setStatus('Input image ready. Run AlexNet, then use the diagram to inspect every layer.', 'ok');
}});

runButton.addEventListener('click', async () => {{
  if (!imageSelect.value) return;
  runButton.disabled = true;
  setStatus('Running AlexNet locally and capturing selectable layer responses…');
  try {{
    const response = await fetch('/api/run', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{image_name: imageSelect.value, fallback: fallbackToggle.checked}})
    }});
    const data = await response.json();
    renderAnalysisResult(data);
  }} catch (error) {{
    setStatus(`The local app could not complete the run: ${{error}}`, 'error');
  }} finally {{
    runButton.disabled = !imageSelect.value;
  }}
}});

startCameraButton.addEventListener('click', startCamera);
cameraRunButton.addEventListener('click', runCameraFrame);
liveRunButton.addEventListener('click', toggleLiveRun);
resetButton.addEventListener('click', resetDemo);
document.querySelectorAll('.network .layer').forEach(layer => {{
  layer.style.cursor = 'pointer';
  layer.addEventListener('click', () => selectStage(layer.dataset.layer, {{scroll: true}}));
  layer.addEventListener('keydown', event => {{
    if (event.key === 'Enter' || event.key === ' ') {{
      event.preventDefault();
      selectStage(layer.dataset.layer, {{scroll: true}});
    }}
  }});
}});

loadCaptions();
loadImages();
</script>
</body>
</html>
"""
