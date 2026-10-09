"""ISL Bridge: recognise Indian Sign Language greeting words from a camera or a clip.

Run locally (from the project root):
    streamlit run app/streamlit_app.py

Deployed on Streamlit Community Cloud with main file path: app/streamlit_app.py

Pipeline (same as the rest of the project):
    video -> MediaPipe hands + pose -> normalised features -> Random Forest -> word

Two ways in:
    * Live camera: a small browser component (app/recorder/index.html) records a
      few seconds from your webcam and sends the clip to Python.
    * Upload: pick a video file.
"""

import base64
import html
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # the extractor uses paths relative to the project root
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import joblib  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from src.extract.extract_landmarks import process_video  # noqa: E402
from src.live.live_recognizer import recognize  # noqa: E402

TASK_MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
MAX_SECONDS = 10      # longest clip we accept
RECORD_SECONDS = 5    # longest live recording
DEFAULT_THRESHOLD = 0.40

# Left/right partner of every pose landmark (used to un-mirror an uploaded selfie video)
_SWAP = list(range(33))
for _a, _b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
               (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
    _SWAP[_a], _SWAP[_b] = _b, _a

recorder = components.declare_component("isl_recorder", path=str(ROOT / "app" / "recorder"))


# ----------------------------------------------------------------------------
# Pipeline helpers
# ----------------------------------------------------------------------------
def unmirror(hands, pose):
    """Undo a mirrored (selfie-style) video: flip x and swap left/right."""
    hands = hands[:, ::-1].copy()
    hands[..., 0] = 1.0 - hands[..., 0]
    pose = pose[:, _SWAP].copy()
    pose[..., 0] = 1.0 - pose[..., 0]
    return hands, pose


@st.cache_resource(show_spinner="Downloading hand and pose models (first run only)...")
def ensure_task_models():
    folder = ROOT / "data" / "models"
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in TASK_MODEL_URLS.items():
        path = folder / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
    return True


@st.cache_resource(show_spinner="Loading the classifier...")
def load_bundle():
    for path in (ROOT / "models" / "isl_rf.joblib", ROOT / "data" / "models" / "isl_rf.joblib"):
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError("Classifier not found. Run python src/models/train_final.py first.")


def inspect_video(path):
    """Read the first frame (for the true aspect ratio) and estimate the length."""
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    seconds = n / fps if fps and fps > 0 and n and n > 0 else None
    return {"aspect": w / h, "seconds": seconds}


def analyse(data, suffix, mirrored, bundle):
    """Run the full pipeline on video bytes. Returns a result dict (or {'error': ...})."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        info = inspect_video(tmp_path)
        if info is None:
            return {"error": "This video could not be read. Try an .mp4 file."}
        if info["seconds"] is not None and info["seconds"] > MAX_SECONDS:
            return {"error": f"This clip is about {info['seconds']:.0f} s long. "
                             f"Please trim it to under {MAX_SECONDS} s (just the sign)."}
        out = process_video(tmp_path)
        if out is None:
            return {"error": "No frames could be read from this video."}
        hands, pose, _fps = out
        if mirrored:
            hands, pose = unmirror(hands, pose)
        return recognize(bundle, list(hands), list(pose), info["aspect"], DEFAULT_THRESHOLD)
    finally:
        os.unlink(tmp_path)


# ----------------------------------------------------------------------------
# Look and feel
# ----------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 12% -10%, #1A2550 0%, rgba(11,16,32,0) 60%), radial-gradient(900px 500px at 100% 0%, #2A1B4D 0%, rgba(11,16,32,0) 55%), #0B1020; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 860px; padding-top: 2.2rem; padding-bottom: 3rem; }
.hero { text-align: center; padding: 1.2rem 0 0.4rem; }
.eyebrow { letter-spacing: .22em; text-transform: uppercase; font-size: .72rem; color: #9AA6C4; }
.hero h1 { font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 3.4rem; line-height: 1.05; margin: .35rem 0 .5rem; padding: 0;
  background: linear-gradient(120deg, #DCE6FF 0%, #A9C2FF 40%, #C7A8FF 75%, #F2B8D6 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: #AEB9D6; font-size: 1.05rem; margin: 0 auto; max-width: 560px; line-height: 1.6; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 1.6rem 0 1rem; }
.stat { border: 1px solid rgba(255,255,255,.09); border-radius: 16px; padding: 14px 10px; text-align: center; background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.015)); }
.stat b { display: block; font-family: 'Fraunces', Georgia, serif; font-size: 1.7rem; font-weight: 600; color: #E8ECF8; }
.stat span { font-size: .78rem; color: #9AA6C4; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: .4rem 0 1.4rem; }
.chip { font-size: .8rem; padding: 5px 12px; border-radius: 999px; color: #C9D4F2; border: 1px solid rgba(255,255,255,.1); background: rgba(255,255,255,.04); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(255,255,255,.08); }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 16px; border-radius: 10px 10px 0 0; color: #9AA6C4; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #E8ECF8; }
.hint { color: #9AA6C4; font-size: .92rem; line-height: 1.6; margin: .8rem 0 1rem; }
.result { border-radius: 22px; padding: 22px 24px; margin-top: 1.1rem; border: 1px solid rgba(255,255,255,.1);
  background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02)); }
.result.ok { border-color: rgba(94,234,212,.35); box-shadow: 0 0 60px -25px rgba(94,234,212,.5); }
.result.unsure { border-color: rgba(245,193,108,.35); box-shadow: 0 0 60px -25px rgba(245,193,108,.4); }
.result.err { border-color: rgba(255,107,129,.35); }
.result .label { font-size: .72rem; letter-spacing: .18em; text-transform: uppercase; color: #9AA6C4; }
.result .word { font-family: 'Fraunces', Georgia, serif; font-size: 2.8rem; font-weight: 600; line-height: 1.1; margin: .25rem 0; color: #E8ECF8; }
.result.ok .word { background: linear-gradient(120deg, #9EF3E2, #8FB6FF); -webkit-background-clip: text; background-clip: text; color: transparent; }
.result.unsure .word { color: #F5C16C; }
.result .sub { color: #AEB9D6; font-size: .95rem; }
.bars { margin-top: 1.1rem; display: grid; gap: 7px; }
.brow { display: grid; grid-template-columns: 120px 1fr 44px; align-items: center; gap: 10px; font-size: .85rem; color: #AEB9D6; }
.brow .track { height: 8px; border-radius: 8px; background: rgba(255,255,255,.07); overflow: hidden; }
.brow .fill { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #5B7BFF, #B79CFF); }
.brow.top { color: #E8ECF8; font-weight: 600; }
.brow.top .fill { background: linear-gradient(90deg, #5EEAD4, #8FB6FF); }
.brow .pct { text-align: right; font-variant-numeric: tabular-nums; }
.steps { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: .8rem; }
.step { border: 1px solid rgba(255,255,255,.09); border-radius: 16px; padding: 14px 16px; background: rgba(255,255,255,.03); }
.step b { color: #E8ECF8; display: block; margin-bottom: 2px; }
.step span { color: #9AA6C4; font-size: .88rem; line-height: 1.5; }
.fine { color: #7F8BAA; font-size: .8rem; line-height: 1.6; margin-top: 1.4rem; text-align: center; }
[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed rgba(143,182,255,.4); border-radius: 18px; background: rgba(255,255,255,.03); }
.stButton > button { border-radius: 12px; font-weight: 600; border: 0; color: #0B1020; background: linear-gradient(120deg, #8FB6FF, #B79CFF); padding: .55rem 1.4rem; }
.stButton > button:hover { filter: brightness(1.08); color: #0B1020; }
@media (max-width: 640px) { .hero h1 { font-size: 2.5rem; } .steps { grid-template-columns: 1fr; } .brow { grid-template-columns: 96px 1fr 40px; } }
</style>
"""


def md(block):
    """Render an HTML block. Lines are stripped so Markdown never treats them as code."""
    st.markdown("\n".join(line.strip() for line in block.splitlines() if line.strip()),
                unsafe_allow_html=True)


def result_html(res, threshold):
    if "error" in res:
        return ('<div class="result err"><div class="label">Could not analyse</div>'
                f'<div class="sub" style="margin-top:.4rem">{html.escape(res["error"])}</div></div>')
    word, conf = res["word"], res["confidence"]
    ok = conf >= threshold
    state = "ok" if ok else "unsure"
    title = html.escape(word) if ok else "Not sure"
    sub = (f"Confidence {conf:.0%}" if ok else
           f"Best guess: {html.escape(word)} ({conf:.0%}). Try again with a clearer clip.")
    rows = []
    for i, (name, p) in enumerate(res["all"]):
        cls = "brow top" if i == 0 else "brow"
        rows.append(f'<div class="{cls}"><span>{html.escape(name)}</span>'
                    f'<div class="track"><div class="fill" style="width:{max(p * 100, 1.5):.1f}%"></div></div>'
                    f'<span class="pct">{p:.0%}</span></div>')
    return (f'<div class="result {state}"><div class="label">Recognised sign</div>'
            f'<div class="word">{title}</div><div class="sub">{sub}</div>'
            f'<div class="bars">{"".join(rows)}</div></div>')


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.set_page_config(page_title="ISL Bridge", page_icon="🤟", layout="centered",
                   initial_sidebar_state="collapsed")
st.markdown(STYLE, unsafe_allow_html=True)

bundle = load_bundle()
ensure_task_models()

chips = "".join(f'<span class="chip">{html.escape(n)}</span>' for n in bundle["names"])
md(f"""
<div class="hero">
<div class="eyebrow">Indian Sign Language</div>
<h1>ISL Bridge</h1>
<p>Show a sign to your camera and watch it become a word. A student research prototype that recognises nine everyday greetings.</p>
</div>
<div class="stats">
<div class="stat"><b>{len(bundle["names"])}</b><span>greeting words</span></div>
<div class="stat"><b>6</b><span>training signers</span></div>
<div class="stat"><b>88%</b><span>on signers it had never seen</span></div>
</div>
<div class="chips">{chips}</div>
""")

with st.expander("Settings"):
    threshold = st.slider("Confidence needed before it gives an answer", 0.20, 0.90, DEFAULT_THRESHOLD, 0.05,
                          help="Below this the app says 'not sure' instead of guessing.")

tab_cam, tab_up, tab_about = st.tabs(["Live camera", "Upload a clip", "How it works"])

with tab_cam:
    md("""
    <div class="hint">Press <b>Start camera</b>, then <b>Record a sign</b>. After a 3 second countdown it records
    for up to 5 seconds. Sit or stand back so <b>both shoulders and both hands</b> are in view, with good light.</div>
    """)
    rec = recorder(max_seconds=RECORD_SECONDS, countdown=3, key="cam", default=None)
    if rec and rec.get("id") != st.session_state.get("cam_id"):
        st.session_state["cam_id"] = rec["id"]
        ext = ".mp4" if "mp4" in rec.get("mime", "") else ".webm"
        with st.spinner("Finding hands and body, then classifying (10 to 40 seconds)..."):
            st.session_state["cam_result"] = analyse(base64.b64decode(rec["b64"]), ext, False, bundle)
    if st.session_state.get("cam_result"):
        md(result_html(st.session_state["cam_result"], threshold))

with tab_up:
    md("""
    <div class="hint">Upload a short video of <b>one sign</b> (2 to 5 seconds, under 10). Back-camera footage works best.</div>
    """)
    uploaded = st.file_uploader("Upload a short video of one sign", type=["mp4", "mov", "avi", "webm", "mkv"],
                                label_visibility="collapsed")
    mirrored = st.checkbox("My video is mirrored (some front cameras do this)")
    if uploaded is not None:
        data = uploaded.getvalue()
        st.video(data)
        if st.button("Recognize"):
            with st.spinner("Finding hands and body, then classifying (10 to 40 seconds)..."):
                st.session_state["up_result"] = analyse(data, Path(uploaded.name).suffix or ".mp4", mirrored, bundle)
    if uploaded is not None and st.session_state.get("up_result"):
        md(result_html(st.session_state["up_result"], threshold))

with tab_about:
    md("""
    <div class="hint">Everything runs the same way for the camera and for uploads.</div>
    <div class="steps">
    <div class="step"><b>1. See</b><span>MediaPipe, a pretrained model, finds 21 points on each hand and the upper body in every frame.</span></div>
    <div class="step"><b>2. Normalise</b><span>Positions are measured from the shoulders, so distance and position in the frame matter less.</span></div>
    <div class="step"><b>3. Resample</b><span>Every clip becomes exactly 32 frames, so fast and slow signers can be compared.</span></div>
    <div class="step"><b>4. Classify</b><span>A Random Forest trained on 190 clips of 9 words from the public INCLUDE dataset votes for a word.</span></div>
    </div>
    <div class="fine">The 88% figure comes from leave-one-signer-out testing: train on 5 people, test on the 6th. It comes
    from the dataset's own people and setting, so new cameras, rooms and signers can score lower. It only knows nine
    words, does not read continuous signing or facial expression, and is not a replacement for a human interpreter or
    for use in emergencies. Videos are processed temporarily to make the prediction and are not saved by this app.</div>
    """)
