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
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Playfair+Display:ital,wght@1,400&family=Roboto:wght@300;400;500&family=Roboto+Mono:wght@400;500&display=swap');
:root{--r:#CF1A31;--pale:#F7D6DA;--mono:'Roboto Mono',ui-monospace,Consolas,monospace;--serif:'Playfair Display',Georgia,serif}
.stApp{background:#fff;color:var(--r);font-family:'Roboto',system-ui,sans-serif;font-weight:300}
header[data-testid="stHeader"],#MainMenu,footer{display:none}
.block-container{max-width:none;padding:14px 25px 3rem}
.stApp p,.stApp label,.stApp li{color:var(--r)}
em{font-family:var(--serif);font-style:italic;font-weight:400}
.mono{font-family:var(--mono);font-size:.8rem;letter-spacing:.04em;text-transform:uppercase}
.hero{min-height:94vh;display:flex;flex-direction:column}
.mark{font-family:'Archivo Black',Impact,sans-serif;font-size:15vw;line-height:.86;letter-spacing:-.02em;white-space:nowrap;margin:0 0 .7rem;color:var(--r)}
.row3{display:flex;justify-content:space-between;align-items:baseline;font-size:1.05rem;gap:1rem;flex-wrap:wrap}
.row3 nav{display:flex;gap:14px}
.row3 a{color:var(--r);text-decoration:none}
.row3 a:hover{text-decoration:underline}
.row3 .serif{font-family:var(--serif);font-style:italic}
.statement{margin:12vh 0 0 50%}
.big{font-weight:300;font-size:clamp(2.2rem,4.4vw,4.4rem);line-height:1.02;letter-spacing:-.025em;margin:0;padding:0;color:var(--r)}
.big em{font-size:1.04em;letter-spacing:-.04em}
.corners{margin-top:auto;display:flex;justify-content:space-between;align-items:flex-end;padding-top:6vh}
.zoomwrap{height:230vh;width:100vw;margin-left:calc(50% - 50vw);margin-top:-14vh;position:relative;z-index:2}
.zoomstick{position:sticky;top:0;height:100vh;display:grid;place-items:start center}
.zoomframe{width:min(900px,90vw);height:50vh;background:#161616;color:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1.2rem;overflow:hidden}
.zoomframe img{height:55%;max-width:80%}
.zoomframe .zl{display:flex;flex-direction:column;align-items:center;gap:.3rem;text-align:center}
.zoomframe .serif{font-family:var(--serif);font-style:italic;font-size:clamp(1.6rem,3vw,3rem)}
@supports (animation-timeline:scroll()){.zoomframe{animation:grow linear both;animation-timeline:scroll(nearest block);animation-range:0px 100vh}}
@keyframes grow{from{width:min(900px,90vw);height:50vh}to{width:100vw;height:100vh}}
.sec{margin:6vh 0 2rem;max-width:980px;margin-left:auto;margin-right:auto}
.stTabs,[data-testid="stExpander"]{max-width:980px;margin-left:auto;margin-right:auto}
.stTabs [data-baseweb="tab-list"]{gap:6px}
.stTabs [data-baseweb="tab-border"]{background:var(--r);height:1px}
.stTabs [data-baseweb="tab-highlight"]{background:var(--r);height:3px}
.stTabs [data-baseweb="tab"]{color:var(--r);font-weight:400;height:46px;padding:0 14px;background:transparent}
[data-testid="stExpander"] details{border:1px solid var(--r);border-radius:0;background:#fff}
[data-testid="stFileUploaderDropzone"]{border:1.5px dashed var(--r);border-radius:0;background:#fff}
.stButton>button{border:1.5px solid var(--r);border-radius:999px;background:#fff;color:var(--r);font-family:var(--serif);font-style:italic;font-size:1.15rem;padding:.35rem 1.6rem}
.stButton>button:hover{background:var(--r);color:#fff;border-color:var(--r)}
.hint{font-size:.98rem;line-height:1.6;margin:.9rem 0 1rem;color:var(--r)}
.ticket{margin-top:1.4rem;border:1.5px solid var(--r);padding:16px 22px 0;background:#fff;animation:print .9s cubic-bezier(.2,.7,.2,1) both}
@keyframes print{from{clip-path:inset(0 0 100% 0);transform:translateY(-10px)}to{clip-path:inset(0 0 -4px -4px);transform:none}}
.tk-top{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.74rem}
.ticket .word{font-family:var(--serif);font-style:italic;font-weight:400;font-size:clamp(2.8rem,9vw,5rem);letter-spacing:-.03em;line-height:1.05;margin:.5rem 0 .2rem}
.ticket.unsure .word{opacity:.55}
.ticket .sub{font-size:1rem}
.ticket.err{border-style:dashed}
.bars{margin:1.1rem 0 1.2rem;display:grid;gap:6px}
.brow{display:grid;grid-template-columns:120px 1fr 44px;align-items:center;gap:10px;font-size:.84rem}
.brow .track{height:6px;background:var(--pale)}
.brow .fill{height:100%;background:var(--r);opacity:.45}
.brow.top{font-weight:500}
.brow.top .fill{opacity:1}
.brow .pct{text-align:right;font-family:var(--mono);font-size:.76rem}
.tear{display:flex;align-items:center;gap:8px;margin:0 -22px;padding:0 22px;font-family:var(--mono);font-size:.72rem}
.tear i{flex:1;border-top:1.5px dashed var(--r)}
.stub{display:flex;align-items:center;gap:14px;padding:12px 0 14px;font-size:.78rem}
.stub b{flex:0 0 110px;height:26px;background:repeating-linear-gradient(90deg,var(--r) 0 2px,#fff 2px 5px,var(--r) 5px 6px,#fff 6px 10px)}
.about{display:grid;grid-template-columns:1fr 1fr;column-gap:3vw;row-gap:12vh;margin:16vh 0}
.about .a2{align-self:end;max-width:30rem;font-size:1rem;line-height:1.4;color:var(--r)}
.about .a4{align-self:start}
.fig{display:flex;gap:18px;align-items:flex-end;max-width:460px}
.fig img{width:42%}
.cmp{flex:1;display:grid;gap:8px;font-size:.82rem}
.cmp .b{height:8px;background:var(--pale)}
.cmp .b i{display:block;height:100%;background:var(--r)}
.cmp .b.dim i{opacity:.45}
.cmp b{font-family:var(--mono);font-weight:500}
.how{margin:10vh 0 3rem}
.howhead{display:flex;justify-content:space-between;align-items:center;gap:2rem;flex-wrap:wrap;margin-bottom:2rem}
.ab{display:flex;gap:min(18vw,240px);font-size:1rem}
.chipo{border:1.5px solid var(--r);border-radius:50%;padding:.9rem 2.4rem;font-family:var(--serif);font-style:italic;font-size:1.9rem}
.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}
.panel{position:relative;background:var(--r);color:#fff;min-height:440px;padding:25px;display:flex;flex-direction:column;justify-content:flex-end}
.panel::before{content:"";position:absolute;left:0;right:0;top:40%;border-top:1.5px dashed rgba(255,255,255,.85)}
.panel::after{content:"";position:absolute;left:0;right:0;top:58%;border-top:1px solid rgba(255,255,255,.7)}
.panel .n{position:absolute;top:22px;left:25px;font-family:var(--mono);font-weight:500}
.panel h3{margin:0;color:#fff;font-weight:400;font-size:2rem;line-height:1.05;padding:0}
.panel h3 em{display:block}
.stApp .panel p{color:#fff;font-size:1rem;line-height:1.3;margin:1rem 0 0}
.panel code{display:block;margin-top:.8rem;font-family:var(--mono);font-size:.74rem;color:#fff;background:none;padding:0;opacity:.85}
.fine{font-family:var(--mono);font-size:.74rem;line-height:1.7;max-width:60rem;margin:0 0 2rem}
@media (max-width:900px){.statement{margin-left:0}.about{grid-template-columns:1fr}.panels{grid-template-columns:1fr 1fr}}
@media (max-width:600px){.panels{grid-template-columns:1fr}.brow{grid-template-columns:96px 1fr 40px}.ab{gap:30vw}}
@media (prefers-reduced-motion:reduce){.zoomframe,.ticket{animation:none}}
</style>
"""


def md(block):
    """Render an HTML block. Lines are stripped so Markdown never treats them as code."""
    st.markdown("\n".join(line.strip() for line in block.splitlines() if line.strip()),
                unsafe_allow_html=True)


def result_html(res, threshold):
    if "error" in res:
        return ('<div class="ticket err"><div class="tk-top"><span>Could not read this clip</span><span>ISL/ERR</span></div>'
                f'<div class="sub" style="margin:.6rem 0 1rem">{html.escape(res["error"])}</div></div>')
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
    return (f'<div class="ticket {state}"><div class="tk-top"><span>Recognised sign</span><span>ISL/RES.01</span></div>'
            f'<div class="word">{title}</div><div class="sub">{sub}</div>'
            f'<div class="bars">{"".join(rows)}</div>'
            '<div class="tear"><span>&#9986;</span><i></i><span>tear along line</span></div>'
            '<div class="stub"><span>Student prototype. Not a replacement for a human interpreter.</span><b></b></div></div>')


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.set_page_config(page_title="ISL Bridge", page_icon="🤟", layout="centered",
                   initial_sidebar_state="collapsed")
st.markdown(STYLE, unsafe_allow_html=True)

bundle = load_bundle()
ensure_task_models()

def hand_uri(line, dot, fill):
    chains = [[(100, 170), (74, 152), (54, 132), (42, 110), (34, 90)],
              [(100, 170), (84, 122), (80, 92), (78, 66), (77, 44)],
              [(100, 170), (100, 118), (102, 82), (103, 54), (104, 30)],
              [(100, 170), (116, 122), (122, 90), (125, 66), (127, 46)],
              [(100, 170), (130, 134), (140, 108), (146, 90), (150, 74)]]
    parts = ['<polyline points="%s" fill="none" stroke="%s" stroke-width="1.6" stroke-linejoin="round"/>'
             % (" ".join(f"{x},{y}" for x, y in ch), line) for ch in chains]
    for x, y in sorted({p for ch in chains for p in ch}):
        parts.append(f'<circle cx="{x}" cy="{y}" r="3.2" fill="{fill}" stroke="{dot}" stroke-width="1.6"/>')
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">' + "".join(parts) + "</svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


md("""
<div class="hero">
<div class="mark">ISL BRIDGE</div>
<div class="row3"><span>Indian Sign Language recognizer</span>
<nav><a href="#try">Try it</a><a href="#about">About</a><a href="#how">How it works</a></nav>
<span class="serif">student prototype</span></div>
<div class="statement"><h1 class="big">Show one sign to the camera <em>and get the word back.</em></h1></div>
<div class="corners"><span class="mono">Model<br>Random Forest, 9 words</span><span class="mono">&#9670; Ready</span></div>
</div>
""")
md("""
<div class="zoomwrap"><div class="zoomstick"><div class="zoomframe">
<img src="@@H@@" alt="Hand landmarks">
<div class="zl"><span class="mono">21 points per hand, 33 on the body</span><span class="serif">Hello</span></div>
</div></div></div>
""".replace("@@H@@", hand_uri("#fff", "#CF1A31", "#161616")))
md("""<div id="try" class="sec"><h2 class="big">Try it. <em>Record one sign.</em></h2></div>""")

with st.expander("Settings"):
    threshold = st.slider("Confidence needed before it gives an answer", 0.20, 0.90, DEFAULT_THRESHOLD, 0.05,
                          help="Below this the app says 'not sure' instead of guessing.")

tab_cam, tab_up = st.tabs(["Live camera", "Upload a clip"])

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

STEPS = [
    ("Extract", "landmarks", "extract_landmarks.py", "MediaPipe finds 21 points on each hand and 33 on the body, frame by frame. A missing hand is stored as NaN, never as zero."),
    ("Find the", "signers", "assign_signers.py", "The dataset has no person labels, so clips are grouped by clothing colour and checked by eye. Six people."),
    ("Build", "features", "build_features.py", "Positions are measured from the shoulders and hand shapes from the wrist. Every clip becomes 32 frames of 184 numbers."),
    ("Test", "honestly", "baseline.py", "Train on five people, test on the sixth, six times over. The Random Forest scores 88.4% on people it has never seen."),
    ("Train the", "demo model", "train_final.py", "The final forest learns from all 190 clips and is saved as one small file that the website loads."),
    ("Serve the", "result", "streamlit_app.py", "The browser records a clip, the server repeats steps 1 and 3, and the forest votes. Under 40% confidence it says not sure."),
]
panels = "".join(f'<div class="panel"><span class="n">0{i}</span><h3>{a}<em>{b}</em></h3><p>{d}</p><code>{f}</code></div>'
                 for i, (a, b, f, d) in enumerate(STEPS, 1))
md("""
<div id="about" class="about">
<h2 class="big">It reads your hands, not your face. <em>Nine greetings, six signers, one honest test.</em></h2>
<div><div class="fig"><img src="@@H@@" alt="Hand landmarks"><div class="cmp">
<span>Random split <b>94.2%</b></span><div class="b dim"><i style="width:94.2%"></i></div>
<span>Unseen signer <b>88.4%</b></span><div class="b"><i style="width:88.4%"></i></div></div></div></div>
<p class="a2">A college project built on the public INCLUDE dataset. MediaPipe finds the landmarks and a Random Forest names the word. The 94.2% score lets the same people appear in training and testing, so we report 88.4% instead. <em>Worst signer 55.6%, best 100%.</em></p>
<h2 class="big a4">First, we test on people it has never seen &rarr; <em>then we trust the number.</em></h2>
</div>
""".replace("@@H@@", hand_uri("#CF1A31", "#CF1A31", "#fff")))
md(f"""
<div id="how" class="how"><div class="howhead">
<div class="ab"><span>video</span><span>&rarr;</span><span>word</span></div>
<p class="big" style="font-size:1.4rem">Six steps, one pipeline. <em>Each one is a file.</em></p>
<div class="chipo">How it works</div></div>
<div class="panels">{panels}</div></div>
<div class="fine">Student prototype. It knows nine greetings, does not read continuous signing or facial expression, and is not a replacement for a human interpreter or for use in emergencies. Videos are processed temporarily to make the prediction and are not saved. New cameras, rooms and signers can score lower than 88%.</div>
""")
