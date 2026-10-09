"""Web demo: upload a short clip of one sign, get the recognised word.

Run locally (from the project root):
    streamlit run app/streamlit_app.py

Deployed on Streamlit Community Cloud with main file path: app/streamlit_app.py

It runs the same pipeline as the rest of the project:
    video -> MediaPipe hands + pose -> normalised features -> Random Forest -> word
"""

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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.extract.extract_landmarks import process_video  # noqa: E402
from src.live.live_recognizer import recognize  # noqa: E402

TASK_MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
MAX_SECONDS = 10

# Left/right partner of every pose landmark (used to un-mirror a video)
_SWAP = list(range(33))
for _a, _b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
               (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
    _SWAP[_a], _SWAP[_b] = _b, _a


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


st.set_page_config(page_title="ISL word recognizer", page_icon="🤟")
st.title("Indian Sign Language word recognizer")
st.caption("Research prototype · 9 greeting words · trained on 6 signers from the INCLUDE dataset")

bundle = load_bundle()
ensure_task_models()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence needed to answer", 0.20, 0.90, 0.40, 0.05,
                          help="Below this, the app says 'not sure' instead of guessing.")
    st.header("Words it knows")
    st.write(", ".join(bundle["names"]))

with st.expander("How to record a good clip", expanded=True):
    st.markdown(
        "- Record **one sign** only, about **2 to 5 seconds**, and keep the clip under "
        f"{MAX_SECONDS} seconds.\n"
        "- Stand or sit back so **both shoulders and both hands** are in view.\n"
        "- Good light and a plain background help.\n"
        "- Use the back camera if you can. If you use a front camera and the picture is mirrored, "
        "tick the box below."
    )

uploaded = st.file_uploader("Upload a short video of one sign",
                            type=["mp4", "mov", "avi", "webm", "mkv"])
mirrored = st.checkbox("My video is mirrored (some front cameras do this)")

if uploaded is not None:
    data = uploaded.getvalue()
    st.video(data)
    if st.button("Recognize", type="primary"):
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            info = inspect_video(tmp_path)
            if info is None:
                st.error("Could not read this video. Try an .mp4 file.")
            elif info["seconds"] is not None and info["seconds"] > MAX_SECONDS:
                st.error(f"This clip is about {info['seconds']:.0f} s long. "
                         f"Please trim it to under {MAX_SECONDS} s (just the sign).")
            else:
                with st.spinner("Finding hands and body, then classifying (this can take 10-40 s)..."):
                    out = process_video(tmp_path)
                if out is None:
                    st.error("Could not read any frames from this video.")
                else:
                    hands, pose, _fps = out
                    if mirrored:
                        hands, pose = unmirror(hands, pose)
                    res = recognize(bundle, list(hands), list(pose), info["aspect"], threshold)
                    if "error" in res:
                        st.warning(res["error"])
                    else:
                        if res["confident"]:
                            st.success(f"**{res['word']}**  ·  confidence {res['confidence']:.0%}")
                        else:
                            st.warning(f"Not sure. Best guess: **{res['word']}** ({res['confidence']:.0%})")
                        chart = pd.DataFrame(res["all"], columns=["word", "probability"]).set_index("word")
                        st.bar_chart(chart)
                        st.caption("Confidence is the share of the model's votes, not a guaranteed "
                                   "chance of being correct.")
        finally:
            os.unlink(tmp_path)

with st.expander("About this project and its limits"):
    st.markdown(
        "- Pipeline: video, then MediaPipe hand and body landmarks, then normalised features, "
        "then a Random Forest classifier.\n"
        "- Trained on 190 clips of 9 greeting words from the public INCLUDE dataset, signed by "
        "6 people in one setting. On a person it had never seen, the same model scored about 88% "
        "in our tests, and results on new people, cameras and rooms can be lower.\n"
        "- It only knows these 9 words. It is a student prototype and must not be relied on for "
        "emergencies or as a replacement for a human interpreter.\n"
        "- Your video is processed temporarily to make the prediction and is not saved by this app."
    )
