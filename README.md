# ISL Bridge

Recognises Indian Sign Language (ISL) greeting words from a camera or a short video clip. A college project.

**Live demo:** https://isltext.streamlit.app/

It knows nine words: Hello, How are you, Alright, Good Morning, Good afternoon, Good evening, Good night, Thank you, Pleased.

## How it works

1. **Extract:** MediaPipe finds 21 points on each hand and 33 on the body in every frame.
2. **Features:** positions are measured from the shoulders and hand shapes from the wrist. Each clip is resampled to 32 frames of 184 numbers.
3. **Classify:** a Random Forest votes for a word. Below 40% confidence the app says "not sure".

## Results

| Test | Accuracy |
|---|---|
| Leave-one-signer-out (train on 5 people, test on the 6th) | **88.4%** |
| Random split (same people in train and test, so inflated) | 94.2% |
| Chance (9 classes) | 11% |

The honest figure is 88.4%. The worst single signer scored 55.6% and the best 100%.

## Run it locally

Needs Python 3.12.

```
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Local webcam demo (press SPACE to record a sign): `python src/live/live_recognizer.py`

## Rebuild the model

The raw videos are not in this repo. Download the Greetings category of INCLUDE into `data/raw/`, then run in order:

```
python src/extract/extract_landmarks.py
python src/extract/assign_signers.py
python src/extract/fix_signers.py
python src/features/build_features.py
python src/models/baseline.py
python src/models/train_final.py
```

## Project layout

```
app/        Streamlit website and camera recorder
src/        extract, features, models, live demo
models/     trained classifier used by the website
eval/       baseline results and confusion matrix
```

## Limitations

- 9 words, 190 clips, 6 signers, one setting. New cameras, rooms and signers can score lower than 88%.
- Isolated words only. No continuous signing or facial expression.
- Not yet tested on our own recordings.
- A student prototype. Not a replacement for a human interpreter, and not for emergencies.

## Data and credits

Trained on the public INCLUDE dataset by AI4Bharat (https://zenodo.org/records/4010759). Hand and body tracking by Google MediaPipe.

Team: add names here.
