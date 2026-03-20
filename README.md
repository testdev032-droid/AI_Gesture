# Gesture-to-Magic Studio

A Streamlit project where a **local Teachable Machine gesture model** detects hand signs and turns them into AI-powered magical actions.

## Folder structure
- `app.py` → main Streamlit app
- `config.py` → loads keys and local model paths from `.env`
- `ai_helpers.py` → Groq + Hugging Face helper functions
- `gesture_utils.py` → load local Teachable Machine model + run predictions
- `models/keras_model.h5` → exported Teachable Machine model
- `models/labels.txt` → class labels file
- `.env.example` → sample environment variables
- `requirements.txt` → dependencies

## Setup

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Create your `.env`
Copy `.env.example` to `.env` and fill in values.

### 3) Export your Teachable Machine model
In Teachable Machine:
- Create an **Image Project**
- Train your gesture classes
- Export as **Keras**

You will get:
- `keras_model.h5`
- `labels.txt`

### 4) Put files in the models folder
```text
models/
  keras_model.h5
  labels.txt
```

### 5) Add API keys
```env
GROQ_API_KEY=your_groq_key
HF_API_KEY=your_huggingface_key
```

### 6) Run
```bash
streamlit run app.py
```

## Notes
- Webcam input uses `st.camera_input`, which captures a snapshot from the browser camera.
- If API keys are missing, the app still works with fallback text, but image generation may be skipped.
- If your model files are in a different location, update `TM_MODEL_PATH` and `TM_LABELS_PATH` in `.env`.
