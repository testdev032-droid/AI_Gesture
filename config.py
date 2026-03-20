import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "keras_model.h5"
LABELS_PATH = MODELS_DIR / "labels.txt"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")

GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.1-8b-instant")

# With current Hugging Face Inference Providers, use a model ID instead of the old
# api-inference URL. The app calls it through huggingface_hub.InferenceClient.
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")

APP_TITLE = "Gesture-to-Magic Studio"
APP_ICON = "✨"
MAX_SPELL_LOG = 8
CAMERA_HELP = "Show one clear hand sign, then click Take Photo."
