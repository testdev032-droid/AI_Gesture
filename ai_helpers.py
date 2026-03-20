"""AI helper functions for turning gestures into spells, text, and images."""

from __future__ import annotations

import requests
from typing import Optional, Tuple

from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont

import config


DEFAULT_GESTURE_TO_SPELL = {
    "Open Palm": "Shield of Light",
    "Fist": "Fire Burst",
    "Peace": "Healing Aura",
    "Point": "Lightning Strike",
    "Thumbs Up": "Phoenix Blessing",
    "No Gesture": "Dormant Magic",
}


SPELL_IMAGE_PROMPTS = {
    "Shield of Light": "A radiant magical shield forming from a glowing open hand, golden energy barrier, fantasy spell casting, cinematic fantasy art, luminous particles, enchanted temple",
    "Fire Burst": "A powerful fire spell exploding from a clenched fist, flames and sparks, fantasy combat magic, cinematic scene, glowing embers, epic magical action",
    "Healing Aura": "A peaceful healing spell with green and silver glowing energy, floating magical particles, restoration magic, fantasy art, gentle mystical light",
    "Lightning Strike": "A sharp bolt of magical lightning cast from a pointing hand, storm energy, glowing blue electricity, ancient ruins, epic fantasy spell scene",
    "Phoenix Blessing": "A fiery phoenix spirit blessing the caster, warm magical glow, flame wings, fantasy spell scene, sacred mystical fire",
    "Dormant Magic": "A quiet mystical chamber with faint glowing runes, sleeping magic, soft enchanted mist, fantasy environment art",
}


def get_spell_name_for_gesture(gesture_label: str, custom_mapping: Optional[dict] = None) -> str:
    mapping = custom_mapping or DEFAULT_GESTURE_TO_SPELL
    return mapping.get(gesture_label, "Arcane Pulse")


def build_spell_image_prompt(spell_name: str, gesture_label: str, extra_context: str = "") -> str:
    base_prompt = SPELL_IMAGE_PROMPTS.get(
        spell_name,
        f"A magical fantasy spell scene for {spell_name}, triggered by a {gesture_label} hand gesture, glowing enchanted energy, cinematic fantasy art",
    )
    if extra_context.strip():
        return f"{base_prompt}. {extra_context.strip()}"
    return base_prompt


def generate_magic_response(gesture_label: str, spell_name: str, extra_context: str = "") -> str:
    fallback_text = (
        f"Spell Name: {spell_name}\n\n"
        f"The studio reads the gesture '{gesture_label}' and channels {spell_name}. "
        f"Runes shimmer through the air as the magic awakens. "
        f"A wave of energy surges outward and the room responds like a living spellbook."
    )

    if not config.GROQ_API_KEY:
        return fallback_text

    prompt = f"""
You are the narrator inside a magical gesture-controlled AI studio.
The user performed this gesture: {gesture_label}
The detected spell is: {spell_name}
Extra scene context: {extra_context}

Write a student-friendly fantasy response with:
1. Spell Name
2. 3 short vivid lines of magical narration
3. A one-line power summary

Keep it immersive, exciting, and easy to read.
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.GROQ_TEXT_MODEL,
                "messages": [
                    {"role": "system", "content": "You create clean fantasy spell text for a Streamlit classroom project."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.9,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return fallback_text


def generate_magic_visual(prompt: str) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Generate a magical scene image from Hugging Face using InferenceClient.

    This avoids the deprecated direct api-inference model URL pattern that can return 410 Gone.
    """
    if not config.HF_API_KEY:
        return None, "HF_API_KEY is missing. Add your Hugging Face key in .env."

    try:
        client = InferenceClient(provider=config.HF_PROVIDER, api_key=config.HF_API_KEY)
        image = client.text_to_image(prompt, model=config.HF_IMAGE_MODEL)
        return image.convert("RGB"), None
    except Exception as error:
        return None, (
            f"Image generation failed with model '{config.HF_IMAGE_MODEL}' and provider "
            f"'{config.HF_PROVIDER}': {error}"
        )


def create_spell_card(spell_name: str, gesture_label: str, narration_text: str, scene_image: Optional[Image.Image]) -> Image.Image:
    card_width, card_height = 900, 560
    card = Image.new("RGB", (card_width, card_height), (14, 10, 24))
    draw = ImageDraw.Draw(card)

    draw.rounded_rectangle((10, 10, card_width - 10, card_height - 10), radius=26, outline=(175, 136, 255), width=4)
    draw.rounded_rectangle((24, 24, card_width - 24, card_height - 24), radius=20, outline=(80, 60, 140), width=2)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 20)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    if scene_image is not None:
        image_copy = scene_image.copy().resize((360, 360))
        card.paste(image_copy, (40, 110))
    else:
        draw.rounded_rectangle((40, 110, 400, 470), radius=24, fill=(35, 28, 58), outline=(120, 92, 190), width=2)
        draw.text((100, 270), "No spell image", font=body_font, fill=(220, 210, 255))

    draw.text((40, 36), "Gesture-to-Magic Studio", font=small_font, fill=(180, 165, 235))
    draw.text((430, 60), spell_name, font=title_font, fill=(245, 238, 255))
    draw.text((430, 108), f"Gesture: {gesture_label}", font=body_font, fill=(190, 175, 245))

    summary = narration_text.replace("\n", " ")
    wrapped_lines = wrap_text(summary, max_chars=38)
    y = 160
    for line in wrapped_lines[:10]:
        draw.text((430, y), line, font=body_font, fill=(235, 230, 255))
        y += 34

    draw.text((430, 495), "Magic recognized from webcam capture", font=small_font, fill=(160, 145, 220))
    return card


def wrap_text(text: str, max_chars: int = 40):
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        trial_line = " ".join(current_line + [word])
        if len(trial_line) <= max_chars:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))
    return lines
