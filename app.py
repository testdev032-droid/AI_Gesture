import streamlit as st
from PIL import Image

import config
from ai_helpers import (
    DEFAULT_GESTURE_TO_SPELL,
    build_spell_image_prompt,
    create_spell_card,
    generate_magic_response,
    generate_magic_visual,
    get_spell_name_for_gesture,
)
from gesture_utils import load_local_teachable_machine_model, predict_gesture_from_image


st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
)


def initialize_state():
    defaults = {
        "prediction": None,
        "captured_image": None,
        "input_source": "",
        "spell_name": "",
        "spell_text": "",
        "spell_prompt": "",
        "spell_scene_image": None,
        "spell_card_image": None,
        "spell_log": [],
        "gesture_mapping": {
            "Palm": "Shield of Light",
            "Peace": "Healing Aura",
            "Pointer": "Lightning Strike",
            "Thumbs Up": "Phoenix Blessing",
        },
        "last_generated_key": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


def reset_magic_session():
    defaults = {
        "prediction": None,
        "captured_image": None,
        "input_source": "",
        "spell_name": "",
        "spell_text": "",
        "spell_prompt": "",
        "spell_scene_image": None,
        "spell_card_image": None,
        "spell_log": [],
        "gesture_mapping": {
            "Palm": "Shield of Light",
            "Peace": "Healing Aura",
            "Pointer": "Lightning Strike",
            "Thumbs Up": "Phoenix Blessing",
        },
        "last_generated_key": "",
    }
    for key, value in defaults.items():
        st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_model_and_labels():
    return load_local_teachable_machine_model(
        str(config.MODEL_PATH),
        str(config.LABELS_PATH),
    )


def render_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top, #241339 0%, #100914 42%, #07070b 100%);
            color: #f5efff;
        }
        .hero-panel {
            padding: 22px;
            border-radius: 22px;
            border: 1px solid rgba(191, 158, 255, 0.20);
            background: linear-gradient(135deg, rgba(44,24,70,0.92), rgba(15,12,30,0.96));
            margin-bottom: 1rem;
        }
        .main-panel {
            padding: 20px;
            border-radius: 22px;
            border: 1px solid rgba(191, 158, 255, 0.18);
            background: rgba(255,255,255,0.03);
            box-shadow: 0 0 32px rgba(118, 80, 255, 0.10);
            margin-bottom: 1rem;
        }
        .status-card {
            padding: 14px;
            border-radius: 16px;
            background: rgba(148, 99, 255, 0.10);
            border: 1px solid rgba(180, 148, 255, 0.18);
            text-align: center;
            min-height: 86px;
        }
        .spell-box {
            padding: 16px;
            border-radius: 16px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 12px;
        }
        .small-note {
            color: #ccbdfa;
            font-size: 0.95rem;
        }
        .source-pill {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(108, 75, 214, 0.18);
            border: 1px solid rgba(193, 170, 255, 0.18);
            color: #efe7ff;
            margin-bottom: 12px;
        }
        .detect-card {
            padding: 14px;
            border-radius: 16px;
            background: rgba(94, 67, 170, 0.14);
            border: 1px solid rgba(186, 163, 255, 0.18);
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_hidden_prompt(prediction_label: str, spell_name: str, input_source: str) -> str:
    source_line = (
        "The magic scene should clearly feel triggered by a live webcam-captured hand gesture."
        if input_source == "webcam"
        else "The magic scene should clearly feel triggered by the uploaded hand gesture image."
    )

    return build_spell_image_prompt(
        spell_name,
        prediction_label,
        extra_context=(
            f"{source_line} Keep the hand sign influence visible in the composition. "
            "Make it cinematic, fantasy-rich, colorful, magical, and suitable for a student project."
        ),
    )


def generate_magic_bundle():
    if not st.session_state.prediction:
        st.warning("Capture or upload an image first.")
        return

    prediction_label = st.session_state.prediction["label"]
    spell_name = st.session_state.spell_name
    prompt = build_hidden_prompt(prediction_label, spell_name, st.session_state.input_source)
    st.session_state.spell_prompt = prompt

    with st.spinner("Casting AI magic from the detected gesture..."):
        spell_text = generate_magic_response(
            prediction_label,
            spell_name,
            f"This spell came from a {st.session_state.input_source} hand-gesture image.",
        )
        st.session_state.spell_text = spell_text

        image_result, error_message = generate_magic_visual(prompt)
        if image_result is not None:
            st.session_state.spell_scene_image = image_result
            st.session_state.spell_card_image = create_spell_card(
                spell_name,
                prediction_label,
                spell_text,
                image_result,
            )
        else:
            st.session_state.spell_scene_image = None
            st.session_state.spell_card_image = None
            st.error(error_message or "Could not generate the spell image.")

        st.session_state.spell_log.insert(
            0,
            {
                "gesture": prediction_label,
                "spell": spell_name,
                "text": spell_text,
                "source": st.session_state.input_source,
            },
        )
        st.session_state.spell_log = st.session_state.spell_log[: config.MAX_SPELL_LOG]


def show_header():
    st.markdown(
        f"""
        <div class="hero-panel">
            <h1 style="margin:0 0 6px 0;">✨ {config.APP_TITLE}</h1>
            <p style="margin:0;">
                Show a hand gesture in the webcam or upload a gesture image. The app detects the sign,
                maps it to a magical spell, then creates AI story and AI image output automatically.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_info_popup():
    with st.expander("ℹ️ What this app does and what it detects", expanded=False):
        st.write(
            """
            This app looks at a webcam capture or uploaded image, predicts the hand gesture using your
            Teachable Machine model, and turns it into a magic spell experience.
            """
        )
        st.markdown("**Detected gestures**")
        st.markdown("- Palm")
        st.markdown("- Peace")
        st.markdown("- Pointer")
        st.markdown("- Thumbs Up")
        st.markdown("**Flow**")
        st.markdown("1. Capture or upload a hand gesture image")
        st.markdown("2. Detect the gesture")
        st.markdown("3. Convert it into a spell")
        st.markdown("4. Generate AI spell story")
        st.markdown("5. Generate AI spell image")


def show_hud():
    hud1, hud2, hud3, hud4 = st.columns(4)
    with hud1:
        st.markdown(
            f'<div class="status-card"><b>Supported Gestures</b><br>Palm · Peace · Pointer · Thumbs Up</div>',
            unsafe_allow_html=True,
        )
    with hud2:
        current_gesture = st.session_state.prediction["label"] if st.session_state.prediction else "Waiting"
        st.markdown(
            f'<div class="status-card"><b>Detected Gesture</b><br>{current_gesture}</div>',
            unsafe_allow_html=True,
        )
    with hud3:
        current_spell = st.session_state.spell_name or "No Spell Yet"
        st.markdown(
            f'<div class="status-card"><b>Active Spell</b><br>{current_spell}</div>',
            unsafe_allow_html=True,
        )
    with hud4:
        st.markdown(
            f'<div class="status-card"><b>Spell Log</b><br>{len(st.session_state.spell_log)}</div>',
            unsafe_allow_html=True,
        )


def normalize_label(label: str) -> str:
    cleaned = label.strip()
    lookup = {
        "Open Palm": "Palm",
        "Palm": "Palm",
        "Peace": "Peace",
        "Pointer": "Pointer",
        "Point": "Pointer",
        "Thumbs Up": "Thumbs Up",
        "Thumbsup": "Thumbs Up",
        "No Gesture": "No Gesture",
    }
    return lookup.get(cleaned, cleaned)


def prediction_panel(current_image: Image.Image, source_name: str):
    st.session_state.captured_image = current_image
    st.session_state.input_source = source_name

    st.image(
        current_image,
        caption="Gesture image used for magic casting",
        use_container_width=True,
    )
    st.markdown(
        f'<div class="source-pill">Input source: {source_name.title()}</div>',
        unsafe_allow_html=True,
    )

    try:
        model, labels = get_model_and_labels()
        prediction = predict_gesture_from_image(model, labels, current_image)
        prediction["label"] = normalize_label(prediction["label"])
        for item in prediction["top_predictions"]:
            item["label"] = normalize_label(item["label"])
        st.session_state.prediction = prediction

        spell_name = get_spell_name_for_gesture(
            prediction["label"],
            st.session_state.gesture_mapping,
        )
        st.session_state.spell_name = spell_name

        st.success(f"Detected gesture: {prediction['label']}")
        st.progress(
            float(prediction["confidence"]),
            text=f"Confidence: {prediction['confidence']:.1%}",
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="detect-card"><b>Detected</b><br>{prediction["label"]}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="detect-card"><b>Mapped Spell</b><br>{spell_name}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("See all prediction scores"):
            for item in prediction["top_predictions"]:
                st.progress(
                    float(item["confidence"]),
                    text=f"{item['label']} — {item['confidence']:.1%}",
                )

        image_key = f"{source_name}:{prediction['label']}:{prediction['confidence']:.4f}"
        if st.session_state.last_generated_key != image_key:
            if st.button("Generate Magic From This Image ✨", use_container_width=True):
                generate_magic_bundle()
                st.session_state.last_generated_key = image_key

    except Exception as error:
        st.error(
            f"Could not load or run the local model: {error}\n\n"
            "Tip: use Python 3.10/3.11 and tensorflow==2.15.0 for Teachable Machine .h5 models."
        )


def show_input_panel():
    st.markdown('<div class="main-panel">', unsafe_allow_html=True)
    st.subheader("📷 Input")
    st.markdown(
        '<p class="small-note">Use webcam capture or upload an image. The app will detect Palm, Peace, Pointer, or Thumbs Up.</p>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Webcam Capture", "Upload Image"])

    with tabs[0]:
        camera_file = st.camera_input("Capture gesture from webcam", help=config.CAMERA_HELP)
        if camera_file is not None:
            webcam_image = Image.open(camera_file).convert("RGB")
            prediction_panel(webcam_image, "webcam")
        else:
            st.info("Take a photo with your webcam to detect a gesture.")

    with tabs[1]:
        uploaded_file = st.file_uploader("Upload a gesture image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            prediction_panel(uploaded_image, "upload")
        else:
            st.info("Upload a hand gesture image to detect a gesture.")
    st.markdown('</div>', unsafe_allow_html=True)


def show_output_panel():
    st.markdown('<div class="main-panel">', unsafe_allow_html=True)
    st.subheader("🪄 Output")

    if st.session_state.spell_name:
        st.markdown(
            f"""
            <div class="spell-box">
                <b>Spell Name</b><br>{st.session_state.spell_name}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.spell_text:
        st.markdown(
            f"""
            <div class="spell-box">
                <b>Spell Narration</b><br><br>{st.session_state.spell_text.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    img1, img2 = st.columns(2)
    with img1:
        if st.session_state.spell_scene_image is not None:
            st.image(
                st.session_state.spell_scene_image,
                caption="AI-generated spell scene",
                use_container_width=True,
            )
        else:
            st.info("The spell image will appear here after generation.")

    with img2:
        if st.session_state.spell_card_image is not None:
            st.image(
                st.session_state.spell_card_image,
                caption="Collectible spell card",
                use_container_width=True,
            )
        else:
            st.info("The spell card will appear here after generation.")

    st.markdown('</div>', unsafe_allow_html=True)


def show_spell_log():
    st.markdown('<div class="main-panel">', unsafe_allow_html=True)
    top1, top2 = st.columns([0.8, 0.2])
    with top1:
        st.subheader("📚 Spell Log")
    with top2:
        if st.button("Reset Studio", use_container_width=True):
            reset_magic_session()
            st.rerun()

    if st.session_state.spell_log:
        for index, item in enumerate(st.session_state.spell_log, start=1):
            with st.expander(f"Spell {index}: {item['spell']} ({item['gesture']})"):
                st.write(f"Source: {item['source']}")
                st.write(item["text"])
    else:
        st.caption("Your spell history will appear here after you generate magic.")
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    render_styles()
    show_header()
    show_info_popup()
    show_hud()

    left_col, right_col = st.columns([0.95, 1.05], gap="large")
    with left_col:
        show_input_panel()
    with right_col:
        show_output_panel()

    show_spell_log()


if __name__ == "__main__":
    main()
