import streamlit as st
import torch
from PIL import Image
import numpy as np
from skimage import color, transform
import io
import os

from models import UNet, ResNetUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "Colorization (ResNet-UNet)": {
        "path": "best_resnet_model.pth",
        "class": ResNetUNet
    },
    "Colorization (Standard U-Net)": {
        "path": "best_model_aggressive.pth",
        "class": UNet
    }
}

# --- PAGE SETUP ---

st.set_page_config(page_title="Colorizer", layout="wide")
st.title("🎨Image Colorizer Studio")


# --- MODEL LOADER ---

@st.cache_resource(show_spinner=False)
def load_model(model_name):
    model_info = MODELS[model_name]
    model_class = model_info["class"]
    model_path = model_info["path"]

    model = model_class().to(DEVICE)

    try:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            return model, None
        else:
            return None, f"File not found: {model_path}"
    except Exception as e:
        return None, str(e)


# --- SIDEBAR ---

st.sidebar.header("⚙️ Settings")
selected_model = st.sidebar.selectbox("Select Model", list(MODELS.keys()), index=0)
saturation = st.sidebar.slider("Saturation Boost", 0.0, 3.0, 1.6, help="Fixes the 'Sepia Effect'")

with st.spinner(f"Loading {selected_model}..."):
    model, err = load_model(selected_model)
    if model is None:
        st.error(f"❌ {err}")
        st.stop()

# --- MAIN LOGIC ---

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    # 1. Load Original High-Res Image
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    # 2. Extract High-Res L Channel
    orig_lab = color.rgb2lab(image_np)
    orig_L = orig_lab[:, :, 0]
    orig_h, orig_w = orig_L.shape

    img_resized = image_pil.resize((256, 256))
    img_resized_np = np.array(img_resized)
    img_resized_lab = color.rgb2lab(img_resized_np).astype("float32")

    # Normalize L to [-1, 1]
    img_l_input = (img_resized_lab[:, :, 0] / 50.0) - 1.0
    tensor_input = torch.from_numpy(img_l_input).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with col1:
        st.subheader("Original (Grayscale)")
        st.image(image_pil.convert('L'), caption=f" Input ({orig_w}x{orig_h})", use_container_width=True)

    if st.button("✨ Colorize Image"):
        with st.spinner("Processing..."):
            with torch.no_grad():
                ab_pred = model(tensor_input)
                ab_pred = ab_pred.cpu().numpy()[0]

            # --- POST-PROCESSING ---

            # A. Resize predicted 'ab' to match Original Size
            ab_high_res = transform.resize(
                ab_pred.transpose((1, 2, 0)),
                (orig_h, orig_w),
                anti_aliasing=True
            )

            # B. Apply Saturation Boost
            ab_high_res = ab_high_res * 128.0 * saturation

            # C. Combine High-Res L with High-Res Color
            lab_final = np.zeros((orig_h, orig_w, 3))
            lab_final[:, :, 0] = orig_L
            lab_final[:, :, 1:] = ab_high_res

            # D. Convert to RGB
            with np.errstate(invalid='ignore'):
                final_rgb = color.lab2rgb(lab_final)

            final_image = Image.fromarray((final_rgb * 255).astype(np.uint8))

        with col2:
            st.subheader("Model Prediction")
            st.image(final_image, caption=f"Result (Sat: {saturation}x)", use_container_width=True)

        buf = io.BytesIO()
        final_image.save(buf, format="PNG")
        st.download_button("⬇️ Download Result", data=buf.getvalue(), file_name="colorized.png", mime="image/png")