import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import os


# --- Custom Modules ---
from models import UNet, LightweightUNet
from utils import add_noise

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(SCRIPT_DIR, 'UNet.pth')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'LightweightUNet.pth')

# IMG_SIZE = 128  <-- REMOVED! We don't force this anymore.

# --- Page Setup ---
st.set_page_config(page_title="Restoration AI", layout="wide")
st.title("✨ Image Restoration AI")
st.markdown("Upload an image to see the **U-Net** remove noise/grain in real-time.")


# --- 1. Model Loader ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = UNet()
    model = LightweightUNet()
    try:
        # Load weights (handle CPU/GPU automatically)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        return None, str(e)


model, device = load_model()

if model is None:
    st.error(f"❌ Error loading model: {device}")
    st.stop()

# --- 2. Sidebar Controls ---
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Input Mode", ["Simulate Noise (Demo)", "Fix Real Noisy Image"])

noise_factor = 0.5
if mode == "Simulate Noise (Demo)":
    noise_factor = st.sidebar.slider("Noise Intensity", 0.0, 1.0, 0.5)

# --- 3. Image Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load
    image = Image.open(uploaded_file).convert('RGB')

    # --- DYNAMIC RESIZING LOGIC ---
    # The U-Net has 3 downsample layers, so input must be divisible by 2^3 = 8.
    # We use 16 just to be safe and ensure even padding.
    w, h = image.size
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16

    if (new_w != w) or (new_h != h):
        st.warning(f"⚠️ Resizing image from ({w}x{h}) to ({new_w}x{new_h}) to fit U-Net constraints.")
        image = image.resize((new_w, new_h))

    # Transform (No Resize needed now!)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    # --- 4. Processing ---
    with st.spinner("AI is working..."):

        if mode == "Simulate Noise (Demo)":
            corrupted_tensor = add_noise(input_tensor, noise_type='gaussian', factor=noise_factor)
            model_input = corrupted_tensor
        else:
            model_input = input_tensor
            corrupted_tensor = input_tensor

        with torch.no_grad():
            output_tensor = model(model_input)


    # --- 5. Visualization ---
    def tensor_to_img(t):
        return t.squeeze().cpu().permute(1, 2, 0).numpy()


    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Uploaded")
        st.image(image, caption=f"Original ({new_w}x{new_h})", use_column_width=True)

    with col2:
        st.subheader("2. Model Input")
        st.image(tensor_to_img(model_input), caption="Input to U-Net", use_column_width=True)

    with col3:
        st.subheader("3. AI Result")
        st.image(tensor_to_img(output_tensor), caption="Restored Output", use_column_width=True)

    # --- Download Button ---
    result_array = (tensor_to_img(output_tensor) * 255).astype(np.uint8)
    result_image = Image.fromarray(result_array)

    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="⬇️ Download Result",
        data=byte_im,
        file_name="restored_image.png",
        mime="image/png"
    )
