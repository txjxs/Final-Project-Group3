import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import os
import warnings
from skimage import color, transform

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# --- IMPORT YOUR MODELS ---
from models import ResNetUNet, CVAE, LightweightUNet
from utils import add_noise


# --- PAGE SETUP ---
st.set_page_config(page_title="Restoration AI", layout="wide", page_icon="✨")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'Models')


# ==========================================
# ⚙️ CONFIGURATION: MAP TASKS TO MODELS
# ==========================================
TASK_CONFIG = {
   "Denoising": {
       "class": LightweightUNet,
       "file": "LightweightUNet.pth"
   },
   "Colorization": {
       "class": ResNetUNet,
       "file": "best_resnet_model.pth"
   },
   "Combined (VAE)": {
       "class": CVAE,
       "file": "best_model_VAE.pth"
   }
}




# --- HELPER FUNCTIONS ---
@st.cache_resource
@st.cache_resource
def load_task_model(task_name):
    """Loads the specific model assigned to the selected task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = TASK_CONFIG.get(task_name)
    if not config:
        return None, f"No configuration found for {task_name}"

    model_class = config["class"]
    filename = config["file"]

    # --- 1. Initialize Model Architecture ---
    try:
        if task_name == "Combined (VAE)":
            # Your VAE likely outputs 2 channels (ab) to be merged with L later.
            # If your model was trained to output RGB directly, change '2' to '3'.
            model = model_class(in_channels=1, out_channels=2, latent_dim=128)
        else:
            model = model_class()
    except Exception as e:
        return None, f"Error initializing class {model_class.__name__}: {e}"

    weight_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(weight_path):
        return None, f"⚠️ Weights file missing! Expected: '{filename}' in Models folder."

    # --- 2. Load Weights Safely ---
    try:
        print(f"Loading {task_name} from {weight_path}...")
        checkpoint = torch.load(weight_path, map_location=device)

        # Handle different saving conventions
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # THIS IS THE CASE FOR YOUR FILE (based on the error log)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Fallback: The file is the state_dict itself
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return model, device

    except Exception as e:
        return None, f"Error loading weights: {str(e)}"

def lab_to_rgb(L, ab):
    """
    Convert L*a*b* tensors back to RGB
    """
    # Ensure inputs are on CPU and detached
    L = L.cpu().detach()
    ab = ab.cpu().detach()

    L_denorm = L * 100.0  # [0, 100]
    a_denorm = ab[:, 0:1] * 255.0 - 128.0
    b_denorm = ab[:, 1:2] * 255.0 - 128.0

    lab = torch.cat([L_denorm, a_denorm, b_denorm], dim=1)
    lab_np = lab.permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)

    rgb_list = []
    for i in range(lab_np.shape[0]):
        rgb = color.lab2rgb(lab_np[i])
        rgb = np.clip(rgb, 0, 1)
        rgb_list.append(rgb)

    rgb_np = np.stack(rgb_list, axis=0)
    return torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()


def preprocess_image(image, device):
   w, h = image.size
   new_w = (w // 16) * 16
   new_h = (h // 16) * 16


   if (new_w != w) or (new_h != h):
       image = image.resize((new_w, new_h))


   transform = transforms.Compose([transforms.ToTensor()])
   return transform(image).unsqueeze(0).to(device), image




def tensor_to_img(t):
   return t.squeeze().cpu().detach().permute(1, 2, 0).numpy()


def preprocess_image_for_color(pil_image, transform=None):
    """
    The Master Decolorizer: Converts a PIL RGB image into L and ab tensors.
    Used by both the Dataset (training) and the App (inference).
    """
    # 1. Apply PyTorch transforms
    if transform:
        pil_image = transform(pil_image)

    # 2. Convert to Numpy
    img_np = np.array(pil_image)

    # 3. Convert RGB to Lab
    img_lab = color.rgb2lab(img_np).astype("float32")

    # 4. Normalize to [-1, 1] range
    img_lab[:, :, 0] = (img_lab[:, :, 0] / 50.0) - 1.0  
    img_lab[:, :, 1:] = (img_lab[:, :, 1:] / 128.0) 

    # 5. Convert to Tensor
    img_tensor = torch.from_numpy(img_lab.transpose((2, 0, 1)))

    # 6. Split into Input (L) and Target (ab)
    L = img_tensor[[0], ...] 
    ab = img_tensor[[1, 2], ...]  

    return L, ab

def get_download_link(img_tensor, filename):
   result_array = (tensor_to_img(img_tensor) * 255).astype(np.uint8)
   result_image = Image.fromarray(result_array)
   buf = io.BytesIO()
   result_image.save(buf, format="PNG")
   return buf.getvalue()




# ==========================================
# 🖥️ APPLICATION UI
# ==========================================


# --- Sidebar ---
st.sidebar.title("✨ Restoration AI")
page_mode = st.sidebar.radio("Select Task", list(TASK_CONFIG.keys()))
st.sidebar.markdown("---")


# --- Load Model for Current Page ---
model, device_or_err = load_task_model(page_mode)


if model is None:
   st.error(f"❌ Could not load model for {page_mode}.")
   st.error(device_or_err)
   st.stop()


device = device_or_err


# ==========================================
# PAGE 1: DENOISING
# ==========================================
if page_mode == "Denoising":
   st.title("Denoising Studio")


   # Controls
   mode = st.sidebar.radio("Input Mode", ["Simulate Noise (Demo)", "Real Noisy Image"])
   noise_factor = 0.0
   if mode == "Simulate Noise (Demo)":
       noise_factor = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.3)


   uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])


   if uploaded_file:
       image = Image.open(uploaded_file).convert('RGB')
       input_tensor, resized_img = preprocess_image(image, device)


       # Prepare Input
       if mode == "Simulate Noise (Demo)":
           model_input = add_noise(input_tensor, noise_type='gaussian', factor=noise_factor)
       else:
           model_input = input_tensor


       # AUTOMATIC INFERENCE
       with st.spinner("Processing..."):
           with torch.no_grad():
               output_tensor = model(model_input)


       # Display
       c1, c2, c3 = st.columns(3)
       # Updated to use_container_width
       c1.image(resized_img, caption="Original", use_container_width=True)
       c2.image(tensor_to_img(model_input), caption="Input (Noisy)", use_container_width=True)
       c3.image(tensor_to_img(output_tensor), caption="Result (Clean)", use_container_width=True)


       st.download_button("Download Result", get_download_link(output_tensor, "denoised.png"), "denoised.png",
                          "image/png")


# ==========================================
# PAGE 2: COLORIZATION
# ==========================================
elif page_mode == "Colorization":
    st.title("Colorization Studio")
    
    # Add Saturation Slider
    saturation = st.sidebar.slider("🎨 Saturation Boost", 0.0, 3.0, 1.6, help="Fixes Sepia Effect")

    uploaded_file = st.file_uploader("Upload B&W Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # 1. Load Original High-Res Image
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image_pil)
        
        # 2. Extract Original High-Res L Channel
        # We perform the RGB->Lab conversion here to keep the full resolution L channel
        orig_lab = color.rgb2lab(image_np)
        orig_L = orig_lab[:, :, 0] # Range [0, 100]
        orig_h, orig_w = orig_L.shape
        
        # 3. Prepare Low-Res Input for Model (256x256)
        img_resized = image_pil.resize((256, 256))
        img_resized_np = np.array(img_resized)
        img_resized_lab = color.rgb2lab(img_resized_np).astype("float32")
        
        # Normalize L to [-1, 1]
        img_l_input = (img_resized_lab[:, :, 0] / 50.0) - 1.0
        tensor_input = torch.from_numpy(img_l_input).unsqueeze(0).unsqueeze(0).to(device)

        # AUTOMATIC INFERENCE
        with st.spinner("Colorizing..."):
            with torch.no_grad():
                # Model predicts low-res color (256x256)
                ab_pred = model(tensor_input)
                ab_pred = ab_pred.cpu().numpy()[0] # (2, 256, 256)
                
            # --- POST-PROCESSING (High-Res Fusion) ---
            
            # A. Resize predicted 'ab' to match Original Size
            # transpose to (256, 256, 2) -> resize -> (H, W, 2)
            ab_high_res = transform.resize(
                ab_pred.transpose((1, 2, 0)), 
                (orig_h, orig_w),
                mode='reflect',
                anti_aliasing=True
            )
            
            # B. Apply Saturation Boost
            ab_high_res = ab_high_res * 128.0 * saturation
            
            # C. Combine High-Res L with High-Res Color
            lab_final = np.zeros((orig_h, orig_w, 3))
            lab_final[:, :, 0] = orig_L       # Sharp L from original
            lab_final[:, :, 1:] = ab_high_res # Blurry but vibrant color from AI
            
            # D. Convert to RGB
            with np.errstate(invalid='ignore'):
                final_rgb = color.lab2rgb(lab_final)
                
            # Convert to uint8 for display/download
            final_img_uint8 = (final_rgb * 255).astype(np.uint8)

        # Display
        c1, c2 = st.columns(2)
        
        # Show grayscale version so user knows what the AI sees
        c1.image(image_pil.convert('L'), caption=f"Original Input ({orig_w}x{orig_h})", use_container_width=True)
        c2.image(final_img_uint8, caption=f"Colorized Result (Sat: {saturation}x)", use_container_width=True)

        # st.download_button("Download Result", get_download_link(final_img_uint8, "colorized.png"), "colorized.png",
        #                    "image/png")
# ==========================================
elif page_mode == "Combined (VAE)":
    st.title("Full Restoration (VAE)")

    # --- 1. SIDEBAR CONTROLS ---
    # Slider removed. Just the mode selection.
    vae_mode = st.sidebar.radio("Input Mode", ["Simulate Noise (Demo)", "Real Noisy Image"], key="vae_input_mode")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # --- 2. PREPROCESSING (Exact Match to Training Code) ---
        original_image = Image.open(uploaded_file).convert('RGB')

        # Resize to 128x128
        FIXED_SIZE = (128, 128)
        image = original_image.resize(FIXED_SIZE)

        # Convert to Numpy RGB [0, 1]
        rgb_np = np.array(image).astype(np.float32) / 255.0

        # Convert to L*a*b* using skimage
        lab_gt = color.rgb2lab(rgb_np)

        # Extract L channel and Normalize to [0, 1]
        L_gt = lab_gt[:, :, 0] / 100.0

        # --- 3. NOISE INJECTION ---
        if vae_mode == "Simulate Noise (Demo)":
            # Add fixed Gaussian noise (0.1)
            noise_factor = 0.1
            noise = np.random.randn(*L_gt.shape) * noise_factor
            L_noisy = np.clip(L_gt + noise, 0, 1)

            # Use Noisy L for Model, Clean L for final Reconstruction
            L_for_model = L_noisy
            L_for_recon = L_gt

            input_caption = "Model Input (Simulated Noise)"
            L_display = (L_noisy * 255).astype(np.uint8)

        else:
            # Real Noisy Image: Use uploaded L channel directly
            L_noisy = L_gt
            L_for_model = L_noisy
            L_for_recon = L_noisy

            input_caption = "Model Input (Real Upload)"
            L_display = (L_noisy * 255).astype(np.uint8)

        # --- 4. PREPARE TENSORS ---
        L_input_tensor = torch.from_numpy(L_for_model).unsqueeze(0).unsqueeze(0).float().to(device)
        L_recon_tensor = torch.from_numpy(L_for_recon).unsqueeze(0).unsqueeze(0).float().to(device)

        # --- 5. AUTOMATIC INFERENCE ---
        with st.spinner("Restoring..."):
            with torch.no_grad():
                output = model(L_input_tensor)
                pred_ab = output[0] if isinstance(output, tuple) else output

                # Combine L + Predicted AB -> RGB
                restored_tensor = lab_to_rgb(L_recon_tensor, pred_ab)
                restored_img = tensor_to_img(restored_tensor)

        # --- 6. DISPLAY RESULTS ---
        c1, c2, c3 = st.columns(3)

        c1.image(image, caption="1. Actual Uploaded Image", use_container_width=True)
        c2.image(L_display, caption=f"2. {input_caption}", use_container_width=True)
        c3.image(restored_img, caption="3. Restored Output", use_container_width=True)

        st.download_button("Download Result", get_download_link(restored_tensor, "vae_restored.png"),
                           "vae_restored.png",
                           "image/png")