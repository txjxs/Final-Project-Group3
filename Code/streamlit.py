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
from models import ResNetUNet, CVAE #LightweightUNet
from utils import add_noise


# --- PAGE SETUP ---
st.set_page_config(page_title="Restoration AI", layout="wide", page_icon="✨")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'Models')


# ==========================================
# ⚙️ CONFIGURATION: MAP TASKS TO MODELS
# ==========================================
TASK_CONFIG = {
   # "Denoising": {
   #     "class": LightweightUNet,
   #     "file": "LightweightUNet.pth"
   # },
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
def load_task_model(task_name):
   """Loads the specific model assigned to the selected task."""
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


   config = TASK_CONFIG.get(task_name)
   if not config:
       return None, f"No configuration found for {task_name}"


   model_class = config["class"]
   filename = config["file"]


   try:
      if config == "Combined (VAE)":
         model = model_class(1,2,128)
      else:
       model = model_class()
   except Exception as e:
       return None, f"Error initializing class {model_class.__name__}: {e}"


   weight_path = os.path.join(MODEL_DIR, filename)
   if not os.path.exists(weight_path):
       return None, f"⚠️ Weights file missing! Expected: '{filename}' in Models folder."


   try:
        print(f"1. Loading checkpoint from {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device)
        
        # Check for different common key names used to save weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("2. Found 'model_state_dict' key.")
            model.load_state_dict(checkpoint['model_state_dict'])
            
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("2. Found 'state_dict' key.")
            model.load_state_dict(checkpoint['state_dict'])
            
        else:
            print("2. Assuming file is raw state_dict.")
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        return model, device
   except Exception as e:
       return None, f"Error loading weights: {str(e)}"




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

    uploaded_file = st.file_uploader("Upload Damaged Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # 1. Load Original (Reference)
        original_image = Image.open(uploaded_file).convert('RGB')
        
        # 2. Preprocess: Resize to 128x128 and Convert to Grayscale
        FIXED_SIZE = (128, 128)
        img_for_model = original_image.resize(FIXED_SIZE).convert('L')
        img_for_display = original_image.resize(FIXED_SIZE).convert('RGB')

        # 3. Prepare Tensor: Scale to [-1, 1] range
        # Note: ToTensor() gives [0, 1]. We subtract 0.5 and divide by 0.5 to get [-1, 1].
        img_array = np.array(img_for_model) / 255.0 
        img_array = (img_array - 0.5) / 0.5
        input_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0).to(device)

        # 4. Display Columns
        c1, c2 = st.columns(2)
        c1.image(img_for_display, caption="Original Input (Resized)", use_container_width=True)

        # 5. AUTOMATIC INFERENCE
       # 5. AUTOMATIC INFERENCE
        with st.spinner("Restoring..."):
            with torch.no_grad():
                output = model(input_tensor)
                ab_channels = output[0] if isinstance(output, tuple) else output
                
                # Post-process: Concatenate Input L (grayscale) + Predicted AB
                L = input_tensor.cpu().detach()
                ab = ab_channels.cpu().detach()
                

                # We shift it to [-0.5, 0.5] so it can be Green/Blue too.
                if ab.min() >= 0 and ab.max() <= 1.0:
                     ab = ab - 0.5
                # --- FIX END ---

                # Concatenate to (1, 3, 128, 128)
                lab_image = torch.cat([L, ab], dim=1)
                
                # Convert Tensor -> Numpy -> RGB for Streamlit
                lab_np = lab_image[0].permute(1, 2, 0).numpy()
                
                # Un-normalize 
                # L was scaled to [-1, 1], map back to [0, 100]
                lab_np[:,:,0] = (lab_np[:,:,0] + 1.0) * 50.0  
                
                # ab was shifted to [-0.5, 0.5], map to [-128, 128]
                # We multiply by 255 to cover the full spectrum
                lab_np[:,:,1:] = lab_np[:,:,1:] * 255.0       
                
                # Convert Lab to RGB
                rgb_image = color.lab2rgb(lab_np.astype("float64"))
                
                # Display Result
                c2.image(rgb_image, caption="Restored Output", use_container_width=True)
