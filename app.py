"""
Infrastructure Anomaly Detection - Falcon AI Solutions
=======================================================
Dual-model demonstration featuring:
- Visual Concrete Model (SDNET2018)
- Thermal Infrared Model (SDNET2021)

Powered by Falcon AI Solutions
Transforming Business With AI

Version: 2.0 - Dual Model Edition
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import zipfile
from typing import Tuple, List, Optional
import gdown

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Falcon AI Solutions | Anomaly Detection",
    page_icon="https://www.falconaisolutions.com/lovable-uploads/f28589dc-87f2-42d0-9d82-a577d7ce34ab.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GOOGLE DRIVE MODEL CONFIGURATION
# ============================================================================

MODELS_CONFIG = {
    "concrete": {
        "name": "Visual Concrete Model",
        "description": "Trained on SDNET2018 visual concrete images",
        "gdrive_id": "18A7L8toYWNsVfu4VGWxG6Clj8xYmQQjH",
        "filename": "concrete_autoencoder_week1.pth",
        "dataset": "SDNET2018",
        "image_types": "Visual RGB concrete images",
        "best_for": "Surface cracks, spalling, visual defects",
        "training_images": "~28,000 images"
    },
    "thermal": {
        "name": "Thermal Infrared Model",
        "description": "Trained on SDNET2021 thermal bridge images",
        "gdrive_id": "1VE3ZR12aRD4ojtvbdR2uqRNxRA90deCl",
        "filename": "thermal_autoencoder_week2.pth",
        "dataset": "SDNET2021",
        "image_types": "Thermal infrared images",
        "best_for": "Subsurface defects, delamination, moisture",
        "training_images": "~5,000+ thermal images"
    }
}

# ============================================================================
# FALCON AI SOLUTIONS CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    
    .header-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    .header-tagline {
        font-size: 1.1rem;
        margin-top: 1rem;
        opacity: 0.9;
        font-style: italic;
        border-top: 1px solid rgba(255,255,255,0.3);
        padding-top: 0.75rem;
    }
    
    .model-selector-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #3b82f6;
        margin: 1rem 0;
    }
    
    .model-info-card {
        background: white;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        border-left: 4px solid #2563eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.75rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border: 1px solid rgba(37, 99, 235, 0.1);
        text-align: center;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #1e3a8a;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-normal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 24px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 24px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.3);
    }
    
    .status-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 24px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.3);
    }
    
    .info-box {
        background-color: #dbeafe;
        border-left: 5px solid #2563eb;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 6px;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.875rem;
        border-radius: 10px;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(30, 58, 138, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder with BatchNorm (NO Dropout).
    Layer order: Conv2d → ReLU → BatchNorm2d
    """
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder: Conv → ReLU → BatchNorm pattern
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),      # 0
            nn.ReLU(),                                                  # 1
            nn.BatchNorm2d(32),                                         # 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),     # 3
            nn.ReLU(),                                                  # 4
            nn.BatchNorm2d(64),                                         # 5
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),    # 6
            nn.ReLU(),                                                  # 7
            nn.BatchNorm2d(128),                                        # 8
        )
        
        # Decoder: ConvTranspose → ReLU → BatchNorm pattern
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 0
            nn.ReLU(),                                                                           # 1
            nn.BatchNorm2d(64),                                                                  # 2
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # 3
            nn.ReLU(),                                                                           # 4
            nn.BatchNorm2d(32),                                                                  # 5
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # 6
            nn.Sigmoid(),                                                                        # 7
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def download_model_from_gdrive(gdrive_id: str, filename: str) -> str:
    """Download model from Google Drive."""
    try:
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        output_path = models_dir / filename
        
        if output_path.exists():
            return str(output_path)
        
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        return str(output_path)
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return None

@st.cache_resource
def load_model(model_type: str) -> Optional[ConvAutoencoder]:
    """Load model from Google Drive."""
    if model_type not in MODELS_CONFIG:
        st.error(f"Unknown model type: {model_type}")
        return None
    
    config = MODELS_CONFIG[model_type]
    
    with st.spinner(f"Loading {config['name']}..."):
        try:
            model_path = download_model_from_gdrive(config["gdrive_id"], config["filename"])
            
            if model_path is None:
                return None
            
            model = ConvAutoencoder()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            
            st.success(f"✓ {config['name']} loaded successfully!")
            return model
            
        except Exception as e:
            st.error(f"Error loading {config['name']}: {str(e)}")
            return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def calculate_reconstruction_error(original: torch.Tensor, reconstructed: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        error = torch.mean((original - reconstructed) ** 2, dim=1)
        return error.squeeze().cpu().numpy()

def generate_anomaly_map(error: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    error_norm = (error - error.min()) / (error.max() - error.min() + 1e-8)
    cmap = plt.cm.get_cmap('jet')
    heatmap = cmap(error_norm)[:, :, :3]
    return (heatmap * 255).astype(np.uint8)

def create_overlay(original: Image.Image, anomaly_map: np.ndarray, alpha: float = 0.5) -> Image.Image:
    anomaly_pil = Image.fromarray(anomaly_map).resize(original.size, Image.Resampling.BILINEAR)
    return Image.blend(original, anomaly_pil, alpha=alpha)

def get_severity_level(score: float) -> Tuple[str, str]:
    if score < 0.3:
        return "NORMAL", "status-normal"
    elif score < 0.6:
        return "MINOR ANOMALY", "status-warning"
    else:
        return "CRITICAL ANOMALY", "status-danger"

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    img = tensor.squeeze().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def create_download_link(images: List[Image.Image], scores: List[float], filename: str = "results.zip") -> str:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for idx, (img, score) in enumerate(zip(images, scores)):
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            zip_file.writestr(f"result_{idx+1}_score_{score:.3f}.png", img_buffer.getvalue())
    zip_buffer.seek(0)
    b64 = base64.b64encode(zip_buffer.read()).decode()
    return f'<a href="data:application/zip;base64,{b64}" download="{filename}" style="color: #2563eb; font-weight: 600;">📥 Download All Results</a>'

# ============================================================================
# SESSION STATE
# ============================================================================

if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "concrete"

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🦅 Falcon AI Solutions</div>
        <div class="header-subtitle">Infrastructure Anomaly Detection | Dual Model Demo</div>
        <div class="header-tagline">Transforming Business With AI</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.5rem;">
            <p style="color: #1e3a8a; font-weight: 600; margin: 0;">Powered by</p>
            <p style="color: #2563eb; font-size: 1.2rem; font-weight: 700; margin: 0.25rem 0;">Falcon AI Solutions</p>
            <p style="color: #64748b; font-size: 0.85rem; margin: 0;">www.falconaisolutions.com</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Selection
        st.markdown('<div class="model-selector-box">', unsafe_allow_html=True)
        st.subheader("🔄 Select Model")
        
        model_choice = st.radio(
            "Choose detection model:",
            ["Visual Concrete", "Thermal Infrared"],
            help="Select which trained model to use for anomaly detection"
        )
        
        model_type = "concrete" if model_choice == "Visual Concrete" else "thermal"
        st.session_state.selected_model = model_type
        
        config = MODELS_CONFIG[model_type]
        
        st.markdown(f"""
        <div class="model-info-card">
            <strong>{config['name']}</strong><br>
            <small style="color: #64748b;">
            📊 Dataset: {config['dataset']}<br>
            📷 Type: {config['image_types']}<br>
            🎯 Best for: {config['best_for']}<br>
            📈 Training: {config['training_images']}
            </small>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Load model
        model = load_model(model_type)
        
        if model is None:
            st.error("⚠️ Model loading failed. Please check configuration.")
            st.stop()
        
        st.divider()
        
        # Detection settings
        st.subheader("Detection Settings")
        threshold = st.slider(
            "Anomaly Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher threshold = fewer detections (less sensitive)"
        )
        st.session_state.threshold = threshold
        
        overlay_alpha = st.slider(
            "Overlay Transparency",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        st.divider()
        
        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        This demo showcases **dual-model AI capabilities** for infrastructure inspection:
        
        **Visual Concrete Model:**
        - Detects surface-level defects
        - Trained on 28,000+ images
        - Identifies cracks, spalling
        
        **Thermal Infrared Model:**
        - Detects subsurface issues
        - Thermal imaging analysis
        - Finds delamination, moisture
        
        **Powered by Falcon AI Solutions**
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Images")
        
        upload_mode = st.radio(
            "Upload Mode",
            ["Single Image", "Batch Upload"],
            horizontal=True
        )
        
        if upload_mode == "Single Image":
            uploaded_file = st.file_uploader(
                f"Choose an image for {MODELS_CONFIG[model_type]['name']}",
                type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
                help=f"Upload {MODELS_CONFIG[model_type]['image_types']}"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)
                
                if st.button("🔍 Detect Anomalies", type="primary"):
                    with st.spinner("AI processing..."):
                        img_tensor = preprocess_image(image)
                        with torch.no_grad():
                            reconstruction = model(img_tensor)
                        error = calculate_reconstruction_error(img_tensor, reconstruction)
                        anomaly_score = float(error.mean())
                        anomaly_map = generate_anomaly_map(error, threshold)
                        overlay = create_overlay(image, anomaly_map, overlay_alpha)
                        reconstructed_img = tensor_to_image(reconstruction)
                        
                        st.session_state.processed_results = [{
                            'original': image,
                            'reconstructed': reconstructed_img,
                            'anomaly_map': Image.fromarray(anomaly_map),
                            'overlay': overlay,
                            'score': anomaly_score
                        }]
        
        else:  # Batch upload
            uploaded_files = st.file_uploader(
                f"Choose multiple images for {MODELS_CONFIG[model_type]['name']}",
                type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.success(f"✓ {len(uploaded_files)} images uploaded")
                
                if st.button("🔍 Process Batch", type="primary"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        progress = (idx + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        
                        image = Image.open(uploaded_file)
                        img_tensor = preprocess_image(image)
                        with torch.no_grad():
                            reconstruction = model(img_tensor)
                        error = calculate_reconstruction_error(img_tensor, reconstruction)
                        anomaly_score = float(error.mean())
                        anomaly_map = generate_anomaly_map(error, threshold)
                        overlay = create_overlay(image, anomaly_map, overlay_alpha)
                        reconstructed_img = tensor_to_image(reconstruction)
                        
                        results.append({
                            'original': image,
                            'reconstructed': reconstructed_img,
                            'anomaly_map': Image.fromarray(anomaly_map),
                            'overlay': overlay,
                            'score': anomaly_score,
                            'filename': uploaded_file.name
                        })
                    
                    st.session_state.processed_results = results
                    st.success(f"✓ Processed {len(results)} images")
    
    with col2:
        st.header("📊 Results")
        
        if st.session_state.processed_results:
            if len(st.session_state.processed_results) > 1:
                st.subheader("Batch Summary")
                scores = [r['score'] for r in st.session_state.processed_results]
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Images Processed", len(scores))
                with metric_col2:
                    st.metric("Avg Anomaly Score", f"{np.mean(scores):.3f}")
                with metric_col3:
                    anomalies = sum(1 for s in scores if s > threshold)
                    st.metric("Anomalies Detected", anomalies)
                st.divider()
            
            for idx, result in enumerate(st.session_state.processed_results):
                if len(st.session_state.processed_results) > 1:
                    st.subheader(f"Image {idx + 1}: {result.get('filename', 'Unknown')}")
                
                score = result['score']
                severity, severity_class = get_severity_level(score)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Anomaly Score</div>
                    <div class="metric-value">{score:.4f}</div>
                    <div class="{severity_class}">{severity}</div>
                </div>
                """, unsafe_allow_html=True)
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📸 Comparison",
                    "🔥 Heatmap",
                    "🎯 Overlay",
                    "📈 Analysis"
                ])
                
                with tab1:
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        st.image(result['original'], caption="Original", use_container_width=True)
                    with comp_col2:
                        st.image(result['reconstructed'], caption="Reconstruction", use_container_width=True)
                
                with tab2:
                    st.image(result['anomaly_map'], caption="Anomaly Heatmap", use_container_width=True)
                    st.caption("🔵 Blue = Normal | 🔴 Red = High Anomaly")
                
                with tab3:
                    st.image(result['overlay'], caption="Anomaly Overlay", use_container_width=True)
                
                with tab4:
                    st.markdown(f"""
                    **AI Analysis:**
                    - **Model Used:** {MODELS_CONFIG[model_type]['name']}
                    - **Dataset:** {MODELS_CONFIG[model_type]['dataset']}
                    - **Score:** {score:.4f}
                    - **Classification:** {severity}
                    - **Recommendation:** {'Manual inspection recommended' if score > threshold else 'No immediate action required'}
                    
                    **Technical Details:**
                    - Image size: {result['original'].size}
                    - Detection threshold: {threshold:.2f}
                    - Powered by: Falcon AI Solutions
                    """)
                
                st.divider()
            
            if st.session_state.processed_results:
                st.subheader("💾 Export Results")
                overlays = [r['overlay'] for r in st.session_state.processed_results]
                scores = [r['score'] for r in st.session_state.processed_results]
                download_link = create_download_link(overlays, scores, "falcon_ai_results.zip")
                st.markdown(download_link, unsafe_allow_html=True)
        
        else:
            st.info("👈 Upload images and click 'Detect Anomalies' to see AI-powered results")
            st.markdown("""
            <div class="info-box">
                <strong>Expected Output:</strong>
                <ul>
                    <li>AI-powered anomaly detection</li>
                    <li>Color-coded heatmaps</li>
                    <li>Quantified risk scores</li>
                    <li>Actionable recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem 0;">
        <strong style="color: #1e3a8a;">Falcon AI Solutions</strong><br>
        <span style="color: #2563eb; font-weight: 600;">Transforming Business With AI</span><br>
        Infrastructure Anomaly Detection | Dual Model Demo<br>
        <a href="https://www.falconaisolutions.com" target="_blank" style="color: #2563eb;">www.falconaisolutions.com</a><br>
        <small>Powered by Machine Learning</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
