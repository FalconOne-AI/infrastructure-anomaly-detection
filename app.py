"""
Concrete Anomaly Detection - Enhanced Visual Prototype
Unsupervised autoencoder approach for infrastructure inspection
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import io
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO
import base64

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Concrete Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
    }
    .status-crack {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ConcreteAutoencoder(nn.Module):
    """Convolutional Autoencoder for concrete anomaly detection"""
    
    def __init__(self):
        super(ConcreteAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_INFO = {
    'Bridge Deck': {
        'model_file': 'deck_autoencoder_epoch45_best.pth',
        'balanced_threshold': 0.000163,
        'conservative_threshold': 0.001025,
        'normal_mean': 0.000220,
        'normal_median': 0.000146,
        'crack_mean': 0.000214,
        'crack_median': 0.000180,
        'separation': 1.2,
        'best_epoch': 45,
        'best_loss': 0.000067,
        'accuracy': 59.2,
        'training_images': 11239,
        'color': '#1f77b4'
    },
    'Wall': {
        'model_file': 'wall_autoencoder_epoch41_best.pth',
        'balanced_threshold': 0.000091,
        'conservative_threshold': 0.000180,
        'normal_mean': 0.000076,
        'normal_median': 0.000059,
        'crack_mean': 0.000106,
        'crack_median': 0.000075,
        'separation': 1.4,
        'best_epoch': 41,
        'best_loss': 0.000025,
        'accuracy': 59.5,
        'training_images': 14238,
        'color': '#ff7f0e'
    },
    'Pavement': {
        'model_file': 'pavement_autoencoder_epoch22_best.pth',
        'balanced_threshold': 0.000100,
        'conservative_threshold': 0.000200,
        'normal_mean': 0.000085,
        'normal_median': 0.000070,
        'crack_mean': 0.000110,
        'crack_median': 0.000090,
        'separation': 1.3,
        'best_epoch': 22,
        'best_loss': 0.000071,
        'accuracy': 59.0,
        'training_images': 15621,
        'color': '#2ca02c'
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model(substrate_type):
    """Load pre-trained autoencoder model from Google Drive"""
    import gdown
    
    # Google Drive File IDs
    GDRIVE_MODELS = {
        'Bridge Deck': '1gEH61QdTh7vfnDpl0ZaP2ryG1I_NbalP',
        'Wall': '1mNH_qP6egf7355KXZGZG3jeD2S9a91hM',
        'Pavement': '1YtW1FHLjOsTOtEqTZz4d3e0p3KpezC-N'
    }
    
    model = ConcreteAutoencoder()
    
    try:
        cache_dir = Path('.model_cache')
        cache_dir.mkdir(exist_ok=True)
        
        model_filename = f"{substrate_type.lower().replace(' ', '_')}_model.pth"
        model_path = cache_dir / model_filename
        
        if not model_path.exists():
            file_id = GDRIVE_MODELS[substrate_type]
            gdrive_url = f'https://drive.google.com/uc?id={file_id}'
            
            with st.spinner(f'⏳ Downloading {substrate_type} model from Google Drive...'):
                gdown.download(gdrive_url, str(model_path), quiet=False)
        
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model input"""
    image = image.resize(target_size, Image.LANCZOS)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor

def calculate_reconstruction_error(model, image_tensor):
    """Calculate reconstruction error and pixel-wise error map"""
    with torch.no_grad():
        reconstruction = model(image_tensor)
        
        # Overall MSE
        mse = torch.mean((image_tensor - reconstruction) ** 2).item()
        
        # Pixel-wise error for heatmap
        pixel_error = torch.mean((image_tensor - reconstruction) ** 2, dim=1).squeeze().numpy()
        
    return mse, reconstruction, pixel_error

def tensor_to_image(tensor):
    """Convert tensor back to PIL Image"""
    img_array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def create_heatmap(error_map, colormap='hot'):
    """Create heatmap visualization of reconstruction error"""
    # Normalize error map
    error_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(error_norm)
    heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)
    
    return Image.fromarray(heatmap_rgb)

def create_overlay(original_image, error_map, alpha=0.5, colormap='hot'):
    """Create overlay of heatmap on original image"""
    # Resize original to match error map size
    orig_array = np.array(original_image.resize((error_map.shape[1], error_map.shape[0])))
    
    # Normalize error map
    error_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
    
    # Create heatmap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(error_norm)
    heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)
    
    # Blend
    overlay = cv2.addWeighted(orig_array, 1-alpha, heatmap_rgb, alpha, 0)
    
    return Image.fromarray(overlay)

def create_download_link(img, filename):
    """Create download link for image"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">📥 Download {filename}</a>'
    return href

def process_single_image(image, model, threshold, config):
    """Process a single image and return results"""
    img_tensor = preprocess_image(image)
    error_score, reconstruction, pixel_error = calculate_reconstruction_error(model, img_tensor)
    
    is_anomaly = error_score > threshold
    recon_image = tensor_to_image(reconstruction)
    heatmap = create_heatmap(pixel_error)
    overlay = create_overlay(image, pixel_error)
    
    return {
        'original': image,
        'reconstruction': recon_image,
        'heatmap': heatmap,
        'overlay': overlay,
        'error_score': error_score,
        'is_anomaly': is_anomaly,
        'pixel_error': pixel_error
    }

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🔍 Concrete Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised Autoencoder - Visual Prototype</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Substrate selection
        substrate_type = st.selectbox(
            "Infrastructure Type",
            list(MODEL_INFO.keys()),
            help="Select the substrate to analyze"
        )
        
        config = MODEL_INFO[substrate_type]
        
        st.markdown("---")
        
        # Performance metrics with color coding
        st.markdown("### 📊 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; font-weight: 700; color: {config['color']};">{config['separation']}x</div>
                <div style="font-size: 0.9rem; color: #666;">Separation Ratio</div>
                <div style="font-size: 0.8rem; color: #999;">Target: 5-10x</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; font-weight: 700; color: {config['color']};">{config['accuracy']}%</div>
                <div style="font-size: 0.9rem; color: #666;">Accuracy</div>
                <div style="font-size: 0.8rem; color: #999;">Target: 85-95%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption(f"**Training Loss:** {config['best_loss']:.6f}")
        st.caption(f"**Best Epoch:** {config['best_epoch']}")
        st.caption(f"**Training Images:** {config['training_images']:,}")
        
        st.markdown("---")
        
        # Threshold settings with better UI
        st.markdown("### 🎚️ Detection Threshold")
        
        threshold_mode = st.radio(
            "Mode",
            ["Balanced", "Conservative", "Custom"],
            help="Balanced: Equal errors | Conservative: Fewer false alarms",
            horizontal=True
        )
        
        if threshold_mode == "Custom":
            threshold = st.slider(
                "Custom Threshold",
                min_value=0.0,
                max_value=config['conservative_threshold'] * 2,
                value=config['balanced_threshold'],
                step=0.000001,
                format="%.6f"
            )
        else:
            threshold = config[f"{threshold_mode.lower()}_threshold"]
            st.info(f"**Threshold:** {threshold:.6f}")
        
        st.markdown("---")
        
        # Visualization settings
        st.markdown("### 🎨 Visualization")
        
        heatmap_colormap = st.selectbox(
            "Heatmap Color",
            ["hot", "jet", "viridis", "plasma", "inferno"],
            index=0
        )
        
        overlay_alpha = st.slider(
            "Overlay Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        st.markdown("---")
        
        # Score reference
        st.markdown("### 📈 Score Reference")
        st.markdown(f"""
        <div style="font-size: 0.85rem;">
        <b>Normal Range:</b><br>
        Mean: <code>{config['normal_mean']:.6f}</code><br>
        Median: <code>{config['normal_median']:.6f}</code><br><br>
        <b>Crack Range:</b><br>
        Mean: <code>{config['crack_mean']:.6f}</code><br>
        Median: <code>{config['crack_median']:.6f}</code>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    st.markdown("### 📤 Upload Images")
    
    # Batch upload with size limit
    st.caption("💡 Upload multiple images (up to 200MB total)")
    uploaded_files = st.file_uploader(
        "Choose concrete images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple images for batch analysis. Maximum total size: 200MB"
    )
    
    # Clear All button (appears after files are uploaded or results exist)
    if 'results' in st.session_state or (uploaded_files and len(uploaded_files) > 0):
        if st.button("🗑️ Clear All", type="secondary", help="Remove all uploaded files and clear analysis"):
            # Clear session state
            if 'results' in st.session_state:
                del st.session_state['results']
            if 'config' in st.session_state:
                del st.session_state['config']
            if 'heatmap_colormap' in st.session_state:
                del st.session_state['heatmap_colormap']
            if 'overlay_alpha' in st.session_state:
                del st.session_state['overlay_alpha']
            st.success("✅ Cleared all files and analysis!")
            # Use rerun with compatibility for older Streamlit versions
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
    
    # Check total file size
    if uploaded_files:
        total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)  # Convert to MB
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("📁 Files Uploaded", len(uploaded_files))
        with col_info2:
            st.metric("💾 Total Size", f"{total_size:.1f} MB")
        with col_info3:
            size_status = "🟢 OK" if total_size <= 200 else "🔴 Too Large"
            st.metric("📊 Status", size_status)
        
        if total_size > 200:
            st.error(f"⚠️ Total file size ({total_size:.1f} MB) exceeds 200 MB limit. Please remove some files.")
            st.stop()
        
        st.success(f"✅ Ready to process {len(uploaded_files)} images ({total_size:.1f} MB)")
    
    if uploaded_files:
        # Load model
        with st.spinner(f"⏳ Loading {substrate_type} model..."):
            model = load_model(substrate_type)
        
        if model is not None:
            # Process button
            if st.button("🔍 Analyze All Images", type="primary", use_container_width=True):
                # Process all images
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing image {idx+1}/{len(uploaded_files)}...")
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    result = process_single_image(image, model, threshold, config)
                    result['filename'] = uploaded_file.name
                    results.append(result)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text(f"✅ Processed {len(uploaded_files)} images!")
                st.session_state['results'] = results
                st.session_state['config'] = config
                st.session_state['heatmap_colormap'] = heatmap_colormap
                st.session_state['overlay_alpha'] = overlay_alpha
    
    # Display results
    if 'results' in st.session_state:
        results = st.session_state['results']
        config = st.session_state['config']
        
        # Update threshold in real-time - recalculate anomaly status
        current_threshold = threshold
        for result in results:
            result['is_anomaly'] = result['error_score'] > current_threshold
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Summary statistics (recalculated with current threshold)
        total_images = len(results)
        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        normal_count = total_images - anomaly_count
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", total_images)
        with col2:
            st.metric("🟢 Normal", normal_count)
        with col3:
            st.metric("🔴 Anomalies", anomaly_count)
        with col4:
            avg_score = np.mean([r['error_score'] for r in results])
            st.metric("Avg Score", f"{avg_score:.6f}")
        
        st.markdown("---")
        
        # Individual results
        for idx, result in enumerate(results):
            st.markdown(f"### Image {idx+1}: {result['filename']}")
            
            # Status badge
            if result['is_anomaly']:
                st.markdown('<div class="status-crack">🔴 CRACK DETECTED</div>', unsafe_allow_html=True)
                confidence = "Very Low" if config['separation'] < 1.5 else "Low"
            else:
                st.markdown('<div class="status-normal">🟢 NORMAL CONCRETE</div>', unsafe_allow_html=True)
                confidence = "Low"
            
            st.markdown("")
            
            # Metrics row
            met_col1, met_col2, met_col3 = st.columns(3)
            with met_col1:
                st.metric("Reconstruction Error", f"{result['error_score']:.6f}")
            with met_col2:
                st.metric("Threshold", f"{current_threshold:.6f}")
            with met_col3:
                st.metric("Confidence", confidence)
            
            # Tabbed visualization
            tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Comparison", "🔥 Heatmap", "🎯 Overlay", "📈 Analysis"])
            
            with tab1:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(result['original'], caption="Original Image", use_container_width=True)
                    st.markdown(create_download_link(result['original'], f"{result['filename']}_original.png"), unsafe_allow_html=True)
                with col_b:
                    st.image(result['reconstruction'], caption="Reconstruction", use_container_width=True)
                    st.markdown(create_download_link(result['reconstruction'], f"{result['filename']}_reconstruction.png"), unsafe_allow_html=True)
            
            with tab2:
                st.image(result['heatmap'], caption="Error Heatmap (Red = High Error)", use_container_width=True)
                st.markdown(create_download_link(result['heatmap'], f"{result['filename']}_heatmap.png"), unsafe_allow_html=True)
                st.caption("Heatmap shows pixel-wise reconstruction errors. Brighter areas indicate higher anomaly likelihood.")
            
            with tab3:
                st.image(result['overlay'], caption="Error Overlay on Original", use_container_width=True)
                st.markdown(create_download_link(result['overlay'], f"{result['filename']}_overlay.png"), unsafe_allow_html=True)
                st.caption("Overlay highlights anomalous regions directly on the original image.")
            
            with tab4:
                # Score comparison chart
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Normal<br>Range', 'This<br>Image', 'Crack<br>Range'],
                    y=[config['normal_mean'], result['error_score'], config['crack_mean']],
                    marker_color=['#28a745', '#17a2b8' if not result['is_anomaly'] else '#dc3545', '#dc3545'],
                    text=[f"{config['normal_mean']:.6f}", f"{result['error_score']:.6f}", f"{config['crack_mean']:.6f}"],
                    textposition='auto',
                ))
                
                fig.add_hline(y=current_threshold, line_dash="dash", line_color="orange",
                             annotation_text=f"Threshold: {current_threshold:.6f}")
                
                fig.update_layout(
                    title="Score Comparison",
                    yaxis_title="Reconstruction Error",
                    showlegend=False,
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Warning
                st.warning(f"""
                ⚠️ **Note:** With {config['separation']}x separation, predictions have approximately 
                {config['accuracy']}% reliability. This prototype demonstrates technical limitations 
                of unsupervised crack detection.
                """)
            
            st.markdown("---")
    
    # About section
    with st.expander("📖 About This Prototype", expanded=False):
        st.markdown(f"""
        ## Unsupervised Anomaly Detection Results
        
        ### Performance Summary
        
        | Substrate | Separation | Accuracy | Training Loss | Status |
        |-----------|------------|----------|---------------|--------|
        | Bridge Deck | 1.2x | 59% | 0.000067 | ❌ Insufficient |
        | Wall | 1.4x | 60% | 0.000025 | ❌ Insufficient |
        | Pavement | 1.3x | 59% | 0.000071 | ❌ Insufficient |
        
        **Target:** 5-10x separation, 85-95% accuracy
        
        ### What This Means
        
        This prototype uses an **unsupervised autoencoder** approach that:
        
        ✅ **Successfully learned** substrate texture patterns (excellent loss convergence)
        
        ❌ **Cannot reliably distinguish** cracks from normal texture (insufficient separation)
        
        ### Key Discovery: Data Contamination
        
        Validation revealed that "uncracked" training folders contained images with visible cracks. 
        This data contamination caused the model to learn cracks as "normal," preventing effective detection.
        
        ### Why Low Separation Occurs
        
        Even with clean data, thin linear cracks create similar reconstruction errors to normal 
        texture variations (aggregate, shadows, joints), making them indistinguishable with this approach.
        
        ### Next Steps
        
        - **Data Cleaning:** Remove contaminated training examples
        - **Thermal Imaging:** Explore temperature-based detection
        - **Supervised Classification:** Train models that explicitly learn crack features (85-95% accuracy target)
        
        ---
        
        **⚠️ Use Case:** Research prototype demonstrating technical challenges. Not for production use.
        """)
    
    # Technical details
    with st.expander("🔧 Technical Details", expanded=False):
        st.markdown("""
        ### Model Architecture
        
        **Type:** Convolutional Autoencoder  
        **Parameters:** ~187,000  
        **Input Size:** 256×256×3 (RGB)
        
        **Encoder:**
        - Conv2d(3→32) + ReLU + BatchNorm
        - Conv2d(32→64) + ReLU + BatchNorm
        - Conv2d(64→128) + ReLU + BatchNorm
        
        **Decoder:**
        - ConvTranspose2d(128→64) + ReLU + BatchNorm
        - ConvTranspose2d(64→32) + ReLU + BatchNorm
        - ConvTranspose2d(32→3) + Sigmoid
        
        ### Training Configuration
        
        - **Optimizer:** Adam
        - **Learning Rate:** 0.001
        - **Batch Size:** 32
        - **Epochs:** 50 (early stopping)
        - **Loss:** MSE (Mean Squared Error)
        - **Hardware:** NVIDIA A100 GPU
        
        ### Visualization Methods
        
        - **Heatmap:** Pixel-wise MSE between original and reconstruction
        - **Overlay:** Weighted blend of original image and error heatmap
        - **Colormap:** Configurable (hot, jet, viridis, plasma, inferno)
        
        ### Dataset
        
        **Source:** SDNET2018 Concrete Crack Detection  
        **Strategy:** Unsupervised (normal images only for training)  
        **Issue:** Data contamination identified during validation
        """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
