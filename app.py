"""
Concrete Anomaly Detection - Visual Prototype
Unsupervised autoencoder approach for infrastructure inspection
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Concrete Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ConcreteAutoencoder(nn.Module):
    """Convolutional Autoencoder for concrete anomaly detection"""
    
    def __init__(self):
        super(ConcreteAutoencoder, self).__init__()
        
        # Encoder - matches your trained model architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),      # Layer 0
            nn.ReLU(),                                                  # Layer 1
            nn.BatchNorm2d(32),                                        # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # Layer 3
            nn.ReLU(),                                                  # Layer 4
            nn.BatchNorm2d(64),                                        # Layer 5
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # Layer 6
            nn.ReLU(),                                                  # Layer 7
            nn.BatchNorm2d(128)                                        # Layer 8
        )
        
        # Decoder - matches your trained model architecture
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Layer 0
            nn.ReLU(),                                                                            # Layer 1
            nn.BatchNorm2d(64),                                                                  # Layer 2
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # Layer 3
            nn.ReLU(),                                                                            # Layer 4
            nn.BatchNorm2d(32),                                                                  # Layer 5
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # Layer 6
            nn.Sigmoid()                                                                          # Layer 7
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
        'training_images': 11239
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
        'training_images': 14238
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
        'training_images': 15621
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model(substrate_type):
    """Load pre-trained autoencoder model from Google Drive"""
    import gdown
    
    # Google Drive File IDs - Configured for deployment
    GDRIVE_MODELS = {
        'Bridge Deck': '1gEH61QdTh7vfnDpl0ZaP2ryG1I_NbalP',
        'Wall': '1mNH_qP6egf7355KXZGZG3jeD2S9a91hM',
        'Pavement': '1YtW1FHLjOsTOtEqTZz4d3e0p3KpezC-N'
    }
    
    model = ConcreteAutoencoder()
    
    try:
        # Create cache directory
        cache_dir = Path('.model_cache')
        cache_dir.mkdir(exist_ok=True)
        
        # Model cache path
        model_filename = f"{substrate_type.lower().replace(' ', '_')}_model.pth"
        model_path = cache_dir / model_filename
        
        # Download from Google Drive if not cached
        if not model_path.exists():
            file_id = GDRIVE_MODELS[substrate_type]
            gdrive_url = f'https://drive.google.com/uc?id={file_id}'
            
            with st.spinner(f'Downloading {substrate_type} model from Google Drive...'):
                gdown.download(gdrive_url, str(model_path), quiet=False)
        
        # Load model
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Please check that Google Drive file IDs are correct and files are publicly accessible")
        return None

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model input"""
    # Resize
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to tensor
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor

def calculate_reconstruction_error(model, image_tensor):
    """Calculate reconstruction error (anomaly score)"""
    with torch.no_grad():
        reconstruction = model(image_tensor)
        mse = torch.mean((image_tensor - reconstruction) ** 2).item()
    return mse, reconstruction

def tensor_to_image(tensor):
    """Convert tensor back to PIL Image"""
    img_array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🔍 Concrete Anomaly Detection - Visual Prototype")
    st.markdown("*Unsupervised autoencoder approach for infrastructure inspection*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Substrate selection
        substrate_type = st.selectbox(
            "Select Substrate Type",
            list(MODEL_INFO.keys()),
            help="Choose the infrastructure type to analyze"
        )
        
        config = MODEL_INFO[substrate_type]
        
        st.divider()
        
        # Model performance metrics
        st.subheader("📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Separation Ratio", f"{config['separation']}x", 
                     delta="Target: 5-10x", delta_color="inverse")
        with col2:
            st.metric("Accuracy", f"{config['accuracy']}%",
                     delta="Target: 85-95%", delta_color="inverse")
        
        st.caption(f"**Training Loss:** {config['best_loss']:.6f}")
        st.caption(f"**Best Epoch:** {config['best_epoch']}")
        st.caption(f"**Training Images:** {config['training_images']:,}")
        
        st.divider()
        
        # Threshold settings
        st.subheader("🎚️ Detection Threshold")
        threshold_mode = st.radio(
            "Threshold Mode",
            ["Balanced", "Conservative", "Custom"],
            help="Balanced: Equal false positives/negatives\nConservative: Fewer false alarms"
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
            st.info(f"Using {threshold_mode} threshold: **{threshold:.6f}**")
        
        st.divider()
        
        # Score reference
        st.subheader("📈 Score Reference")
        st.markdown(f"""
        **Normal Range:**  
        Mean: `{config['normal_mean']:.6f}`  
        Median: `{config['normal_median']:.6f}`
        
        **Crack Range:**  
        Mean: `{config['crack_mean']:.6f}`  
        Median: `{config['crack_median']:.6f}`
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a concrete image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of concrete infrastructure"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Process button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner(f"Loading {substrate_type} model..."):
                    model = load_model(substrate_type)
                
                if model is not None:
                    with st.spinner("Analyzing image..."):
                        # Preprocess
                        img_tensor = preprocess_image(image)
                        
                        # Calculate reconstruction error
                        error_score, reconstruction = calculate_reconstruction_error(model, img_tensor)
                        
                        # Store in session state
                        st.session_state['error_score'] = error_score
                        st.session_state['reconstruction'] = reconstruction
                        st.session_state['threshold'] = threshold
                        st.session_state['config'] = config
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if 'error_score' in st.session_state:
            error_score = st.session_state['error_score']
            reconstruction = st.session_state['reconstruction']
            threshold = st.session_state['threshold']
            config = st.session_state['config']
            
            # Display reconstruction
            recon_image = tensor_to_image(reconstruction)
            st.image(recon_image, caption="Model Reconstruction", use_container_width=True)
            
            # Detection result
            is_anomaly = error_score > threshold
            
            if is_anomaly:
                st.error("🔴 **CRACK DETECTED**")
                confidence = "Very Low" if config['separation'] < 1.5 else "Low"
            else:
                st.success("🟢 **NORMAL CONCRETE**")
                confidence = "Low"
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Reconstruction Error", f"{error_score:.6f}")
            with col_b:
                st.metric("Threshold", f"{threshold:.6f}")
            with col_c:
                st.metric("Confidence", confidence)
            
            # Score visualization
            st.markdown("---")
            st.markdown("### Score Context")
            
            # Create visual bar chart
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Add ranges
            fig.add_trace(go.Bar(
                x=['Normal Range', 'Your Image', 'Crack Range'],
                y=[config['normal_mean'], error_score, config['crack_mean']],
                marker_color=['green', 'blue' if not is_anomaly else 'red', 'red'],
                text=[f"{config['normal_mean']:.6f}", f"{error_score:.6f}", f"{config['crack_mean']:.6f}"],
                textposition='auto',
            ))
            
            # Add threshold line
            fig.add_hline(y=threshold, line_dash="dash", line_color="orange",
                         annotation_text=f"Threshold: {threshold:.6f}")
            
            fig.update_layout(
                title="Score Comparison",
                yaxis_title="Reconstruction Error",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Warning message
            st.warning(f"""
            ⚠️ **Note:** With {config['separation']}x separation, this prediction has approximately 
            {config['accuracy']}% reliability. This prototype demonstrates the technical challenge 
            of unsupervised crack detection.
            """)
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results")
    
    # About section
    with st.expander("📖 About This Prototype", expanded=False):
        st.markdown(f"""
        ## Unsupervised Anomaly Detection Results
        
        ### Current Model: {substrate_type}
        
        #### Performance Metrics
        - **Score Separation:** {config['separation']}x (Target: 5-10x) ❌
        - **Detection Accuracy:** ~{config['accuracy']}% (Target: 85-95%) ❌
        - **Training Loss:** {config['best_loss']:.6f} ✅
        - **Training Images:** {config['training_images']:,} ✅
        
        #### What This Means
        
        This prototype uses an **unsupervised autoencoder** approach that:
        
        ✅ **Successfully learned** substrate texture patterns  
        - Model achieved excellent loss convergence
        - Can reconstruct concrete textures accurately
        
        ❌ **Cannot reliably distinguish** cracks from normal texture  
        - Only {config['separation']}x score separation achieved
        - Results in ~60% accuracy (coin-flip territory)
        
        #### Key Discovery: Data Contamination
        
        During validation, we discovered that "uncracked" training folders contained images 
        with visible cracks. This data contamination explains the poor separation:
        
        - Model learned to reconstruct BOTH normal texture AND cracks as "normal"
        - When shown a crack, it reconstructs it well → low error → no detection
        
        **Visual Evidence:** Reconstruction quality tests revealed cracks in training data  
        **Impact:** Model cannot distinguish what it learned as normal
        
        #### Why Low Separation Occurs
        
        Even with clean data, cracks are thin linear features that create similar reconstruction 
        errors to normal texture variations (aggregate patterns, shadows, joints).
        
        **Analogy:** Teaching someone to perfectly copy concrete textures - they get so good 
        that they reproduce both normal patterns AND cracks with equal accuracy, making them 
        indistinguishable.
        
        #### Technical Validation
        
        | Substrate | Training Loss | Separation | Accuracy |
        |-----------|---------------|------------|----------|
        | Bridge Deck | 0.000067 | 1.2x | 59% |
        | Wall | 0.000025 | 1.4x | 60% |
        | Pavement | 0.000071 | 1.3x | 59% |
        
        All models achieved excellent convergence but insufficient separation for production use.
        
        #### Recommended Next Steps
        
        1. **Data Cleaning:** Remove contaminated images from training sets
        2. **Supervised Classification:** Train models that explicitly learn crack features
        3. **Thermal Imaging:** Explore temperature-based detection (physical defects → thermal signatures)
        4. **Expected Improvement:** 85-95% accuracy with supervised approach
        
        #### Use Case
        
        This prototype serves as:
        - **Phase 1 Diagnostic:** Validates infrastructure and identifies limitations
        - **Technical Baseline:** Demonstrates substrate-specific model pipeline
        - **Research Finding:** Documents why unsupervised approach requires clean data
        
        **Status:** Phase 2 (supervised classification) in development  
        **Contact:** For production deployment inquiries
        
        ---
        
        **⚠️ Disclaimer:** This is a research prototype demonstrating technical challenges 
        in unsupervised anomaly detection. Not intended for production use in current state.
        """)
    
    # Technical details
    with st.expander("🔧 Technical Details", expanded=False):
        st.markdown("""
        ### Model Architecture
        
        **Type:** Convolutional Autoencoder  
        **Parameters:** 187,011  
        **Input Size:** 256×256×3 (RGB)  
        **Architecture:**
        
        **Encoder:**
        - Conv2d(3→64, k=3, s=2) + ReLU
        - Conv2d(64→128, k=3, s=2) + ReLU
        - Conv2d(128→256, k=3, s=2) + ReLU
        
        **Decoder:**
        - ConvTranspose2d(256→128, k=3, s=2) + ReLU
        - ConvTranspose2d(128→64, k=3, s=2) + ReLU
        - ConvTranspose2d(64→3, k=3, s=2) + Sigmoid
        
        ### Training Configuration
        
        - **Optimizer:** Adam
        - **Learning Rate:** 0.001
        - **Batch Size:** 32
        - **Epochs:** 50 (with early stopping)
        - **Loss Function:** MSE (Mean Squared Error)
        - **Hardware:** NVIDIA A100 GPU (Google Colab Pro)
        
        ### Dataset
        
        **Source:** SDNET2018 (Concrete Crack Detection)  
        **Training Strategy:** Unsupervised (normal images only)  
        **Testing:** Separate cracked image sets
        
        ### Anomaly Score
        
        Reconstruction error calculated as Mean Squared Error (MSE) between original 
        and reconstructed image. Higher error indicates potential anomaly.
        
        ```python
        error = MSE(original, reconstruction)
        is_anomaly = error > threshold
        ```
        """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
