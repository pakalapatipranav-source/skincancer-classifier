"""
Streamlit application for skin cancer detection - Professional Edition
"""
import os
import sys
import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Skin Cancer Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "AI-Powered Skin Cancer Detection System - Research Project"
    }
)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
# Don't import tensorflow at top level - import lazily when needed
from typing import Optional, Tuple

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import utils (they'll handle tensorflow imports internally)
try:
    from utils.model_utils import (
        load_model,
        load_label_binarizer,
        load_class_names,
        load_model_metadata,
        preprocess_image
    )
except Exception as e:
    st.error(f"Error importing utilities: {e}")
    # Define fallback functions
    def load_model(*args, **kwargs):
        return None
    def load_label_binarizer(*args, **kwargs):
        return None
    def load_class_names(*args, **kwargs):
        return ['benign', 'malignant']
    def load_model_metadata(*args, **kwargs):
        return None
    def preprocess_image(*args, **kwargs):
        return None

# CSS will be injected in main() function

# Constants
MODEL_PATH = "skin_cancer_model.h5"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


@st.cache_resource
def load_model_cached():
    """Load and cache the model."""
    return load_model(MODEL_PATH)


@st.cache_resource
def load_class_names_cached():
    """Load and cache class names."""
    class_names = load_class_names()
    if not class_names or class_names == ['benign', 'malignant']:
        lb = load_label_binarizer()
        if lb:
            class_names = lb.classes_.tolist() if hasattr(lb, 'classes_') else ['benign', 'malignant']
    return class_names


def validate_image_file(uploaded_file) -> Tuple[bool, Optional[str]]:
    """Validate uploaded image file."""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    uploaded_file.seek(0, os.SEEK_END)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
    
    try:
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
        return True, None
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def preprocess_image_for_prediction(uploaded_file) -> Optional[np.ndarray]:
    """Preprocess uploaded image for model prediction."""
    try:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        img_array = preprocess_image(temp_path, target_size=(224, 224))
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None


def preprocess_pil_image(pil_image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess a PIL Image for model prediction.
    Uses EfficientNet's preprocess_input to match training preprocessing.
    
    Args:
        pil_image: PIL Image object
        target_size: Target size for the image (height, width)
        
    Returns:
        Preprocessed image array
    """
    try:
        # Lazy import to avoid loading tensorflow at startup
        from tensorflow.keras.applications.efficientnet import preprocess_input
        
        # Resize image (using PIL directly, no need for tensorflow.keras.preprocessing)
        img_resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Use EfficientNet's preprocess_input (normalizes to [-1, 1] range)
        # This matches the training preprocessing
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing PIL image: {e}")
        return None


def predict_image(model, img_array: np.ndarray, class_names: list) -> Tuple[str, float, np.ndarray]:
    """Make prediction on preprocessed image."""
    try:
        predictions = model.predict(img_array, verbose=0)
        pred_array = predictions[0]
        num_model_outputs = pred_array.size if hasattr(pred_array, 'size') else len(pred_array)
        num_expected_classes = len(class_names)
        
        if num_model_outputs == 1 and num_expected_classes == 2:
            prob = float(pred_array[0] if isinstance(pred_array, np.ndarray) else pred_array)
            if prob > 0.5:
                predicted_class = class_names[1]
                confidence = prob * 100
            else:
                predicted_class = class_names[0]
                confidence = (1 - prob) * 100
            all_predictions = np.array([1 - prob, prob])
        elif num_model_outputs == num_expected_classes:
            predicted_idx = int(np.argmax(pred_array))
            if predicted_idx >= len(class_names):
                predicted_idx = 0
            predicted_class = class_names[predicted_idx]
            confidence = float(pred_array[predicted_idx] * 100)
            all_predictions = pred_array
        else:
            if num_model_outputs > 0:
                predicted_idx = int(np.argmax(pred_array))
                if predicted_idx < len(class_names):
                    predicted_class = class_names[predicted_idx]
                    confidence = float(pred_array[predicted_idx] * 100)
                else:
                    predicted_class = class_names[0] if len(class_names) > 0 else "unknown"
                    confidence = float(pred_array[0] * 100) if len(pred_array) > 0 else 0.0
                if num_model_outputs < num_expected_classes:
                    all_predictions = np.pad(pred_array, (0, num_expected_classes - num_model_outputs), 'constant')
                else:
                    all_predictions = pred_array[:num_expected_classes]
            else:
                predicted_class = class_names[0] if len(class_names) > 0 else "unknown"
                confidence = 0.0
                all_predictions = np.zeros(num_expected_classes)
        
        return predicted_class, confidence, all_predictions
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        # Return a valid array with zeros instead of empty array
        fallback_predictions = np.zeros(len(class_names)) if len(class_names) > 0 else np.array([0.0])
        return "", 0.0, fallback_predictions


def create_confidence_chart(predictions_dict: dict, predicted_class: str) -> go.Figure:
    """Create an interactive confidence chart."""
    df = pd.DataFrame(list(predictions_dict.items()), columns=['Class', 'Confidence'])
    df = df.sort_values('Confidence', ascending=True)
    
    colors = ['#ff6b6b' if c.lower() == 'malignant' else '#51cf66' for c in df['Class']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['Class'].str.capitalize(),
            x=df['Confidence'],
            orientation='h',
            marker=dict(color=colors),
            text=[f"{conf:.1f}%" for conf in df['Confidence']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence Scores",
        xaxis_title="Confidence (%)",
        yaxis_title="",
        height=200,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        showlegend=False
    )
    
    return fig




def inject_global_css():
    st.markdown(
        """
        <style>
          .hero {
            padding: 1.25rem 1.25rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(102,126,234,0.18), rgba(118,75,162,0.18));
            border: 1px solid rgba(120,120,120,0.15);
          }
          .hero h1 { margin: 0; line-height: 1.15; }
          .hero p { margin: .35rem 0 0 0; color: rgba(120,120,120,0.95); }
          .card {
            padding: 1rem 1rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(120,120,120,0.15);
          }
          .muted { color: rgba(120,120,120,0.95); }
          .tiny { font-size: 0.9rem; }
          .pill {
            display: inline-block;
            padding: .25rem .6rem;
            border-radius: 999px;
            border: 1px solid rgba(120,120,120,0.25);
            font-size: .85rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def make_gauge(confidence_pct: float, title: str = "Malignancy confidence"):
    # Lightweight gauge using Plotly (you already import plotly.graph_objects as go)
    confidence_pct = float(max(0, min(100, confidence_pct)))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence_pct,
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 33], "color": "rgba(0, 200, 0, 0.15)"},
                    {"range": [33, 66], "color": "rgba(255, 165, 0, 0.15)"},
                    {"range": [66, 100], "color": "rgba(255, 0, 0, 0.15)"},
                ],
                "threshold": {"line": {"width": 3}, "thickness": 0.75, "value": confidence_pct},
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def push_history(record: dict):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, record)
    st.session_state.history = st.session_state.history[:10]


def main():
    # Inject CSS
    inject_global_css()
    
    # Title - show immediately
    st.title("🔬 Skin Cancer Detection System")
    st.write("")  # Spacing

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: .75rem 0 0.25rem 0;">
              <div style="font-size: 2rem;">🧠</div>
              <div style="font-size: 1.05rem; font-weight: 700;">Skin Cancer Detection Demo</div>
              <div class="muted tiny">Educational ML project</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("### ⚙️ Settings")
        show_confidence_breakdown = st.toggle("Show confidence breakdown", value=True)
        show_history = st.toggle("Show session history", value=True)

        st.markdown("---")
        st.markdown("### 📌 Notes")
        st.caption("This demo is for educational/research exploration only — not medical advice.")

        try:
            metadata = load_model_metadata()
            if metadata:
                st.markdown("### 📊 Model Snapshot")
                st.caption(f"Backbone: {metadata.get('model_name', 'CNN / EfficientNet')}")
                if "test_accuracy" in metadata:
                    st.caption(f"Reported test accuracy: {metadata.get('test_accuracy', 0):.1%}")
        except Exception:
            pass  # Silently fail - metadata is optional

    # --- Hero header ---
    st.markdown(
        """
        <div class="hero">
          <h1>Skin Lesion Classifier</h1>
          <p>Upload an image and see the model's prediction + confidence. Designed as a clean demo for portfolios and research applications.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Tabs --- (Create tabs first so UI renders immediately)
    tab_analyze, tab_explain, tab_model, tab_about = st.tabs(
        ["🔎 Analyze", "🧪 Explain", "📦 Model", "ℹ️ About"]
    )
    
    # Initialize model in session state (load on demand)
    if "model" not in st.session_state:
        st.session_state.model = None
        st.session_state.class_names = ["benign", "malignant"]
        st.session_state.model_loading = False

    # ========== ANALYZE TAB ==========
    with tab_analyze:
        # Don't load model automatically - only load when user clicks predict
        model = st.session_state.model
        class_names = st.session_state.class_names
        
        if model is None:
            st.info("ℹ️ Model will load automatically when you upload an image and click 'Run prediction'.")
        
        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            st.markdown("### Upload an image")
            uploaded_file = st.file_uploader(
                "Supported formats: JPG, PNG",
                type=[ext.replace(".", "") for ext in ALLOWED_EXTENSIONS],
                label_visibility="collapsed",
            )

            predict_btn = None
            image = None
            
            if uploaded_file is None:
                st.info("👆 **Upload an image above to get started!**")
            else:
                ok, err = validate_image_file(uploaded_file)
                if not ok:
                    st.info("Tip: Use a clear, centered image with good lighting.")
                    if err:
                        st.warning(err)
                else:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption="Uploaded image", width='stretch')
                    predict_btn = st.button("Run prediction", type="primary")

        with right:
            st.markdown("### Results")
            st.markdown(
                """
                <div class="card">
                  <div class="muted tiny">Output appears here after you run a prediction.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if predict_btn and image is not None:
                # Load model NOW (only when user clicks predict)
                if st.session_state.model is None:
                    try:
                        with st.spinner("🔄 Loading model (first time only)..."):
                            st.session_state.model = load_model_cached()
                            st.session_state.class_names = load_class_names_cached()
                    except Exception as e:
                        st.error(f"❌ Error loading model: {e}")
                        st.exception(e)
                
                model = st.session_state.model
                class_names = st.session_state.class_names
                
                if model is None:
                    st.error("❌ Model not loaded. Cannot make predictions.")
                else:
                    with st.spinner("🧠 Running inference..."):
                        # Preprocess the PIL image to numpy array
                        img_array = preprocess_pil_image(image)
                        if img_array is not None:
                            predicted_class, confidence, all_predictions = predict_image(model, img_array, class_names)
                        else:
                            st.error("Failed to preprocess image")
                            predicted_class, confidence, all_predictions = "", 0.0, np.array([])

                # Only show results if prediction was successful
                if not predicted_class:
                    st.error("❌ Failed to make prediction. Please try again.")
                else:
                    # Normalize interpretation for binary case if your classes include benign/malignant
                    pred_lower = str(predicted_class).lower()
                    is_malignant = "malig" in pred_lower

                    # Cards row
                    c1, c2, c3 = st.columns(3, gap="medium")
                    with c1:
                        st.markdown(
                            f"""
                        <div class="card">
                          <div class="muted tiny">Prediction</div>
                          <div style="font-size:1.3rem; font-weight:800;">{predicted_class}</div>
                          <div class="pill">Top class</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with c2:
                        st.markdown(
                            f"""
                        <div class="card">
                          <div class="muted tiny">Confidence</div>
                          <div style="font-size:1.3rem; font-weight:800;">{confidence:.1f}%</div>
                          <div class="pill">Model score</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with c3:
                        status = "⚠️ Review suggested" if is_malignant else "✅ Lower concern"
                        st.markdown(
                            f"""
                        <div class="card">
                          <div class="muted tiny">Guidance</div>
                          <div style="font-size:1.1rem; font-weight:800;">{status}</div>
                          <div class="muted tiny">Not medical advice</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Gauge + chart
                    st.write("")
                    gcol, pcol = st.columns([0.45, 0.55], gap="large")

                    with gcol:
                        # If you can identify which index is malignant, use that.
                        # Here we try: if a class name contains "malig", use that as malignancy confidence.
                        malignancy_conf = None
                        # Check if all_predictions has valid data
                        if len(all_predictions) > 0 and len(all_predictions) >= len(class_names):
                            for i, name in enumerate(class_names):
                                if i < len(all_predictions) and "malig" in str(name).lower():
                                    malignancy_conf = float(all_predictions[i] * 100.0)
                                    break
                        if malignancy_conf is None:
                            malignancy_conf = float(confidence) if is_malignant else float(100.0 - confidence)

                        st.plotly_chart(make_gauge(malignancy_conf), width='stretch', key='gauge_chart')

                    with pcol:
                        if show_confidence_breakdown:
                            # Build dict with proper bounds checking
                            predictions_dict = {}
                            if len(all_predictions) > 0:
                                # Handle case where predictions don't match class names
                                for i, class_name in enumerate(class_names):
                                    if i < len(all_predictions):
                                        predictions_dict[class_name] = float(all_predictions[i] * 100)
                                    else:
                                        predictions_dict[class_name] = 0.0
                            else:
                                # Fallback if all_predictions is empty
                                for class_name in class_names:
                                    predictions_dict[class_name] = float(confidence) if class_name.lower() == predicted_class.lower() else 0.0
                            
                            if predictions_dict:
                                fig = create_confidence_chart(predictions_dict, predicted_class)
                                st.plotly_chart(fig, width='stretch', key='confidence_breakdown_chart')
                            else:
                                st.warning("Unable to generate confidence breakdown.")
                        else:
                            st.caption("Confidence breakdown hidden (toggle in sidebar).")

                    # Recommendation block (keep this conservative)
                    st.markdown("---")
                    if is_malignant:
                        st.warning(
                            "This prediction suggests higher concern. Please consult a dermatologist for any medical questions. "
                            "This demo is not a diagnostic tool."
                        )
                    else:
                        st.info(
                            "This prediction suggests lower concern. If you notice changes or have concerns, consult a healthcare professional."
                        )

                    # History
                    push_history(
                    {
                        "prediction": predicted_class,
                        "confidence_pct": round(float(confidence), 2),
                        "classes": len(class_names),
                    }
                )
                if show_history and "history" in st.session_state and st.session_state.history:
                    st.markdown("#### 🕒 Session history (last 10)")
                    st.dataframe(pd.DataFrame(st.session_state.history), width='stretch', hide_index=True)

    # ========== EXPLAIN TAB ==========
    with tab_explain:
        st.markdown("### How this demo works")
        st.write(
            """
            **What happens when you click Run prediction**
            - The image is resized + normalized
            - The model runs inference (no training happens here)
            - We display the top predicted class and confidence
            
            **Important**
            - Medical images are complex, and confidence ≠ correctness
            - This demo is for educational/research exploration only
            """
        )

        with st.expander("What would improve this model next?"):
            st.write(
                """
                - Data augmentation (lighting, zoom, rotation)
                - Transfer learning with stronger pretrained backbones
                - Calibration (so confidence aligns better with accuracy)
                - Interpretability (Grad-CAM / saliency maps)
                - Class imbalance handling + better evaluation metrics
                """
            )

    # ========== MODEL TAB ==========
    with tab_model:
        st.markdown("### Model & Data")
        meta = load_model_metadata()
        if meta:
            st.json(meta)
        else:
            st.info("No metadata file detected. (Optional) Add one to show training details in the demo.")

        st.markdown("### Repo tips")
        st.write(
            """
            Consider adding:
            - `requirements.txt`
            - a short `MODEL_CARD.md` (what data, what limitations, how to use)
            - example images (optional)
            """
        )

    # ========== ABOUT TAB ==========
    with tab_about:
        st.markdown("### About this project")
        st.write(
            """
            This is a portfolio-quality ML demo built with **Streamlit** and **TensorFlow**.
            
            **Disclaimer:** Educational and research exploration only. Not for medical diagnosis.
            """
        )

    st.markdown("---")
    st.caption("Built with Streamlit • TensorFlow • Plotly • Educational use only")


# Streamlit runs the entire script, so call main() directly
try:
    main()
except Exception as e:
    st.error(f"❌ Fatal error in application: {e}")
    st.exception(e)
    st.info("Please check the terminal for more details.")
