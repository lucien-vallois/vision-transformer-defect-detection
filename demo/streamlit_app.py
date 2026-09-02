#!/usr/bin/env python3
"""
Local Streamlit interface for Vision Transformer defect detection.

Requires a checkpoint produced by this project.
Run with: streamlit run demo/streamlit_app.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

from src.inference import DefectDetector
from src.utils.visualization import plot_attention_maps, plot_multi_head_attention

# Page configuration
st.set_page_config(
    page_title="Vision Transformer Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Class names and colors
CLASS_NAMES = ["ok", "scratch", "crack", "dent", "corrosion"]
CLASS_COLORS = {
    "ok": "#28a745",
    "scratch": "#ffc107",
    "crack": "#dc3545",
    "dent": "#17a2b8",
    "corrosion": "#6f42c1",
}

# Initialize session state
if "detector" not in st.session_state:
    st.session_state.detector = None
    st.session_state.model = None
    st.session_state.history = []
    st.session_state.model_path = None


def load_model(model_path: str = None):
    """Load the defect detection model"""
    if model_path is None or model_path == "":
        # Try to find a default model
        default_paths = [
            "models/checkpoints/best_model.pth",
            "experiments/smoke/checkpoints/best_model.pth",
        ]

        for path in default_paths:
            if Path(path).exists():
                model_path = path
                break

    if model_path is None:
        return False, "No model found. Please train a model first or specify a model path."

    try:
        detector = DefectDetector(model_path, device="auto")
        st.session_state.detector = detector
        st.session_state.model = detector.model
        st.session_state.model_path = model_path
        return True, f"Model loaded successfully from {model_path}"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"


def predict_defect(image: Image.Image, show_attention: bool = True, save_to_history: bool = True):
    """Predict defect from uploaded image"""
    if st.session_state.detector is None:
        return None, "Please load a model first!"

    if image is None:
        return None, "Please upload an image!"

    try:
        # Convert PIL to numpy array
        img_array = np.array(image)

        # Run prediction
        result = st.session_state.detector.predict(img_array)

        # Save to history
        if save_to_history:
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "predicted_class": result["class"],
                "confidence": result["confidence"],
                "probabilities": result.get("probabilities", {}),
                "latency_ms": result.get("latency_ms", 0),
            }
            st.session_state.history.append(history_entry)

        return result, None

    except Exception as e:
        return None, f"Error during prediction: {str(e)}"


def create_probability_chart(result: dict):
    """Create interactive probability chart"""
    all_probs = result.get("probabilities", {})
    classes = list(all_probs.keys()) if all_probs else CLASS_NAMES
    probs = [all_probs.get(c, 0.0) for c in classes]

    fig = go.Figure(
        data=[
            go.Bar(
                x=probs,
                y=[class_name.title() for class_name in classes],
                orientation="h",
                marker=dict(color=[CLASS_COLORS.get(c, "#6c757d") for c in classes]),
                text=[f"{p:.2%}" for p in probs],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Confidence",
        xaxis_range=[0, 1],
        height=300,
        showlegend=False,
    )

    return fig


def main():
    """Main Streamlit app"""

    # Title
    st.title("🔍 Vision Transformer Defect Detection")
    st.markdown(
        "Local surface-defect classification. Results are experimental and require a trained checkpoint."
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model loading
        st.subheader("Model")
        model_path_input = st.text_input(
            "Model Path",
            value=st.session_state.model_path or "",
            placeholder="models/checkpoints/best_model.pth",
        )

        if st.button("Load Model", type="primary"):
            success, message = load_model(model_path_input)
            if success:
                st.success(message)
            else:
                st.error(message)

        if st.session_state.detector is not None:
            st.success("✅ Model loaded")
            st.info(f"Device: {st.session_state.detector.device}")
            st.info(f"Format: {st.session_state.detector.model_format}")

        st.divider()

        # Settings
        st.subheader("Settings")
        show_attention = st.checkbox("Show Attention Maps", value=True)
        show_multi_head = st.checkbox("Show Multi-Head Attention", value=False)
        num_layers = len(st.session_state.model.blocks) if st.session_state.model is not None else 0
        layer_idx = st.slider("Attention Layer", -num_layers, -1, -1) if num_layers > 1 else -1

        st.divider()

        # History
        st.subheader("History")
        st.metric("Predictions", len(st.session_state.history))

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

        if st.button("Export History"):
            if st.session_state.history:
                df = pd.DataFrame(st.session_state.history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

    # Main content
    tab1, tab2, tab3 = st.tabs(["🔬 Single Image", "📊 Batch Analysis", "📈 History & Stats"])

    with tab1:
        st.header("Single Image Detection")

        # Image upload
        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg"],
                help="Upload an image to detect defects",
            )

        with col2:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
            else:
                image = None

            if image is not None:
                st.image(image, caption="Input Image", use_container_width=True)

                # Predict button
                if st.button("🔍 Detect Defect", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        result, error = predict_defect(image, show_attention, save_to_history=True)

                        if error:
                            st.error(error)
                        elif result:
                            # Display results
                            st.success("✅ Analysis complete!")

                            # Prediction summary
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Predicted Class", result["class"])
                            with col_b:
                                st.metric("Confidence", f"{result['confidence']:.2%}")
                            with col_c:
                                st.metric("Latency", f"{result.get('latency_ms', 0):.1f}ms")

                            # Probability chart
                            prob_fig = create_probability_chart(result)
                            st.plotly_chart(prob_fig, use_container_width=True)

                            # Attention maps
                            if show_attention and st.session_state.model is not None:
                                st.subheader("Attention Visualization")

                                try:
                                    # Preprocess image
                                    transform = st.session_state.detector.transform
                                    img_tensor = (
                                        transform(image)
                                        .unsqueeze(0)
                                        .to(st.session_state.detector.device)
                                    )

                                    # Single attention map
                                    if not show_multi_head:
                                        fig = plot_attention_maps(
                                            st.session_state.model,
                                            img_tensor[0],
                                            layer_idx=layer_idx,
                                            head_idx=None,
                                        )
                                        st.pyplot(fig)
                                    else:
                                        # Multi-head attention
                                        fig = plot_multi_head_attention(
                                            st.session_state.model,
                                            img_tensor[0],
                                            layer_idx=layer_idx,
                                        )
                                        st.pyplot(fig)

                                except Exception as e:
                                    st.warning(f"Could not generate attention map: {e}")

    with tab2:
        st.header("Batch Analysis")

        uploaded_files = st.file_uploader(
            "Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
        )

        if uploaded_files and st.button("Process All Images", type="primary"):
            if st.session_state.detector is None:
                st.error("Please load a model first!")
            else:
                results = []
                progress_bar = st.progress(0)

                for i, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file)
                    result, error = predict_defect(
                        image, show_attention=False, save_to_history=False
                    )

                    if result:
                        results.append(
                            {
                                "filename": uploaded_file.name,
                                "class": result["class"],
                                "confidence": result["confidence"],
                                "latency_ms": result.get("latency_ms", 0),
                            }
                        )

                    progress_bar.progress((i + 1) / len(uploaded_files))

                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)

                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Images", len(results))
                    with col2:
                        avg_conf = df["confidence"].mean()
                        st.metric("Avg Confidence", f"{avg_conf:.2%}")
                    with col3:
                        avg_latency = df["latency_ms"].mean()
                        st.metric("Avg Latency", f"{avg_latency:.1f}ms")

                    # Class distribution
                    class_counts = df["class"].value_counts()
                    fig = px.pie(
                        values=class_counts.values,
                        names=class_counts.index,
                        title="Class Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Prediction History & Statistics")

        if st.session_state.history:
            # Convert to DataFrame
            df = pd.DataFrame(st.session_state.history)

            # Display table
            st.subheader("Recent Predictions")
            display_df = df[["timestamp", "predicted_class", "confidence", "latency_ms"]].copy()
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            display_df.columns = ["Timestamp", "Class", "Confidence", "Latency (ms)"]
            st.dataframe(display_df, use_container_width=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", len(df))
            with col2:
                avg_conf = df["confidence"].mean()
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
            with col3:
                avg_latency = df["latency_ms"].mean()
                st.metric("Avg Latency", f"{avg_latency:.1f}ms")
            with col4:
                most_common = (
                    df["predicted_class"].mode()[0]
                    if len(df["predicted_class"].mode()) > 0
                    else "N/A"
                )
                st.metric("Most Common", most_common)

            # Charts
            col_a, col_b = st.columns(2)

            with col_a:
                # Class distribution over time
                df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
                df["hour"] = df["timestamp_dt"].dt.hour
                hourly_counts = (
                    df.groupby(["hour", "predicted_class"]).size().reset_index(name="count")
                )

                fig = px.bar(
                    hourly_counts,
                    x="hour",
                    y="count",
                    color="predicted_class",
                    title="Predictions by Hour",
                    labels={"hour": "Hour of Day", "count": "Number of Predictions"},
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                # Confidence distribution
                fig = px.histogram(
                    df,
                    x="confidence",
                    nbins=20,
                    title="Confidence Distribution",
                    labels={"confidence": "Confidence", "count": "Frequency"},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Class distribution pie chart
            class_counts = df["predicted_class"].value_counts()
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Overall Class Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No prediction history yet. Start making predictions to see statistics here!")


if __name__ == "__main__":
    main()
