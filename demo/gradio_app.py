#!/usr/bin/env python3
"""
Local Gradio interface for Vision Transformer defect detection.

Requires a checkpoint produced by this project.
Run with: python demo/gradio_app.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

from src.inference import DefectDetector
from src.utils.visualization import plot_attention_maps

# Global detector instance
detector = None
model = None

# Class names
CLASS_NAMES = ["ok", "scratch", "crack", "dent", "corrosion"]
CLASS_COLORS = {
    "ok": "#28a745",
    "scratch": "#ffc107",
    "crack": "#dc3545",
    "dent": "#17a2b8",
    "corrosion": "#6f42c1",
}


def load_model(model_path: str = None):
    """Load the defect detection model"""
    global detector, model

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
            return "No model found. Please train a model first or specify a model path."

    try:
        detector = DefectDetector(model_path, device="auto")
        model = detector.model
        return f"Model loaded successfully from {model_path}"
    except Exception as e:
        return f"Error loading model: {str(e)}"


def predict_defect(image: Image.Image, show_attention: bool = True):
    """Predict defect from uploaded image"""
    global detector, model

    if detector is None:
        return None, "Please load a model first!", None

    if image is None:
        return None, "Please upload an image!", None

    try:
        # Convert PIL to numpy array
        img_array = np.array(image)

        # Run prediction
        result = detector.predict(img_array)

        # Get prediction details
        predicted_class = result["class"]
        confidence = result["confidence"]
        all_probs = result.get("probabilities", {})

        # Create confidence bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Bar chart of probabilities
        classes = list(all_probs.keys()) if all_probs else CLASS_NAMES
        probs = [all_probs.get(c, 0.0) for c in classes]
        colors = [CLASS_COLORS.get(c, "#6c757d") for c in classes]

        bars = ax1.barh([class_name.title() for class_name in classes], probs, color=colors)
        ax1.set_xlabel("Confidence", fontsize=12)
        ax1.set_title("Class Probabilities", fontsize=14, fontweight="bold")
        ax1.set_xlim(0, 1)
        ax1.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax1.text(prob + 0.01, i, f"{prob:.2%}", va="center", fontsize=10, fontweight="bold")

        # Prediction summary
        summary_text = f"""
        <div style="text-align: center; padding: 20px;">
            <h2 style="color: {CLASS_COLORS.get(predicted_class, '#000')}; margin-bottom: 10px;">
                {predicted_class.title()}
            </h2>
            <p style="font-size: 24px; font-weight: bold; color: #333;">
                Confidence: {confidence:.2%}
            </p>
            <p style="font-size: 14px; color: #666; margin-top: 10px;">
                Latency: {result.get('latency_ms', 0):.1f}ms
            </p>
        </div>
        """

        ax2.axis("off")
        ax2.text(
            0.5,
            0.5,
            f"{predicted_class.title()}\n{confidence:.2%}",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        # Convert figure to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        prob_chart = Image.open(buf)
        plt.close()

        # Generate attention map if requested
        attention_img = None
        if show_attention and model is not None:
            try:
                # Preprocess image for attention visualization
                transform = detector.transform
                img_tensor = transform(image).unsqueeze(0).to(detector.device)

                # Get attention map
                fig = plot_attention_maps(model, img_tensor[0], layer_idx=-1, head_idx=None)

                # Convert to PIL Image
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                attention_img = Image.open(buf)
                plt.close(fig)
            except Exception as e:
                print(f"Error generating attention map: {e}")

        return prob_chart, summary_text, attention_img

    except Exception as e:
        return None, f"Error during prediction: {str(e)}", None


def create_demo():
    """Create the local Gradio interface."""

    # Load model on startup
    model_status = load_model()

    with gr.Blocks(title="Vision Transformer Defect Detection") as demo:
        gr.Markdown(
            """
        # 🔍 Vision Transformer Defect Detection

        Load a trained checkpoint, then upload a surface image for classification.

        This local interface is not a calibrated industrial inspection system.

        **Supported defect types:** OK, Scratch, Crack, Dent, Corrosion
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Configuration")
                model_path_input = gr.Textbox(
                    label="Model Path (optional)",
                    placeholder="models/checkpoints/best_model.pth",
                    value="",
                )
                load_btn = gr.Button("Load Model", variant="primary")
                model_status_output = gr.Textbox(
                    label="Model Status", value=model_status, interactive=False
                )

            with gr.Column(scale=2):
                gr.Markdown("### Image Upload & Prediction")
                image_input = gr.Image(label="Upload Image", type="pil", height=300)

                with gr.Row():
                    show_attention = gr.Checkbox(label="Show Attention Map", value=True)
                    predict_btn = gr.Button("Detect Defect", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Prediction Results")
                prob_chart_output = gr.Image(label="Class Probabilities")
                prediction_output = gr.HTML(label="Prediction")

            with gr.Column():
                gr.Markdown("### Attention Visualization")
                attention_output = gr.Image(label="Attention Map")

        # Event handlers
        load_btn.click(fn=load_model, inputs=model_path_input, outputs=model_status_output)

        predict_btn.click(
            fn=predict_defect,
            inputs=[image_input, show_attention],
            outputs=[prob_chart_output, prediction_output, attention_output],
        )

        image_input.change(
            fn=predict_defect,
            inputs=[image_input, show_attention],
            outputs=[prob_chart_output, prediction_output, attention_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
