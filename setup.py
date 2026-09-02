"""Package metadata for Vision Transformer defect detection."""

from pathlib import Path
from setuptools import find_packages, setup

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
install_requires = [
    line.strip()
    for line in requirements_file.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setup(
    name="vision-transformer-defect-detection",
    version="0.1.0",
    description="Train and evaluate a Vision Transformer on synthetic surface defects",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lucien-vallois/vision-transformer-defect-detection",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.10,<3.13",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "build>=1.0.0",
        ],
        "demo": [
            "gradio>=4.0,<7",
            "pandas>=2,<3",
            "plotly>=5,<7",
            "streamlit>=1.30,<2",
        ],
        "comparison": ["timm>=0.9,<2"],
        "export": [
            "onnx>=1.17,<2",
            "onnxruntime>=1.16,<2",
            "onnxscript>=0.3,<1",
        ],
        "tensorboard": ["tensorboard>=2.12,<3"],
    },
    entry_points={
        "console_scripts": [
            "vit-train=src.train:main",
            "vit-inference=src.inference:main",
            "vit-export-onnx=scripts.export_onnx:main",
            "vit-generate-data=scripts.generate_synthetic_data:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="vision-transformer, defect-detection, computer-vision, pytorch",
)
