"""Setup script for GraPHFormer."""

from setuptools import setup, find_packages

setup(
    name="graphformer",
    version="1.0.0",
    description="GraPHFormer: Graph-Persistence Hybrid Transformer for Neuron Morphology",
    author="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "dgl>=0.8.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.6.0",
        "tqdm>=4.60.0",
        "networkx>=2.6.0",
        "Pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
)
