"""
setup.py — AHN library installation script.

Install (development mode):
    pip install -e .

Install from source:
    pip install .
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read version without importing the package (avoids circular imports at build)
_VERSION_FILE = Path(__file__).parent / "ahn" / "_version.py"
_version_ns: dict = {}
exec(_VERSION_FILE.read_text(), _version_ns)

LONG_DESCRIPTION = Path("README.md").read_text(encoding="utf-8")

setup(
    name             = "ahn",
    version          = _version_ns["__version__"],
    author           = _version_ns["__author__"],
    license          = _version_ns["__license__"],
    description      = (
        "Artificial Hydrocarbon Networks — "
        "bio-inspired ML for binary classification"
    ),
    long_description          = LONG_DESCRIPTION,
    long_description_content_type = "text/markdown",
    url              = _version_ns["__url__"],
    packages         = find_packages(exclude=["tests*", "docs*"]),
    python_requires  = ">=3.8",
    install_requires = [
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "pandas>=1.3",
        "matplotlib>=3.5",
        "seaborn>=0.12",
    ],
    extras_require = {
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords = [
        "machine learning", "hydrocarbon networks",
        "AHN", "classification", "bio-inspired",
    ],
)
