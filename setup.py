from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sagan-trade",
    version="0.8.0",
    author="Sagan Labs",
    author_email="hello@sagan-docs.vercel.app",
    description="Strategic High-Throughput Symbolic Trading Engine with iterative R2 fitting, FunctionGemma discovery, and Asymmetric Convexity risk management.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/That-Tech-Geek/sagan-trade",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "numba",
        "tensorflow>=2.10",
        "pandas>=1.5",
        "numpy>=1.23",
        "yfinance>=0.2",
        "scikit-learn>=1.1",
        "streamlit>=1.25",
        "plotly>=5.15",
        "cryptography",
        "typer",
        "snaptrade-python-sdk",
        "schedule",
        "requests",
        "huggingface_hub",
        "transformers",
        "datasets",
        "torch",
        "kaggle",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.0",
            "ruff>=0.4",
            "mypy>=1.5",
            "build>=1.0",
            "twine>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sagan = sagan.cli.commands:app",
            "sagan-download-models = sagan.symbolic_lib.download_models:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
