from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sagan-trade",
    version="0.3.5",
    author="Sagan Labs",
    author_email="hello@sagan-docs.vercel.app",
    description="High-throughput symbolic regression engine for mathematical alpha generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sagan-labs/sagan-xai",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10",
        "pandas>=1.3",
        "numpy>=1.21",
        "yfinance>=0.2",
        "scikit-learn>=1.0",
        "streamlit>=1.25",
        "plotly>=5.15",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "sagan = sagan.cli:main",
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
