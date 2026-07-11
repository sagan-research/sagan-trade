from setuptools import setup, find_packages

setup(
    name="sagan-trade",
    version="0.9.6",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "yfinance",
        "matplotlib",
        "seaborn"
    ],
    author="Sambit Mishra",
    author_email="sambit1912@gmail.com",
    description="Algorithmic trading architecture designed by the Autonomous Intelligence Network (AIN).",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/That-Tech-Geek/sagan-trade",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
