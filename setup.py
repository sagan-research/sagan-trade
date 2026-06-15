from setuptools import setup, find_packages

setup(
    name="sagan-trade",
    version="0.9.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "numba",
        "scipy",
        "scikit-learn",
        "streamlit",
        "plotly"
    ],
    author="Sambit Mishra",
    author_email="sambit1912@gmail.com",
    description="Sagan High Frequency Trading Engine with Hawkes & Bates Jump-Diffusion",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/That-Tech-Geek/sagan-trade",
    project_urls={
        'Source': 'https://github.com/That-Tech-Geek/sagan-trade',
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
