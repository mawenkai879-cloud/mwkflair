from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mwkflair",
    version="1.0.0",
    author="MWK",
    author_email="",
    description="FLAIR with HarMA Adapter Integration for Vision-Language Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mwkflair",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "mwkflair": ["model_configs/*.json"],
    },
    keywords=[
        "vision-language",
        "CLIP",
        "FLAIR",
        "HarMA",
        "adapter",
        "fine-tuning",
        "deep-learning",
        "pytorch",
    ],
)
