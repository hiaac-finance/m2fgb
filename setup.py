from setuptools import setup, find_packages


setup(
    name="m2fgb",
    version="1.0",
    packages=find_packages(),
    author="Giovani Valdrighi",
    author_email="giovani.valdrighi@ic.unicamp.br",
    description="A package for fair machine learning with gradient boosting.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="github.com/hiaac-finance/m2fgb",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "lightgbm>=3.2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Framework :: Scikit-learn",
        "Framework :: LightGBM",
    ],
    python_requires=">=3.7",
)