from setuptools import setup, find_packages

setup(
    name="chunker_nn",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.2",
        "pandas==2.1.3",
        "scikit-learn==1.3.2",
        "xgboost==2.0.2",
        "torch==2.2.0",
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "python-dotenv==1.0.0",
        "tqdm==4.66.1"
    ]
) 