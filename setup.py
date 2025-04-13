# setup script for the package
from setuptools import setup, find_packages

setup(
    name="beechunker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
        "minisom>=2.2.9",
        "matplotlib>=3.4.0",
        "watchdog>=2.1.0",
        "click>=8.0.0",
    ],
    entry_points={
        'console_scripts': [
            'beechunker-monitor=beechunker.cli.monitor_cli:main',
            'beechunker-optimizer=beechunker.cli.optimizer_cli:main',
            'beechunker-trainer=beechunker.cli.trainer_cli:main',
        ],
    },
)