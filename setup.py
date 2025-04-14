# setup script for the package
from setuptools import setup, find_packages

setup(
    name="beechunker",
    version="0.1.0",
    description="Intelligent chunk size optimization for BeeGFS using SOM",
    author="BeeChunker Team",
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
        "scipy>=1.7.0",
    ],
    entry_points={
        'console_scripts': [
            'beechunker-monitor=beechunker.cli.monitor_cli:main',
            'beechunker-optimizer=beechunker.cli.optimizer_cli:main',
            'beechunker-train=beechunker.cli.trainer_cli:main',
            'beechunker=beechunker.cli.monitor_cli:main',  # Default command
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: System Administrators',
        'Topic :: System :: Filesystems',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)