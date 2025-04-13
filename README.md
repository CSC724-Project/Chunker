# BeeChunker

A dynamic BeeGFS chunk size optimization system using Self-Organizing Maps.

## Features

- Monitors file access patterns in BeeGFS
- Uses Self-Organizing Maps to learn optimal chunk sizes
- Automatically applies learned configurations to new files
- Provides visualization of access patterns and chunk size distributions

## Installation

```bash
pip install -e .