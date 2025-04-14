#!/usr/bin/env python3
"""
Script to patch the BeeChunkerSOM._extract_features method to support simplified categorical data.
"""
import os
import sys
import re

# Path to your SOM implementation file
SOM_PATH = "/home/jgajbha/Chunker/beechunker/ml/som.py"

# Load the new method from file
def get_new_method():
    with open("modified_extract_features.py", "r") as f:
        return f.read()

def apply_patch():
    # Check if file exists
    if not os.path.exists(SOM_PATH):
        print(f"Error: SOM implementation not found at {SOM_PATH}")
        print("Please update the SOM_PATH variable to point to your som.py file.")
        return False
    
    # Read the current file
    with open(SOM_PATH, "r") as f:
        content = f.read()
    
    # Create a backup
    with open(f"{SOM_PATH}.bak", "w") as f:
        f.write(content)
        print(f"Created backup at {SOM_PATH}.bak")
    
    # Find the method to replace using regex
    pattern = r"(\s+def _extract_features\(self, df\):.*?)(\s+def _create_chunk_size_map)"
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("Error: Could not find the _extract_features method in the SOM implementation")
        return False
    
    # Get new method content
    new_method = get_new_method()
    
    # Replace the method
    updated_content = content.replace(match.group(1), new_method)
    
    # Write the updated file
    with open(SOM_PATH, "w") as f:
        f.write(updated_content)
    
    print(f"Successfully patched {SOM_PATH} with modified _extract_features method")
    print("The SOM implementation now supports both original and simplified data formats")
    return True

if __name__ == "__main__":
    if apply_patch():
        print("\nYou can now train with your simplified data:")
        print("beechunker-train train --input-csv /home/jgajbha/Chunker/data/beegfs_test_results_simplified.csv")
    else:
        print("\nPatch failed. Please check the paths and try again.")