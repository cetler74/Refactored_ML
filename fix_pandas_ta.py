#!/usr/bin/env python3
"""Fix pandas-ta compatibility with new NumPy versions"""

import os
import re

# Find the pandas-ta package directory
import pandas_ta
package_dir = os.path.dirname(pandas_ta.__file__)

# Path to the file with the issue
file_path = os.path.join(package_dir, 'momentum', 'squeeze_pro.py')

# Check if the file exists
if os.path.exists(file_path):
    print(f"Fixing file: {file_path}")
    
    # Read the current content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Replace the problematic import
    new_content = content.replace(
        'from numpy import NaN as npNaN',
        'import numpy as np\nnpNaN = float("nan")'
    )
    
    # Write the modified content back
    with open(file_path, 'w') as file:
        file.write(new_content)
    
    print("Fix applied successfully!")
else:
    print(f"File not found: {file_path}") 