# notebooks/conftest.py
import sys
import os

# Add directory 'src' to modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

