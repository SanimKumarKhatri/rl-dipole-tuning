"""
Ensures the repo root is on sys.path so tests/ can import modules
like nec_core, dipole_env, optimizers directly, regardless of which
directory pytest is invoked from.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
