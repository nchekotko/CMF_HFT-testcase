"""Execute a notebook in-place using the venv kernel via nbclient (bypasses the broken
global nbconvert config). Usage: python _run_nb.py <notebook.ipynb>"""
import os
import sys

import nbformat
from nbclient import NotebookClient

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
client = NotebookClient(
    nb, timeout=900, kernel_name="hw3venv",
    resources={"metadata": {"path": os.path.dirname(os.path.abspath(path)) or "."}},
)
client.execute()
nbformat.write(nb, path)
print("executed OK:", path)
