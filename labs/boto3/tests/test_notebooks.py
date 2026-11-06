"""Fresh kernels execute with AWS credential lookup and network access prohibited."""
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "notebooks").glob("*.ipynb")), ids=lambda p: p.stem)
def test_offline_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert all(c.get("execution_count") is None and not c.get("outputs")
               for c in notebook.cells if c.cell_type == "code")
    notebook.cells.insert(0, nbformat.v4.new_code_cell('''
import socket
import boto3

def forbidden(*args, **kwargs):
    raise AssertionError("Network and AWS clients are forbidden in offline labs")

boto3.client = forbidden
boto3.Session = forbidden
socket.create_connection = forbidden
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
'''))
    env = {k:v for k,v in os.environ.items() if not k.startswith(("AWS_", "OPENAI_"))}
    env.update(AWS_EC2_METADATA_DISABLED="true", IPYTHONDIR="/tmp/ai-eval-ipython",
               JUPYTER_RUNTIME_DIR="/tmp/ai-eval-jupyter-runtime")
    NotebookClient(notebook, timeout=120, kernel_name="python3",
                   resources={"metadata":{"path":str(ROOT / "notebooks")}}).execute(env=env)
