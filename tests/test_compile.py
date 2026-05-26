from helpers import run_denox
from models import MODELS;

def test_compile():
    for onnx in MODELS:
        onnx.save("/tmp/net.onnx")
        output = run_denox("compile", "net.onnx")
        assert output.returncode == 0
    pass
