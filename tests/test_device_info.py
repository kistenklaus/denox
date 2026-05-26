from helpers import run_denox

def test_merge_device_info(tmp_path):
    output = run_denox("query-device-info", "-o", tmp_path / "gpu.yaml");
    assert output.returncode == 0

    output = run_denox("merge-device-info", 
                       tmp_path / "gpu.yaml", 
                       tmp_path / "gpu.yaml", 
                       "-o", tmp_path / "merged.yaml")
    assert output.returncode == 0

    

