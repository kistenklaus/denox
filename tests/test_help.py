from helpers import run_denox

def test_help():
    result = run_denox("--help")

    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert output.strip()

def test_help_bench():
    result = run_denox("bench", "--help")

    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert output.strip()

def test_help_compile():
    result = run_denox("compile", "--help")

    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert output.strip()

def test_help_infer():
    result = run_denox("infer", "--help")

    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert output.strip()

def test_help_populate():
    result = run_denox("populate", "--help")

    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert output.strip()

def test_help_reweight():
    result = run_denox("reweight", "--help")

    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert output.strip()

def test_help_query_device_info():
    result = run_denox("query-device-info", "--help")

    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert output.strip()

def test_help_merge_device_info():
    result = run_denox("merge-device-info", "--help")

    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert output.strip()
