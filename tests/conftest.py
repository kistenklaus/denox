import pytest


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("denox")
