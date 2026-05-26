from pathlib import Path
import subprocess


def find_project_root(start: Path | None = None) -> Path:
    path = (start or Path(__file__)).resolve()

    for parent in [path, *path.parents]:
        if (parent / ".git").is_dir():
            return parent

    raise RuntimeError("Could not find project root containing .git")


PROJECT_ROOT = find_project_root()
CLI_PATH = PROJECT_ROOT / "build" / "bin" / "denox"


def run_denox(
    *args,
    timeout: float = 30.0,
):
    command = [
        str(CLI_PATH),
        *(str(arg) for arg in args),
    ]

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )

    print(result.stdout, end="")
    print(result.stderr, end="")

    return result
