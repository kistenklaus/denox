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
    timeout: float = 3600,
    verbose: bool = True,
):
    command = [
        str(CLI_PATH),
        *(str(arg) for arg in args),
    ]

    if not verbose:
        return subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    output = []

    try:
        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            output.append(line)

        returncode = process.wait(timeout=timeout)

    except subprocess.TimeoutExpired:
        process.kill()
        stdout, _ = process.communicate()

        if stdout:
            print(stdout, end="")
            output.append(stdout)

        raise

    stdout = "".join(output)

    return subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )
