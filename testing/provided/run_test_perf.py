import argparse
from pathlib import Path
import subprocess
import filecmp

def run_test(break_on_fail, local, path_to_test, workers):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    solution = Path("aw429669")
    make = subprocess.run("make", cwd=solution.name, capture_output=True, timeout=300)
    if make.returncode != 0:
        print(f"{solution.name}: FAILED (make)")
        if break_on_fail:
            print(make.stdout)
            exit(1)
        exit(1)
    print(f"Solution: {solution.name}")

    test = Path(path_to_test)
    # workers = int(test.name[test.name.rfind("_") + 1:])
    # Remove old outputs
    for f in Path("outputs").iterdir():
        f.unlink()
    if local:
        command = "mpiexec"
    else:
        command = "srun"
    execution = subprocess.run([command, "-n", str(workers), "./perf_command.sh", solution.name, test.relative_to(Path("."))], capture_output=True, timeout=300)
    if execution.returncode != 0:
        print(f"    {test.name}: FAILED (srun)")
        if break_on_fail:
            print(f"{execution.stdout=}")
            print(f"{execution.stderr=}")
            exit(1)
        exit(1)
    print(f"    {test.name}: PASSED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test runner')
    parser.add_argument('-b', '--breakonfail', action='store_true', help='break and print stdout on fail')
    parser.add_argument('-l', '--local', action='store_true', help='run tests locally (without slurm)')
    parser.add_argument('nodes', help='number of nodes')
    parser.add_argument('workers', help='number of workers per node')
    parser.add_argument('path_to_test', help='Path to test directory')

    args = parser.parse_args()
    run_test(args.breakonfail, args.local, args.path_to_test, int(args.nodes) * int(args.workers))
