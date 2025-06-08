import argparse
from pathlib import Path
import subprocess
import filecmp

def run_test(break_on_fail, local, path_to_test):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    solution = "aw429669"
    make = subprocess.run("make", cwd=solution.name, capture_output=True, timeout=300)
    if make.returncode != 0:
        print(f"{solution.name}: FAILED (make)")
        if break_on_fail:
            print(make.stdout)
            exit(1)
        exit(1)
    print(f"Solution: {solution.name}")

    test = path_to_test
    workers = int(test.name[test.name.rfind("_") + 1:])
    # Remove old outputs
    for f in Path("outputs").iterdir():
        f.unlink()
    if local:
        command = "mpiexec"
    else:
        command = "srun"
    execution = subprocess.run([command, "-n", str(workers), "./test_command.sh", solution.name, test.name], capture_output=True, timeout=300)
    if execution.returncode != 0:
        print(f"    {test.name}: FAILED (srun)")
        if break_on_fail:
            print(f"{execution.stdout=}")
            print(f"{execution.stderr=}")
            exit(1)
        exit(1)
    failed = False
    for i in range(workers):
        if not filecmp.cmp(f"tests/{test.name}/{i}.out", f"outputs/{i}.out", shallow=False):
            print(f"    {test.name}: FAILED (outputs differ on rank {i})")
            failed = True
            if break_on_fail:
                exit(1)
            break
    if not failed:
        print(f"    {test.name}: PASSED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test runner')
    parser.add_argument('-b', '--breakonfail', action='store_true', help='break and print stdout on fail')
    parser.add_argument('-l', '--local', action='store_true', help='run tests locally (without slurm)')
    parser.add_argument('path_to_test', help='run tests locally (without slurm)')

    args = parser.parse_args()
    run_test(args.breakonfail, args.local, args.path_to_test)
