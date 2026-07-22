import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Project Problem Runner")
    parser.add_argument(
        "--problem",
        type=int,
        default=1,
        help="Problem number to run (default: 1)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    if args.problem == 1:
        from src.problems.problem1 import run_problem1

        data_dir = args.data_dir or "problem1/附件1"
        output_dir = args.output_dir or "outputs/problem1"
        result = run_problem1(Path(data_dir), Path(output_dir))

        print("Problem 1 completed successfully.")
        print("Metrics:")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.6f}")
        print(f"Output saved to: {result['output_dir']}")
    else:
        print(f"Problem {args.problem} is not implemented yet.")


if __name__ == "__main__":
    main()
