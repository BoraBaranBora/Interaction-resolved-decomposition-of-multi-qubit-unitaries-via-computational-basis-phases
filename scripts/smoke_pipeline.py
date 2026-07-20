from pipeline_common import run_python


def main() -> None:
    run_python("scripts/smoke_optimization.py", "--gate", "all")
    run_python("scripts/run_noise_analysis.py", "--mode", "smoke")


if __name__ == "__main__":
    main()
