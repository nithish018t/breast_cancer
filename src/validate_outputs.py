import runpy
from pathlib import Path


VALIDATION_SCRIPT = (
    Path(__file__).resolve().parent
    / "mlops_pipeline"
    / "04_validate_outputs.py"
)


if __name__ == "__main__":
    runpy.run_path(str(VALIDATION_SCRIPT), run_name="__main__")
