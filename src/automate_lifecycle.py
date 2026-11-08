import runpy
from pathlib import Path


LIFECYCLE_SCRIPT = (
    Path(__file__).resolve().parent
    / "mlops_pipeline"
    / "08_automate_lifecycle.py"
)


if __name__ == "__main__":
    runpy.run_path(str(LIFECYCLE_SCRIPT), run_name="__main__")
