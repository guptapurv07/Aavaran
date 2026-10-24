import sys
from pathlib import Path

# The project's modules live in src/ and import each other by plain name.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
