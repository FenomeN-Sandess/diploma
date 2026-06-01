from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_basic_structure_exists() -> None:
    expected_paths = [
        "README.md",
        "pyproject.toml",
        ".gitignore",
        ".dockerignore",
        "app.py",
        "configs/dataset.yaml",
        "configs/scenarios.yaml",
        "outputs/.gitkeep",
        "src/crowd_anomaly/__init__.py",
        "src/crowd_anomaly/core/__init__.py",
        "src/crowd_anomaly/reporting.py",
        "tests/test_project_structure.py",
    ]

    for relative_path in expected_paths:
        assert (PROJECT_ROOT / relative_path).exists(), relative_path


def test_no_docs_folder_in_program_part() -> None:
    assert not (PROJECT_ROOT / "docs").exists()


def test_package_imports() -> None:
    package = import_module("crowd_anomaly")
    assert package.__version__


def test_cli_files_exist() -> None:
    cli_files = [
        "scripts/build_dataset.py",
        "scripts/build_features.py",
        "scripts/run_final_eda.py",
        "scripts/run_unsupervised_anomaly.py",
        "scripts/build_project_report.py",
        "scripts/train_forecast.py",
        "scripts/train_baseline.py",
        "scripts/run_all.py",
    ]

    for relative_path in cli_files:
        assert (PROJECT_ROOT / relative_path).is_file(), relative_path


def test_eda_notebook_was_removed_from_program_part() -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "простая_eda.ipynb"
    assert not notebook_path.exists()
