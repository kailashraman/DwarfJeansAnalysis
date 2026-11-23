"""Guards that the repo does not depend on its old on-disk location.

The SatGen prior builders must default to the sibling SatelliteDensityInference
checkout (resolved from the repo location), and no tracked script or source
file may hardcode the pre-move DJA path or the non-public SatGen_Dwarf tree.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SDI_DATA = REPO.parent / "SatelliteDensityInference" / "data"

DEAD_PATHS = (
    "/global/scratch/projects/pc_heptheory/kraman/DwarfJeansAnalysis",
    "/kraman/SatGen_Dwarf",
)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_global_builder_default_h5():
    mod = _load("build_satgen_prior_table")
    assert mod.DEFAULT_H5 == SDI_DATA / "satgen_catalogs" / "m12res8_10k_Diemer+scatter_sim.h5"
    assert mod.DEFAULT_H5.is_relative_to(REPO.parent)


def test_shmr_builder_default_inputs():
    mod = _load("build_satgen_shmr_prior_tables")
    assert mod.DEFAULT_H5 == SDI_DATA / "satgen_catalogs" / "m12res8_10k_Diemer+scatter_sim.h5"
    assert mod.DEFAULT_WEIGHT_DIR == SDI_DATA / "halo_weights_published" / "Diemer"
    assert mod.DEFAULT_WEIGHT_DIR.is_relative_to(REPO.parent)


@pytest.mark.parametrize("subdir", ["scripts", "src"])
def test_no_dead_absolute_paths_in_tracked_files(subdir):
    files = subprocess.run(
        ["git", "ls-files", subdir], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    offenders = []
    for f in files:
        try:
            text = (REPO / f).read_text(errors="ignore")
        except (FileNotFoundError, IsADirectoryError):
            continue
        offenders += [f for d in DEAD_PATHS if d in text]
    assert not offenders, f"dead absolute paths in: {sorted(set(offenders))}"


@pytest.mark.parametrize("name", ["submit_batch", "submit_reprocess"])
@pytest.mark.parametrize("cwd_rel", [".", "scripts", ".."])
def test_submit_wrapper_passes_absolute_script_and_makes_logs(name, cwd_rel, tmp_path):
    """The bash wrapper must hand sbatch an existing absolute script path from any cwd."""
    fake = tmp_path / "bin"
    fake.mkdir()
    out = tmp_path / "sbatch_args.txt"
    sb = fake / "sbatch"
    sb.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > "{out}"\n')
    sb.chmod(0o755)
    logs = REPO / "logs"
    had_logs = logs.exists()
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}
    cwd = (REPO / cwd_rel).resolve()
    script = REPO / "scripts" / f"{name}.sh"
    args = ["--array=0"] if name == "submit_reprocess" else ["segue_1"]
    try:
        subprocess.run(["bash", os.path.relpath(script, cwd), *args], cwd=cwd, env=env, check=True,
                       capture_output=True, text=True)
        passed = out.read_text().splitlines()[-1]
        assert Path(passed).is_absolute() and Path(passed).is_file()
        assert Path(passed) == script
        assert logs.is_dir()
    finally:
        if not had_logs and logs.is_dir() and not any(logs.iterdir()):
            logs.rmdir()


@pytest.mark.parametrize("name,flags", [("build_satgen_prior_table", ["--h5"]),
                                        ("build_satgen_shmr_prior_tables", ["--h5", "--weights-dir"])])
def test_builder_help_lists_input_flags(name, flags):
    res = subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), "--help"],
                         capture_output=True, text=True, check=True)
    for fl in flags:
        assert fl in res.stdout
