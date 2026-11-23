#!/usr/bin/env python
"""Pack the Zenodo data deposit: results/production/ as one archive.

    python scripts/zenodo_pack.py [--out-dir DIR] [--expect-runs N] [--dry-run]
                                                      (out-dir default scratch/zenodo_build)

Writes DIR/dja_production.tar.gz, DIR/MANIFEST.json and DIR/README_DATA.md
(copied from zenodo/README_DATA.md). The archive holds every file under
results/production/<lvdb_key>/<prior>/ (repo-relative paths, unpacks at the
repo root), except anything under a `.ipynb_checkpoints` directory;
results/production/README.md is git-tracked and not archived. The member list is
derived from the directory; every run dir must hold exactly the five expected
files.

Hard failures: zenodo/README_DATA.md missing; tracked files modified or staged
(untracked files are fine); scripts/zenodo_pack.py, scripts/zenodo_fetch.py,
zenodo/README_DATA.md or zenodo/metadata.json untracked (or gitignored), modified or staged (they define the record, so
the recorded commit must contain them); a run dir lacking an expected file or
holding any other file; a file directly under results/production/<key>/; a
symlinked key dir, run dir or file; a run-dir count different from --expect-runs
(default 312); a key dir whose run-dir names are not exactly the expected prior
set (--expect-priors); zenodo/README_DATA.md containing 'TODO';
no run dirs. --dry-run validates and prints the member count and
total bytes without writing anything.

The README bytes are read once, right after the git/build-file checks, and exactly
those bytes are written. After the build the git-state and build-file checks are
re-run; if HEAD moved or anything became dirty the pack fails and removes the
archive and README, leaving no MANIFEST.

Builds are deterministic (sorted members, mtime 0, uid/gid 0, gzip mtime=0), so
a rebuild is byte-identical within one Python/zlib environment.
"""

import argparse
import datetime
import gzip
import hashlib
import json
import os
import posixpath
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION = Path('results') / 'production'
ARCHIVE_NAME = 'dja_production.tar.gz'
README_NAME = 'README_DATA.md'
README_PATH = REPO_ROOT / 'zenodo' / README_NAME
METADATA_PATH = REPO_ROOT / 'zenodo' / 'metadata.json'
EXCLUDED_DIR = '.ipynb_checkpoints'
RUN_FILES = ('audit.json', 'derived.npz', 'posterior_samples.npz', 'run.log',
             'summary.csv')
CHUNK = 1 << 20
EXPECT_RUNS = 312
EXPECT_PRIORS = ('jeffreys', 'loguniform', 'satgen', 'satgen_box',
                 'satgen_shmr_danieli23_const', 'satgen_shmr_fattahi18',
                 'satgen_shmr_kim24', 'satgen_shmr_moster18')
PLACEHOLDERS = ('TODO',)  # <RECORD_ID> stays: the record id is unknown when packing


def collect_members(repo_root, expect_runs=EXPECT_RUNS, expect_priors=EXPECT_PRIORS):
    """-> ({repo-relative posix path: source Path}, [problems]).

    Every entry below results/production/<key>/<prior>/ must be one of
    RUN_FILES (outside `.ipynb_checkpoints`); symlinks, stray files directly
    under a key dir, a run-dir count != expect_runs, and a key dir whose run-dir
    names are not exactly expect_priors are problems."""
    repo_root = Path(repo_root)
    prod = repo_root / PRODUCTION
    rel = lambda p: p.relative_to(repo_root).as_posix()
    files, problems, run_dirs = {}, [], []
    keys = sorted(prod.iterdir()) if prod.is_dir() else []
    for k in keys:
        if k.name == EXCLUDED_DIR:
            continue
        if k.is_symlink():
            problems.append(f'{rel(k)}: symlink (key dirs must be real directories)')
        elif k.is_dir():
            names = set()
            for d in sorted(k.iterdir()):
                if d.name == EXCLUDED_DIR:
                    continue
                if d.is_symlink():
                    problems.append(f'{rel(d)}: symlink (run dirs must be real directories)')
                elif d.is_dir():
                    run_dirs.append(d)
                    names.add(d.name)
                else:
                    problems.append(f'{rel(d)}: unexpected file directly under a key dir')
            if names != set(expect_priors):
                problems.append(f'{rel(k)}: run dirs differ from the expected priors '
                                f'(missing {sorted(set(expect_priors) - names)}, '
                                f'unexpected {sorted(names - set(expect_priors))})')
    if not run_dirs:
        problems.append(f'no run dirs under {prod}')
    elif len(run_dirs) != expect_runs:
        problems.append(f'{len(run_dirs)} run dirs under {prod}, expected {expect_runs} '
                        '(--expect-runs)')
    for d in run_dirs:
        for name in RUN_FILES:
            if not (d / name).is_file():
                problems.append(f'{rel(d)}: missing {name}')
        for f in sorted(d.rglob('*')):
            sub = f.relative_to(d)
            if EXCLUDED_DIR in sub.parts:
                continue
            if f.is_symlink():
                problems.append(f'{rel(f)}: symlink')
            elif f.is_file():
                if sub.as_posix() in RUN_FILES:
                    files[rel(f)] = f
                else:
                    problems.append(f'{rel(f)}: not one of the expected run files')
    return files, problems


def normalize_arcname(arc):
    """Canonical archive member name; rejects absolute, '..' and '//' names."""
    if arc.startswith('/') or '//' in arc or '..' in arc.split('/'):
        raise ValueError(f'unsafe archive member name: {arc!r}')
    norm = posixpath.normpath(arc)
    if norm in ('', '.'):
        raise ValueError(f'unsafe archive member name: {arc!r}')
    return norm


class HashingWriter:
    """Write-through file wrapper computing sha256/md5 and byte count."""

    def __init__(self, fh):
        self.fh, self.sha, self.md5, self.n = fh, hashlib.sha256(), hashlib.md5(), 0

    def write(self, b):
        self.sha.update(b)
        self.md5.update(b)
        self.n += len(b)
        return self.fh.write(b)

    def flush(self):
        self.fh.flush()


class HashingReader:
    def __init__(self, fh):
        self.fh, self.sha = fh, hashlib.sha256()

    def read(self, size=-1):
        b = self.fh.read(size)
        self.sha.update(b)
        return b


def build_archive(path, files):
    """Stream files {arcname: src} into a deterministic tar.gz at path.

    Written to <path>.part and renamed on success. Returns (archive record,
    member records)."""
    norm = {}
    for arc, src in files.items():
        n = normalize_arcname(arc)
        if n in norm:
            raise ValueError(f'duplicate archive member name after normalisation: {n!r}')
        norm[n] = src
    path = Path(path)
    part = path.with_name(path.name + '.part')
    members = []
    try:
        with open(part, 'wb') as raw:
            out = HashingWriter(raw)
            gz = gzip.GzipFile(filename='', mode='wb', fileobj=out,
                               compresslevel=6, mtime=0)
            with tarfile.open(fileobj=gz, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for arc in sorted(norm):
                    src = norm[arc]
                    st0 = src.stat()
                    info = tarfile.TarInfo(arc)
                    info.size = st0.st_size
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with open(src, 'rb') as fh:
                        rd = HashingReader(fh)
                        tar.addfile(info, rd)
                    st1 = src.stat()
                    if (st1.st_size, st1.st_mtime_ns) != (st0.st_size, st0.st_mtime_ns):
                        raise RuntimeError(f'{src} changed while packing')
                    members.append({'path': arc, 'size': info.size,
                                    'sha256': rd.sha.hexdigest()})
            gz.close()
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return ({'name': path.name, 'size': out.n, 'sha256': out.sha.hexdigest(),
             'md5': out.md5.hexdigest(), 'n_members': len(members)}, members)


def git_state(path):
    """(commit, dirty) of the checkout containing path. dirty = tracked files
    modified or staged (untracked files are ignored)."""
    path = str(path)
    try:
        c = subprocess.run(['git', '-C', path, 'rev-parse', 'HEAD'],
                           capture_output=True, text=True, check=True).stdout.strip()
        d = subprocess.run(['git', '-C', path, 'status', '--porcelain',
                            '--untracked-files=no'],
                           capture_output=True, text=True, check=True).stdout.strip()
        return c, bool(d)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def dirty_build_files(paths):
    """Those of `paths` (the files that define the build) that are untracked
    (including gitignored), modified or staged in their own checkout, or whose git state is unknown."""
    bad = []
    for p in map(Path, paths):
        try:
            subprocess.run(['git', '-C', str(p.parent), 'ls-files', '--error-unmatch',
                            '--', p.name],
                           capture_output=True, text=True, check=True)
            d = subprocess.run(['git', '-C', str(p.parent), 'status', '--porcelain',
                                '--', p.name],
                               capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            d = 'unknown'
        if d:
            bad.append(str(p))
    return bad


def file_digests(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(CHUNK), b''):
            sha.update(b)
            md5.update(b)
    return sha.hexdigest(), md5.hexdigest()


def changed_during_build(repo_root, readme_path, commit, dirty):
    """Re-run the start-of-run git and build-file checks after the build; a
    message if HEAD moved or anything became dirty, else None."""
    c2, d2 = git_state(repo_root)
    if (c2, d2) != (commit, dirty):
        return (f'git state changed during the build (was {commit}, dirty={dirty}; '
                f'now {c2}, dirty={d2})')
    bad = dirty_build_files([Path(__file__).resolve(),
                             Path(__file__).resolve().with_name('zenodo_fetch.py'),
                             readme_path, METADATA_PATH])
    if bad:
        return f'build-defining file(s) untracked or differ from HEAD after the build: {bad}'
    return None


def run(out_dir, repo_root=None, readme_path=README_PATH, expect_runs=EXPECT_RUNS,
        dry_run=False, expect_priors=EXPECT_PRIORS):
    repo_root = Path(repo_root or REPO_ROOT)
    readme_path = Path(readme_path)
    problems = []
    if not readme_path.is_file():
        problems.append(f'{README_NAME} not found at {readme_path} (it ships in the record)')
    commit, dirty = git_state(repo_root)
    if commit is None:
        problems.append(f'cannot determine the git state of {repo_root}')
    elif dirty:
        problems.append('checkout has modified/staged tracked files; commit or stash them')
    if commit is not None:
        bad = dirty_build_files([Path(__file__).resolve(),
                                 Path(__file__).resolve().with_name('zenodo_fetch.py'),
                                 readme_path, METADATA_PATH])
        if bad:
            problems.append(f'build-defining file(s) untracked or differ from HEAD: '
                            f'{bad}; commit them first')
    readme_bytes = None
    if not problems:
        # Read once, right after the git/build-file checks; these exact bytes are
        # what ships, so a commit made during the build cannot swap in a newer one.
        readme_bytes = readme_path.read_bytes()
        left = [t for t in PLACEHOLDERS if t in readme_bytes.decode(errors='replace')]
        if left:
            problems.append(f'{README_NAME} still contains placeholder(s) {left}; '
                            'fill them in before packing')
    if not problems:
        files, problems = collect_members(repo_root, expect_runs, expect_priors)
    if problems:
        print(f'FAIL: {len(problems)} problem(s):')
        for q in problems[:50]:
            print('  ', q)
        return 1
    if dry_run:
        print(f'dry run OK: {len(files)} members, '
              f'{sum(f.stat().st_size for f in files.values())} bytes (nothing written)')
        return 0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / 'MANIFEST.json'
    mpath.unlink(missing_ok=True)       # no stale manifest next to a new archive
    print(f'building {ARCHIVE_NAME} ({len(files)} files) ...', flush=True)
    rec, members = build_archive(out_dir / ARCHIVE_NAME, files)
    late = changed_during_build(repo_root, readme_path, commit, dirty)
    if late:
        for n in (ARCHIVE_NAME, README_NAME):
            (out_dir / n).unlink(missing_ok=True)
        print(f'FAIL: {late}; nothing was written (rerun the pack)')
        return 1
    (out_dir / README_NAME).write_bytes(readme_bytes)
    r_sha, r_md5 = hashlib.sha256(readme_bytes).hexdigest(), hashlib.md5(readme_bytes).hexdigest()
    manifest = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'dja_git_commit': commit, 'dja_git_dirty': dirty,
        'readme': {'name': README_NAME, 'size': len(readme_bytes),
                   'sha256': r_sha, 'md5': r_md5},
        'archive': rec,
        'members': members,
    }
    mpath.write_text(json.dumps(manifest, separators=(',', ':')))
    print(f'wrote {mpath}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out-dir', default=str(REPO_ROOT / 'scratch' / 'zenodo_build'),
                    help='output directory (default: scratch/zenodo_build)')
    ap.add_argument('--expect-runs', type=int, default=EXPECT_RUNS,
                    help=f'required number of run dirs (default: {EXPECT_RUNS})')
    ap.add_argument('--expect-priors', default=','.join(EXPECT_PRIORS),
                    help='comma-separated run-dir names every key dir must hold exactly '
                         '(default: the eight production priors)')
    ap.add_argument('--dry-run', action='store_true',
                    help='validate and report member count/bytes; write nothing')
    args = ap.parse_args(argv)
    return run(args.out_dir, expect_runs=args.expect_runs, dry_run=args.dry_run,
               expect_priors=tuple(args.expect_priors.split(',')))


if __name__ == '__main__':
    sys.exit(main())
