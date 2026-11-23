"""zenodo_pack: member selection, determinism, failure modes. Synthetic tree only."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_S = importlib.util.spec_from_file_location('zenodo_pack', ROOT / 'scripts' / 'zenodo_pack.py')
zp = importlib.util.module_from_spec(_S)
_S.loader.exec_module(zp)
_REAL_DIRTY = zp.dirty_build_files


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zf, zu = _load('zenodo_fetch'), _load('zenodo_upload')


def write(path, data=b'x'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    for key in ('draco_1', 'segue 1'):                 # names may contain spaces
        for prior in ('loguniform', 'cored'):
            for f in zp.RUN_FILES:
                write(repo / 'results/production' / key / prior / f,
                      f'{key}/{prior}/{f}'.encode())
    write(repo / 'results/production/README.md', b'tracked readme')
    write(repo / 'results/production/draco_1/loguniform/.ipynb_checkpoints/x-checkpoint.log', b'junk')
    readme = write(tmp_path / 'README_DATA.md', b'# readme\n')
    monkeypatch.setattr(zp, 'git_state', lambda p: ('deadbeef', False))
    monkeypatch.setattr(zp, 'dirty_build_files', lambda paths: [])
    return dict(repo=repo, readme=readme, tmp=tmp_path)


def go(w, out, **kw):
    kw.setdefault('expect_runs', 4)
    kw.setdefault('expect_priors', ('loguniform', 'cored'))
    return zp.run(out, repo_root=w['repo'], readme_path=w['readme'], **kw)


def test_members_and_manifest(world):
    out = world['tmp'] / 'out'
    assert go(world, out) == 0
    man = json.loads((out / 'MANIFEST.json').read_text())
    with tarfile.open(out / zp.ARCHIVE_NAME) as t:
        names = t.getnames()
    assert names == sorted(names) and len(names) == 20
    assert 'results/production/segue 1/cored/summary.csv' in names
    assert not any('README.md' in n or '.ipynb_checkpoints' in n for n in names)
    assert [m['path'] for m in man['members']] == names
    for m in man['members']:
        data = (world['repo'] / m['path']).read_bytes()
        assert m['size'] == len(data) and m['sha256'] == hashlib.sha256(data).hexdigest()
    a = man['archive']
    blob = (out / zp.ARCHIVE_NAME).read_bytes()
    assert a['name'] == zp.ARCHIVE_NAME and a['n_members'] == 20 and a['size'] == len(blob)
    assert a['sha256'] == hashlib.sha256(blob).hexdigest()
    assert a['md5'] == hashlib.md5(blob).hexdigest()
    assert man['dja_git_commit'] == 'deadbeef' and man['dja_git_dirty'] is False
    assert man['created_utc']
    assert (out / 'README_DATA.md').read_bytes() == b'# readme\n'
    assert man['readme']['sha256'] == hashlib.sha256(b'# readme\n').hexdigest()


def test_normalized_tar_members(world):
    out = world['tmp'] / 'out'
    go(world, out)
    with tarfile.open(out / zp.ARCHIVE_NAME) as t:
        for m in t:
            assert (m.mtime, m.uid, m.gid, m.uname, m.gname) == (0, 0, 0, '', '')


def test_deterministic(world):
    go(world, world['tmp'] / 'o1')
    for p in (world['repo'] / 'results/production').rglob('*'):
        if p.is_file():
            p.touch()                                   # new mtimes, same content
    go(world, world['tmp'] / 'o2')
    assert ((world['tmp'] / 'o1' / zp.ARCHIVE_NAME).read_bytes()
            == (world['tmp'] / 'o2' / zp.ARCHIVE_NAME).read_bytes())


def test_missing_run_file_fails(world, capsys):
    (world['repo'] / 'results/production/draco_1/cored/run.log').unlink()
    assert go(world, world['tmp'] / 'out') == 1
    assert 'draco_1/cored: missing run.log' in capsys.readouterr().out
    assert not (world['tmp'] / 'out' / zp.ARCHIVE_NAME).exists()


def test_empty_production_fails(world):
    for d in (world['repo'] / 'results/production').iterdir():
        if d.is_dir():
            shutil.rmtree(d)
    assert go(world, world['tmp'] / 'out') == 1


def test_missing_readme_fails(world, capsys):
    world['readme'].unlink()
    assert go(world, world['tmp'] / 'out') == 1
    assert 'README_DATA.md' in capsys.readouterr().out


def test_dirty_refused(world, monkeypatch, capsys):
    monkeypatch.setattr(zp, 'git_state', lambda p: ('deadbeef', True))
    assert go(world, world['tmp'] / 'out') == 1
    assert 'modified' in capsys.readouterr().out
    assert not (world['tmp'] / 'out').exists()


def test_validation_failure_leaves_previous_build_untouched(world):
    out = world['tmp'] / 'out'
    go(world, out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    (world['repo'] / 'results/production/draco_1/cored/run.log').unlink()
    assert go(world, out) == 1                          # fails before touching out/
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_stale_manifest_removed_when_build_fails(world, monkeypatch):
    out = world['tmp'] / 'out'
    go(world, out)
    assert (out / 'MANIFEST.json').exists()

    def boom(path, files):
        raise RuntimeError('disk full')
    monkeypatch.setattr(zp, 'build_archive', boom)
    with pytest.raises(RuntimeError):
        go(world, out)
    assert not (out / 'MANIFEST.json').exists()         # never next to a new/partial archive


@pytest.mark.parametrize('rel', ['results/production/draco_1/cored/notes.txt',
                                 'results/production/draco_1/cored/summary.csv.bak',
                                 'results/production/draco_1/cored/sub/summary.csv',
                                 'results/production/segue 1/loguniform/audit.json.part'])
def test_unexpected_file_in_run_dir_fails(world, capsys, rel):
    write(world['repo'] / rel, b'stray')
    assert go(world, world['tmp'] / 'out') == 1
    assert 'not one of the expected run files' in capsys.readouterr().out
    assert not (world['tmp'] / 'out').exists()


def test_file_directly_under_key_dir_fails(world, capsys):
    write(world['repo'] / 'results/production/draco_1/stray.json', b'x')
    assert go(world, world['tmp'] / 'out') == 1
    assert 'directly under a key dir' in capsys.readouterr().out


def test_ipynb_checkpoints_ignored_at_every_level(world):
    write(world['repo'] / 'results/production/.ipynb_checkpoints/a', b'j')
    write(world['repo'] / 'results/production/draco_1/.ipynb_checkpoints/a', b'j')
    assert go(world, world['tmp'] / 'out') == 0


def test_run_dir_count_checked(world, capsys):
    assert go(world, world['tmp'] / 'out', expect_runs=312) == 1
    assert '4 run dirs' in (o := capsys.readouterr().out) and 'expected 312' in o
    shutil.rmtree(world['repo'] / 'results/production/segue 1/cored')
    assert go(world, world['tmp'] / 'out') == 1
    assert '3 run dirs' in capsys.readouterr().out
    assert not (world['tmp'] / 'out').exists()


def test_default_expect_runs_is_312(world, capsys):
    assert zp.EXPECT_RUNS == 312
    assert zp.run(world['tmp'] / 'out', repo_root=world['repo'],
                  readme_path=world['readme']) == 1          # 4 runs != 312
    assert 'expected 312' in capsys.readouterr().out


@pytest.mark.parametrize('kind', ['key', 'run', 'file'])
def test_symlinks_rejected(world, capsys, kind):
    prod = world['repo'] / 'results/production'
    elsewhere = world['tmp'] / 'elsewhere'
    if kind == 'key':
        shutil.copytree(prod / 'draco_1', elsewhere)
        shutil.rmtree(prod / 'draco_1')
        (prod / 'draco_1').symlink_to(elsewhere)
        want, runs = 'results/production/draco_1: symlink (key dirs must be real directories)', 2
    elif kind == 'run':
        shutil.copytree(prod / 'draco_1/cored', elsewhere)
        shutil.rmtree(prod / 'draco_1/cored')
        (prod / 'draco_1/cored').symlink_to(elsewhere)
        want, runs = 'results/production/draco_1/cored: symlink (run dirs must be real directories)', 3
    else:
        target = prod / 'draco_1/cored/run.log'
        target.unlink()
        target.symlink_to(prod / 'draco_1/loguniform/run.log')
        want, runs = 'results/production/draco_1/cored/run.log: symlink', 4
    # the symlink is the only problem besides the (here unavoidable) priors mismatch:
    # the run-dir count equals the expected one, so the count message cannot pass the test
    assert go(world, world['tmp'] / 'out', expect_runs=runs) == 1
    out = capsys.readouterr().out
    assert want in out
    assert 'run dirs under' not in out
    assert not (world['tmp'] / 'out').exists()


@pytest.mark.parametrize('bad', ['/abs/f', '../f', 'a/../f', 'a//f', '.', ''])
def test_unsafe_arcnames_rejected(tmp_path, bad):
    src = write(tmp_path / 'f.bin', b'abc')
    with pytest.raises(ValueError, match='unsafe'):
        zp.build_archive(tmp_path / 'x.tar.gz', {bad: src})


def test_arcnames_normalised_and_duplicates_rejected(tmp_path):
    src = write(tmp_path / 'f.bin', b'abc')
    _, mem = zp.build_archive(tmp_path / 'x.tar.gz', {'./a/./f': src})
    assert [m['path'] for m in mem] == ['a/f']
    with pytest.raises(ValueError, match='duplicate'):
        zp.build_archive(tmp_path / 'y.tar.gz', {'./a/f': src, 'a/f': src})
    assert not (tmp_path / 'y.tar.gz').exists()


def _git(d, *a):
    env = {'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t', 'GIT_COMMITTER_NAME': 't',
           'GIT_COMMITTER_EMAIL': 't@t', 'PATH': '/usr/bin:/bin'}
    return subprocess.run(['git', '-C', str(d), *a], check=True, capture_output=True, env=env)


def test_build_defining_files_must_be_committed(world, monkeypatch, capsys):
    monkeypatch.setattr(zp, 'dirty_build_files', _REAL_DIRTY)
    d = world['tmp'] / 'bd'
    for n in ('meta.json', 'script.py', 'zenodo_fetch.py', 'README_DATA.md'):
        write(d / n)
    monkeypatch.setattr(zp, 'METADATA_PATH', d / 'meta.json')
    monkeypatch.setattr(zp, '__file__', str(d / 'script.py'))
    world['readme'] = d / 'README_DATA.md'
    _git(d, 'init', '-q')
    _git(d, 'add', '.')
    _git(d, 'commit', '-q', '-m', 'x')
    assert go(world, world['tmp'] / 'o', dry_run=True) == 0
    (d / 'README_DATA.md').write_bytes(b'edited')               # modified
    assert go(world, world['tmp'] / 'o') == 1
    assert 'build-defining' in capsys.readouterr().out
    assert not (world['tmp'] / 'o').exists()
    _git(d, 'add', 'README_DATA.md')                            # staged
    assert go(world, world['tmp'] / 'o') == 1
    _git(d, 'commit', '-q', '-m', 'y')
    assert go(world, world['tmp'] / 'o', dry_run=True) == 0
    write(d / 'meta.json', b'new')
    _git(d, 'rm', '-q', '--cached', 'meta.json')                # untracked
    assert go(world, world['tmp'] / 'o') == 1
    _git(d, 'add', 'meta.json')
    _git(d, 'commit', '-q', '-m', 'z')
    (d / 'script.py').write_bytes(b'edited')                    # the packer itself
    assert go(world, world['tmp'] / 'o') == 1
    assert not (world['tmp'] / 'o').exists()
    _git(d, 'add', 'script.py')
    _git(d, 'commit', '-q', '-m', 'w')
    (d / 'zenodo_fetch.py').write_bytes(b'edited')              # the fetcher users run
    assert go(world, world['tmp'] / 'o') == 1
    assert not (world['tmp'] / 'o').exists()


def test_gitignored_build_file_is_not_clean(world, monkeypatch, capsys):
    monkeypatch.setattr(zp, 'dirty_build_files', _REAL_DIRTY)
    d = world['tmp'] / 'bd'
    for n in ('meta.json', 'script.py', 'zenodo_fetch.py', 'README_DATA.md'):
        write(d / n)
    write(d / '.gitignore', b'meta.json\n')
    monkeypatch.setattr(zp, 'METADATA_PATH', d / 'meta.json')
    monkeypatch.setattr(zp, '__file__', str(d / 'script.py'))
    world['readme'] = d / 'README_DATA.md'
    _git(d, 'init', '-q')
    _git(d, 'add', '.')
    _git(d, 'commit', '-q', '-m', 'x')
    assert _git(d, 'status', '--porcelain', '--', 'meta.json').stdout == b''   # git sees nothing
    assert go(world, world['tmp'] / 'o', dry_run=True) == 1
    assert 'meta.json' in capsys.readouterr().out
    assert zp.dirty_build_files([d / 'meta.json']) == [str(d / 'meta.json')]
    assert zp.dirty_build_files([d / 'script.py']) == []


@pytest.mark.parametrize('token', ['TODO'])
def test_readme_placeholders_refused(world, capsys, token):
    world['readme'].write_text(f'cite record {token} here\n')
    assert go(world, world['tmp'] / 'out') == 1
    out = capsys.readouterr().out
    assert 'placeholder' in out and token in out
    assert not (world['tmp'] / 'out').exists()
    world['readme'].write_text('todo and <record_id> are fine in lowercase\n')
    assert go(world, world['tmp'] / 'out', dry_run=True) == 0


@pytest.mark.parametrize('op', ['missing_prior', 'extra_prior', 'swapped'])
def test_prior_set_must_match_exactly(world, capsys, op):
    prod = world['repo'] / 'results/production'
    runs = 4
    if op == 'missing_prior':                   # draco_1 loses 'cored'; segue 1 gains a stray
        shutil.rmtree(prod / 'draco_1/cored')
        for f in zp.RUN_FILES:
            write(prod / 'segue 1/other' / f)
    elif op == 'extra_prior':                   # one extra run dir; count raised to match
        for f in zp.RUN_FILES:
            write(prod / 'draco_1/other' / f)
        runs = 5
    else:                                       # right count, wrong name
        shutil.move(prod / 'draco_1/cored', prod / 'draco_1/renamed')
    assert go(world, world['tmp'] / 'out', expect_runs=runs if op != 'missing_prior' else 4) == 1
    out = capsys.readouterr().out
    assert 'differ from the expected priors' in out
    assert 'run dirs under' not in out          # count alone would not have caught it
    assert not (world['tmp'] / 'out').exists()


def test_default_prior_set_is_the_eight_production_priors():
    assert len(zp.EXPECT_PRIORS) == 8 and 'jeffreys' in zp.EXPECT_PRIORS


def test_early_return_before_walking_tree(world, monkeypatch):
    monkeypatch.setattr(zp, 'git_state', lambda p: ('deadbeef', True))
    monkeypatch.setattr(zp, 'collect_members',
                        lambda *a, **k: pytest.fail('tree walked despite earlier problems'))
    assert go(world, world['tmp'] / 'out') == 1
    monkeypatch.setattr(zp, 'git_state', lambda p: ('deadbeef', False))
    monkeypatch.setattr(zp, 'dirty_build_files', lambda paths: ['x'])
    assert go(world, world['tmp'] / 'out') == 1
    world['readme'].unlink()
    monkeypatch.setattr(zp, 'dirty_build_files', lambda paths: [])
    assert go(world, world['tmp'] / 'out') == 1


def test_dry_run_reports_and_writes_nothing(world, capsys):
    out = world['tmp'] / 'out'
    assert go(world, out, dry_run=True) == 0
    text = capsys.readouterr().out
    total = sum(p.stat().st_size for p in (world['repo'] / 'results/production').rglob('*')
                if p.is_file() and p.name in zp.RUN_FILES
                and '.ipynb_checkpoints' not in p.parts)
    assert '20 members' in text and f'{total} bytes' in text
    assert not out.exists()
    (world['repo'] / 'results/production/draco_1/cored/run.log').unlink()
    assert go(world, out, dry_run=True) == 1            # dry run still validates
    assert not out.exists()


def test_roundtrip_pack_fetch_upload(world):
    out = world['tmp'] / 'out'
    assert go(world, out) == 0
    dest = world['tmp'] / 'dest'
    zf.fetch(files_from=out, dest=dest)
    for src in (world['repo'] / 'results/production').rglob('*'):
        if src.is_file() and src.name in zp.RUN_FILES and '.ipynb_checkpoints' not in src.parts:
            assert (dest / src.relative_to(world['repo'])).read_bytes() == src.read_bytes()
    assert not (dest / 'results/production/README.md').exists()
    assert not list(dest.rglob('.ipynb_checkpoints'))
    md5s = zu.verify_build(out)
    assert list(md5s) == [zp.ARCHIVE_NAME, 'MANIFEST.json', 'README_DATA.md']


def _make_repo(tmp_path):
    _git(tmp_path, 'init', '-q')
    (tmp_path / 'f.txt').write_text('a')
    (tmp_path / 'g.txt').write_text('g')
    _git(tmp_path, 'add', '.')
    _git(tmp_path, 'commit', '-qm', 'x')


def test_git_state_dirty_semantics(tmp_path):
    _make_repo(tmp_path)
    assert zp.git_state(tmp_path)[1] is False
    (tmp_path / 'untracked.txt').write_text('u')
    assert zp.git_state(tmp_path)[1] is False            # untracked is fine
    (tmp_path / 'f.txt').write_text('b')
    assert zp.git_state(tmp_path)[1] is True


@pytest.mark.parametrize('op', ['staged_add', 'rename', 'delete_unstaged', 'delete_staged',
                                'staged_modify'])
def test_git_state_dirty_for_staged_rename_delete(tmp_path, op):
    _make_repo(tmp_path)
    assert zp.git_state(tmp_path)[1] is False
    if op == 'staged_add':
        (tmp_path / 'new.txt').write_text('n')
        _git(tmp_path, 'add', 'new.txt')
    elif op == 'rename':
        _git(tmp_path, 'mv', 'f.txt', 'h.txt')
    elif op == 'delete_unstaged':
        (tmp_path / 'f.txt').unlink()
    elif op == 'delete_staged':
        _git(tmp_path, 'rm', '-q', 'f.txt')
    else:
        (tmp_path / 'f.txt').write_text('b')
        _git(tmp_path, 'add', 'f.txt')
    assert zp.git_state(tmp_path)[1] is True
