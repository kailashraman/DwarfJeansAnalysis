"""zenodo_fetch: checksum verification, path traversal, overwrite policy. Offline."""
import hashlib
import importlib.util
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_S = importlib.util.spec_from_file_location('zenodo_fetch', ROOT / 'scripts' / 'zenodo_fetch.py')
zf = importlib.util.module_from_spec(_S)
_S.loader.exec_module(zf)


def make_tar(path, members, mode='w'):
    with tarfile.open(path, mode) as t:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))


def digests(path):
    b = path.read_bytes()
    return hashlib.sha256(b).hexdigest(), hashlib.md5(b).hexdigest()


def make_record(d, members=None, name='a.tar'):
    d.mkdir()
    members = members or {'results/production/x/f.txt': b'hello'}
    make_tar(d / name, members)
    sha, md5 = digests(d / name)
    man = {'archive': {'name': name, 'size': 1, 'sha256': sha, 'md5': md5},
           'members': [{'path': k, 'size': len(v),
                        'sha256': hashlib.sha256(v).hexdigest()} for k, v in members.items()]}
    (d / 'MANIFEST.json').write_text(json.dumps(man))
    return man


def test_fetch_extracts(tmp_path, capsys):
    make_record(tmp_path / 'rec')
    zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest')
    assert (tmp_path / 'dest/results/production/x/f.txt').read_bytes() == b'hello'


def test_manifest_checksum_mismatch(tmp_path):
    d = tmp_path / 'rec'
    man = make_record(d)
    man['archive']['sha256'] = '0' * 64
    (d / 'MANIFEST.json').write_text(json.dumps(man))
    with pytest.raises(zf.FetchError, match='MANIFEST'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')
    assert not (tmp_path / 'dest/results').exists()


class FakeResp:
    def __init__(self, js=None, content=b''):
        self._js, self.content = js, content

    def raise_for_status(self):
        pass

    def json(self):
        return self._js

    def iter_content(self, n):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class FakeSession:
    def __init__(self, record, md5_override=None):
        self.record, self.md5_override = record, md5_override

    def get(self, url, **kw):
        if url.endswith('/api/records/42'):
            files = []
            for p in self.record.iterdir():
                md5 = digests(p)[1]
                if self.md5_override and p.name == 'a.tar':
                    md5 = self.md5_override
                files.append({'key': p.name, 'checksum': f'md5:{md5}',
                              'links': {'self': f'https://z/files/{p.name}'}})
            return FakeResp({'files': files})
        return FakeResp(content=(self.record / url.rsplit('/', 1)[1]).read_bytes())


def test_zenodo_md5_mismatch(tmp_path):
    make_record(tmp_path / 'rec')
    s = FakeSession(tmp_path / 'rec', md5_override='1' * 32)
    with pytest.raises(zf.FetchError, match='Zenodo checksum'):
        zf.fetch('42', 'https://zenodo.org', dest=tmp_path / 'dest', session=s)


def test_fetch_via_api(tmp_path):
    make_record(tmp_path / 'rec')
    zf.fetch('42', 'https://zenodo.org', dest=tmp_path / 'dest',
             session=FakeSession(tmp_path / 'rec'))
    assert (tmp_path / 'dest/results/production/x/f.txt').exists()


@pytest.mark.parametrize('bad', ['../evil.txt', '/abs/evil.txt', 'a/../../evil.txt'])
def test_path_traversal_rejected(tmp_path, bad):
    d = tmp_path / 'rec'
    make_record(d, {bad: b'x'})
    with pytest.raises(zf.FetchError, match='unsafe path'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')
    assert not (tmp_path / 'evil.txt').exists()
    assert not (tmp_path / 'dest').exists() or not any((tmp_path / 'dest').rglob('*.txt'))


def test_refuses_differing_overwrite(tmp_path):
    make_record(tmp_path / 'rec')
    target = tmp_path / 'dest/results/production/x/f.txt'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'local edit')
    with pytest.raises(zf.FetchError, match='differ'):
        zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest')
    assert target.read_bytes() == b'local edit'
    zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest', force=True)
    assert target.read_bytes() == b'hello'


def test_identical_existing_file_ok(tmp_path):
    make_record(tmp_path / 'rec')
    target = tmp_path / 'dest/results/production/x/f.txt'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'hello')
    zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest')


def test_tmp_dir_under_dest_not_tmpdir(tmp_path, monkeypatch):
    make_record(tmp_path / 'rec')
    elsewhere = tmp_path / 'sys_tmp'
    elsewhere.mkdir()
    monkeypatch.setattr(zf.tempfile, 'tempdir', str(elsewhere))
    seen = []
    real = zf.download

    def spy(url, dest, session):
        seen.append(dest)
        real(url, dest, session)
    monkeypatch.setattr(zf, 'download', spy)
    zf.fetch('42', 'https://zenodo.org', dest=tmp_path / 'dest',
             session=FakeSession(tmp_path / 'rec'))
    assert seen and all(tmp_path / 'dest' in p.parents for p in seen)
    assert list(elsewhere.iterdir()) == []
    assert not list((tmp_path / 'dest').glob('.zenodo_fetch_*'))      # cleaned up


def test_tmp_option(tmp_path):
    make_record(tmp_path / 'rec')
    seen = []
    zf_download = zf.download

    def spy(url, dest, session):
        seen.append(dest)
        zf_download(url, dest, session)
    zf.download = spy
    try:
        zf.fetch('42', 'https://zenodo.org', dest=tmp_path / 'dest',
                 session=FakeSession(tmp_path / 'rec'), tmp=tmp_path / 'scr')
    finally:
        zf.download = zf_download
    assert seen and all(tmp_path / 'scr' in p.parents for p in seen)


class NoMd5Session(FakeSession):
    def __init__(self, record, drop):
        super().__init__(record)
        self.drop = drop

    def get(self, url, **kw):
        r = super().get(url, **kw)
        if url.endswith('/api/records/42'):
            for f in r._js['files']:
                if f['key'] == self.drop:
                    f['checksum'] = ''
        return r


@pytest.mark.parametrize('drop', ['MANIFEST.json', 'a.tar'])
def test_missing_record_checksum_fails(tmp_path, drop):
    make_record(tmp_path / 'rec')
    with pytest.raises(zf.FetchError, match='no checksum'):
        zf.fetch('42', 'https://zenodo.org', dest=tmp_path / 'dest',
                 session=NoMd5Session(tmp_path / 'rec', drop))
    assert not (tmp_path / 'dest/results').exists()


def test_manifest_md5_mismatch_vs_record_fails(tmp_path):
    make_record(tmp_path / 'rec')

    class S(FakeSession):
        def get(self, url, **kw):
            r = super().get(url, **kw)
            if url.endswith('/api/records/42'):
                for f in r._js['files']:
                    if f['key'] == 'MANIFEST.json':
                        f['checksum'] = 'md5:' + '2' * 32
            return r
    with pytest.raises(zf.FetchError, match='MANIFEST.json: md5'):
        zf.fetch('42', 'https://zenodo.org', dest=tmp_path / 'dest',
                 session=S(tmp_path / 'rec'))


def test_requires_tarfile_data_filter(tmp_path, monkeypatch):
    make_record(tmp_path / 'rec')
    monkeypatch.delattr(zf.tarfile, 'data_filter')
    with pytest.raises(zf.FetchError, match='data_filter'):
        zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest')
    assert not (tmp_path / 'dest').exists()


def test_corrupt_archive_is_fetch_error(tmp_path, capsys):
    d = tmp_path / 'rec'
    make_record(d)
    (d / 'a.tar').write_bytes(b'not a tar' * 100)
    sha, md5 = digests(d / 'a.tar')
    man = json.loads((d / 'MANIFEST.json').read_text())
    man['archive'].update(sha256=sha, md5=md5)
    (d / 'MANIFEST.json').write_text(json.dumps(man))
    with pytest.raises(zf.FetchError, match='cannot extract'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')
    assert zf.main(['--files-from', str(d), '--dest', str(tmp_path / 'dest2')]) == 1
    assert 'ERROR:' in capsys.readouterr().err


def test_extracted_mtime_is_extraction_time(tmp_path):
    import time
    make_record(tmp_path / 'rec')
    t0 = time.time()
    zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest')
    assert (tmp_path / 'dest/results/production/x/f.txt').stat().st_mtime >= t0 - 1


# ---- archive-level validation: manifest is clean, the tar is not (digests recomputed)

GOOD = 'results/production/x/f.txt'


def tarinfo(name, kind=tarfile.REGTYPE, linkname='', data=b''):
    ti = tarfile.TarInfo(name)
    ti.type, ti.linkname = kind, linkname
    ti.size = len(data) if kind == tarfile.REGTYPE else 0
    return ti, data


def make_hostile_record(d, extra, manifest_members=None):
    """Tar = the GOOD file plus `extra` [(TarInfo, data)]; MANIFEST lists only
    manifest_members (default {GOOD: b'hello'}), digests consistent with the tar."""
    d.mkdir()
    with tarfile.open(d / 'a.tar', 'w') as t:
        ti, data = tarinfo(GOOD, data=b'hello')
        t.addfile(ti, io.BytesIO(data))
        for ti, data in extra:
            t.addfile(ti, io.BytesIO(data) if data else None)
    sha, md5 = digests(d / 'a.tar')
    mm = manifest_members if manifest_members is not None else {GOOD: b'hello'}
    man = {'archive': {'name': 'a.tar', 'size': 1, 'sha256': sha, 'md5': md5},
           'members': [{'path': k, 'size': len(v), 'sha256': hashlib.sha256(v).hexdigest()}
                       for k, v in mm.items()]}
    (d / 'MANIFEST.json').write_text(json.dumps(man))


@pytest.mark.parametrize('extra,match', [
    ([tarinfo('../evil.txt', data=b'x')], 'unsafe path'),
    ([tarinfo('/abs/evil.txt', data=b'x')], 'unsafe path'),
    ([tarinfo('results/production/x/ln', tarfile.SYMTYPE, '../../../../outside')],
     'not a regular file'),
    ([tarinfo('results/production/x/ln', tarfile.SYMTYPE, '/etc/passwd')],
     'not a regular file'),
    ([tarinfo('results/production/x/hl', tarfile.LNKTYPE, '/etc/passwd')],
     'not a regular file'),
    ([tarinfo('results/production/x/hl', tarfile.LNKTYPE, '../../../outside')],
     'not a regular file'),
    ([tarinfo('results/production/x/extra.txt', data=b'x')], 'differ from MANIFEST'),
    ([tarinfo('other/evil.txt', data=b'x')], 'outside results/production'),
    ([tarinfo(GOOD, data=b'hello')], 'duplicate archive member'),
])
def test_hostile_archive_with_clean_manifest_rejected(tmp_path, extra, match):
    d = tmp_path / 'rec'
    make_hostile_record(d, extra)
    dest = tmp_path / 'dest'
    with pytest.raises(zf.FetchError, match=match):
        zf.fetch(files_from=d, dest=dest)
    assert not list(dest.rglob('*'))                    # nothing extracted
    assert not (tmp_path / 'evil.txt').exists() and not (tmp_path / 'outside').exists()


def test_symlink_member_named_like_manifest_member_rejected(tmp_path):
    d = tmp_path / 'rec'
    d.mkdir()
    with tarfile.open(d / 'a.tar', 'w') as t:
        ti, _ = tarinfo(GOOD, tarfile.SYMTYPE, '../../../../outside')
        t.addfile(ti)
    sha, md5 = digests(d / 'a.tar')
    (d / 'MANIFEST.json').write_text(json.dumps({
        'archive': {'name': 'a.tar', 'size': 1, 'sha256': sha, 'md5': md5},
        'members': [{'path': GOOD, 'size': 5, 'sha256': hashlib.sha256(b'hello').hexdigest()}]}))
    with pytest.raises(zf.FetchError, match='not a regular file'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')


def test_manifest_member_missing_from_tar_rejected(tmp_path):
    d = tmp_path / 'rec'
    make_hostile_record(d, [], manifest_members={GOOD: b'hello',
                                                 'results/production/x/g.txt': b'g'})
    with pytest.raises(zf.FetchError, match='1 missing'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')


def test_manifest_path_outside_prefix_rejected(tmp_path):
    d = tmp_path / 'rec'
    make_record(d, {'data/x/f.txt': b'hello'})
    with pytest.raises(zf.FetchError, match='outside results/production'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')


def test_member_sha256_verified_after_extraction(tmp_path):
    d = tmp_path / 'rec'
    make_record(d)
    man = json.loads((d / 'MANIFEST.json').read_text())
    man['members'][0]['sha256'] = '0' * 64              # archive-level digests still match
    (d / 'MANIFEST.json').write_text(json.dumps(man))
    with pytest.raises(zf.FetchError, match='sha256 differs'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')


def test_existing_directory_at_target_is_conflict_even_with_force(tmp_path):
    make_record(tmp_path / 'rec')
    (tmp_path / 'dest' / GOOD).mkdir(parents=True)
    for force in (False, True):
        with pytest.raises(zf.FetchError, match='not regular files'):
            zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest', force=force)


def test_dangling_symlink_at_target_is_conflict_even_with_force(tmp_path):
    make_record(tmp_path / 'rec')
    target = tmp_path / 'dest' / GOOD
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path / 'nowhere')
    for force in (False, True):
        with pytest.raises(zf.FetchError, match='not regular files'):
            zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest', force=force)
    assert not (tmp_path / 'nowhere').exists()


def test_oserror_during_extraction_is_error(tmp_path, capsys):
    make_record(tmp_path / 'rec')
    blocker = tmp_path / 'dest' / 'results' / 'production' / 'x'
    blocker.parent.mkdir(parents=True)
    blocker.write_bytes(b'a file where a directory is needed')
    assert zf.main(['--files-from', str(tmp_path / 'rec'), '--dest', str(tmp_path / 'dest')]) == 1
    assert 'ERROR:' in capsys.readouterr().err
    with pytest.raises(zf.FetchError, match='cannot extract'):
        zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest')


@pytest.mark.parametrize('text', ['not json', '[]', '{}', '{"archive": {}, "members": []}',
                                  '{"archive": {"name": "a.tar", "size": 1, "sha256": "x", '
                                  '"md5": "y"}}',
                                  '{"archive": {"name": "a.tar", "size": 1, "sha256": "x", '
                                  '"md5": "y"}, "members": [{"path": "p"}]}',
                                  '{"archive": {"name": "../a.tar", "size": 1, "sha256": "x", '
                                  '"md5": "y"}, "members": []}'])
def test_malformed_manifest_is_fetch_error(tmp_path, text, capsys):
    d = tmp_path / 'rec'
    make_record(d)
    (d / 'MANIFEST.json').write_text(text)
    with pytest.raises(zf.FetchError, match='MANIFEST.json'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')
    assert zf.main(['--files-from', str(d), '--dest', str(tmp_path / 'dest2')]) == 1
    assert 'ERROR:' in capsys.readouterr().err


def test_force_does_not_write_through_hardlink(tmp_path):
    make_record(tmp_path / 'rec')
    target = tmp_path / 'dest' / GOOD
    target.parent.mkdir(parents=True)
    target.write_bytes(b'local edit')
    other = tmp_path / 'other_tree_link'
    os.link(target, other)                              # real production files have nlink=2
    zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest', force=True)
    assert target.read_bytes() == b'hello'
    assert other.read_bytes() == b'local edit'          # the other link is untouched
    assert target.stat().st_nlink == 1


def test_identical_readonly_existing_file_left_untouched(tmp_path):
    make_record(tmp_path / 'rec')
    target = tmp_path / 'dest' / GOOD
    target.parent.mkdir(parents=True)
    target.write_bytes(b'hello')
    os.utime(target, (1000, 1000))
    target.chmod(0o444)
    ino = target.stat().st_ino
    for force in (False, True):
        zf.fetch(files_from=tmp_path / 'rec', dest=tmp_path / 'dest', force=force)
        st = target.stat()
        assert (st.st_mtime, st.st_ino) == (1000, ino)
        assert target.read_bytes() == b'hello'


def test_manifest_path_checked_before_anything_outside_dest_is_read(tmp_path, monkeypatch):
    d = tmp_path / 'rec'
    make_record(d, {'../victim': b'x'})                 # archive is clean vs the manifest
    victim = tmp_path / 'victim'
    victim.write_bytes(b'precious')
    seen = []
    real = zf.file_digests
    monkeypatch.setattr(zf, 'file_digests', lambda p: (seen.append(Path(p)), real(p))[1])
    with pytest.raises(zf.FetchError, match='unsafe path'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')     # no --force: conflict check hashes targets
    assert all(p.resolve() != victim.resolve() for p in seen)
    assert victim.read_bytes() == b'precious'


def test_data_filter_blocks_symlink_member_when_validation_stubbed(tmp_path, monkeypatch):
    d = tmp_path / 'rec'
    d.mkdir()
    with tarfile.open(d / 'a.tar', 'w') as t:
        ti, _ = tarinfo(GOOD, tarfile.SYMTYPE, '../../../../outside')
        t.addfile(ti)
    sha, md5 = digests(d / 'a.tar')
    (d / 'MANIFEST.json').write_text(json.dumps({
        'archive': {'name': 'a.tar', 'size': 1, 'sha256': sha, 'md5': md5},
        'members': [{'path': GOOD, 'size': 5, 'sha256': hashlib.sha256(b'hello').hexdigest()}]}))
    monkeypatch.setattr(zf, 'validate_archive', lambda *a, **k: None)
    with pytest.raises(zf.FetchError, match='cannot extract'):
        zf.fetch(files_from=d, dest=tmp_path / 'dest')
    assert not (tmp_path / 'dest' / GOOD).is_symlink()
    assert not (tmp_path / 'outside').exists()
