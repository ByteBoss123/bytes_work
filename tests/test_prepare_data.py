"""prepare_data.sh against a local HTTP server serving archives with
non-default layouts: it must find the data, link it into the expected
paths, and fail loudly when files are missing."""
import functools
import sys
import http.server
import shutil
import subprocess
import tarfile
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from fixtures import make_slurp  # noqa: E402


@pytest.fixture()
def server(tmp_path):
    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(srv_dir))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield srv_dir, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _git_repo(src_ann: Path, dest: Path):
    (dest / "dataset").mkdir(parents=True)
    shutil.copytree(src_ann, dest / "dataset" / "slurp")
    for cmd in (["git", "init", "-q"], ["git", "add", "."],
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"]):
        subprocess.run(cmd, cwd=dest, check=True)


def _env(base, srv):
    import os

    return dict(os.environ, no_proxy="127.0.0.1,localhost", NO_PROXY="127.0.0.1,localhost",
                SLURP_URL=f"{base}/slurp_real.tar.gz", SLURP_GIT=str(srv / "slurp_git"),
                VOX1_URL=f"{base}/missing.zip", VOX1_URL2=f"{base}/vox1_test_wav.zip", VERI_URL=f"{base}/veri_test2.txt",
                VERI_URL2=f"{base}/missing.txt", LIBRI_BASE=f"{base}/nope", LIBRI_BASE2=base)


def test_prepare_data_odd_layouts(tmp_path, server):
    srv, base = server
    ann, audio = make_slurp(tmp_path / "raw", n_speakers=2, utts_per_speaker=4)
    _git_repo(ann, srv / "slurp_git")
    with tarfile.open(srv / "slurp_real.tar.gz", "w:gz") as tf:  # nested, non-default folder names
        tf.add(audio, arcname="release/v1/real_audio")
    vox = tmp_path / "vox"
    for spk, clip in (("id10270", "a/00001.wav"), ("id10271", "b/00002.wav")):
        p = vox / "deep" / "wav" / spk / clip
        p.parent.mkdir(parents=True)
        sf.write(p, np.zeros(16000, dtype=np.float32), 16000)
    with zipfile.ZipFile(srv / "vox1_test_wav.zip", "w") as z:
        for p in vox.rglob("*.wav"):
            z.write(p, p.relative_to(vox))
    (srv / "veri_test2.txt").write_text("0 id10270/a/00001.wav id10271/b/00002.wav\n")
    lib = tmp_path / "lib" / "LibriSpeech" / "test-clean" / "1" / "2"
    lib.mkdir(parents=True)
    sf.write(lib / "1-2-0000.flac", np.zeros(16000, dtype=np.float32), 16000)
    (lib / "1-2.trans.txt").write_text("1-2-0000 HELLO\n")
    with tarfile.open(srv / "test-clean.tar.gz", "w:gz") as tf:
        tf.add(tmp_path / "lib" / "LibriSpeech", arcname="LibriSpeech")

    data = tmp_path / "data"
    out = subprocess.run(["bash", str(ROOT / "scripts" / "prepare_data.sh"), str(data), "slurp", "vox1test",
                          "libri_test"], capture_output=True, text=True, env=_env(base, srv))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "SLURP: 16/16 annotated recordings present" in out.stdout
    assert (data / "slurp_audio" / "slurp_real").resolve().name == "real_audio"
    assert (data / "vox1" / "wav" / "id10270" / "a" / "00001.wav").exists()
    assert "LibriSpeech test-clean: 1 utterances" in out.stdout  # first mirror 404s, second works

    # idempotent second run, no downloads needed
    out2 = subprocess.run(["bash", str(ROOT / "scripts" / "prepare_data.sh"), str(data), "slurp", "vox1test"],
                          capture_output=True, text=True, env=_env(base, srv))
    assert out2.returncode == 0, out2.stderr


def test_prepare_data_fails_on_missing_trial_files(tmp_path, server):
    srv, base = server
    vox = tmp_path / "vox" / "wav" / "id10270" / "a"
    vox.mkdir(parents=True)
    sf.write(vox / "00001.wav", np.zeros(1600, dtype=np.float32), 16000)
    with zipfile.ZipFile(srv / "vox1_test_wav.zip", "w") as z:
        z.write(vox / "00001.wav", "wav/id10270/a/00001.wav")
    (srv / "veri_test2.txt").write_text("1 id10270/a/00001.wav id10270/a/99999.wav\n")
    out = subprocess.run(["bash", str(ROOT / "scripts" / "prepare_data.sh"), str(tmp_path / "d"), "vox1test"],
                         capture_output=True, text=True, env=_env(base, srv))
    # VoxCeleb problems are reported but must not abort the other datasets
    assert out.returncode == 0, out.stderr
    assert "trial files missing" in out.stderr + out.stdout and "VoxCeleb1 unavailable" in out.stdout
