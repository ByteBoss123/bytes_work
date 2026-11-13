#!/usr/bin/env bash
# Download the real public datasets used by this project, with layout and
# integrity checks so a surprise in an archive fails here, not mid-run.
#
# Usage: bash scripts/prepare_data.sh DATA_ROOT [slurp] [vox1test] [libri_test] [libri_train100]
# Sources: SLURP audio Zenodo 4274930 (CC BY-NC 4.0); SLURP annotations GitHub;
# VoxCeleb1 test wav via HF mirror ProgramComputer/voxceleb; LibriSpeech OpenSLR 12.
# URLs can be overridden with SLURP_URL, VOX1_URL, VERI_URL, LIBRI_BASE (tests use this).
set -euo pipefail
ROOT=${1:?DATA_ROOT required}
shift || true
WANT=${*:-"slurp vox1test libri_test libri_train100"}
SLURP_URL=${SLURP_URL:-https://zenodo.org/record/4274930/files/slurp_real.tar.gz}
SLURP_GIT=${SLURP_GIT:-https://github.com/pswietojanski/slurp.git}
VOX1_URL=${VOX1_URL:-https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_test_wav.zip}
# official VGG host used by torchaudio.datasets.VoxCeleb1 (fallback)
VOX1_URL2=${VOX1_URL2:-https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip}
VERI_URL=${VERI_URL:-https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt}
VERI_URL2=${VERI_URL2:-https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt}
LIBRI_BASE=${LIBRI_BASE:-https://www.openslr.org/resources/12}
LIBRI_BASE2=${LIBRI_BASE2:-https://us.openslr.org/resources/12}
mkdir -p "$ROOT"
cd "$ROOT"

have() { [[ " $WANT " == *" $1 "* ]]; }
die() { echo "ERROR: $*" >&2; exit 1; }
need_gb() {  # free space check before each download
  local avail
  avail=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  [ "$avail" -ge "$1" ] || die "need ${1} GB free in $ROOT, have ${avail} GB"
}
# stream a .tar.gz straight into tar (no archive copy on disk); try mirrors in order
fetch_tgz() {
  local dest=$1; shift
  mkdir -p "$dest"
  for url in "$@"; do
    echo "downloading $url"
    if wget -q -O - "$url" | tar -xz -C "$dest"; then return 0; fi
    echo "  failed: $url"
  done
  die "all mirrors failed for $dest"
}
# make $2 point at the directory under $1 that contains marker $3
link_to_marker() {
  local base=$1 want=$2 marker=$3 found
  [ -e "$want" ] && return 0
  found=$(find "$base" -name "$marker" -print -quit)
  [ -n "$found" ] || die "could not find '$marker' under $base after extraction"
  ln -s "$(cd "$(dirname "$found")" && pwd)" "$want"
  echo "  layout: $want -> $(dirname "$found")"
}

if have slurp; then
  [ -d slurp_repo ] || git clone -q --depth 1 "$SLURP_GIT" slurp_repo
  if [ ! -e slurp_audio/slurp_real ]; then
    need_gb 8
    fetch_tgz slurp_audio/_extract "$SLURP_URL"
    # pick any recording named in the annotations as the layout marker
    marker=$(python3 -c "import json;print(json.loads(open('slurp_repo/dataset/slurp/test.jsonl').readline())['recordings'][0]['file'])")
    mkdir -p slurp_audio
    link_to_marker slurp_audio/_extract slurp_audio/slurp_real "$marker"
  fi
  python3 - <<'PY'
import json, os
meta = json.load(open("slurp_repo/dataset/slurp/metadata.json"))
want = {f for v in meta.values() for f in v["recordings"]}
have = set(os.listdir("slurp_audio/slurp_real"))
missing = want - have
print(f"SLURP: {len(want & have)}/{len(want)} annotated recordings present")
if len(missing) > 0.01 * len(want):
    raise SystemExit(f"ERROR: {len(missing)} SLURP recordings missing, e.g. {sorted(missing)[:3]}")
PY
fi

if have vox1test; then
  mkdir -p vox1
  if [ ! -e vox1/wav ]; then
    need_gb 3
    wget -q -O vox1/vox1_test_wav.zip "$VOX1_URL" || { echo "  mirror failed: $VOX1_URL"; wget -q -O vox1/vox1_test_wav.zip "$VOX1_URL2"; } \
      || die "VoxCeleb1 test download failed from both hosts"
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).testzip()" vox1/vox1_test_wav.zip \
      || die "VoxCeleb1 zip is corrupt or not a zip"
    (cd vox1 && mkdir -p _extract && { if command -v unzip >/dev/null; then unzip -q vox1_test_wav.zip -d _extract; \
      else python3 -m zipfile -e vox1_test_wav.zip _extract; fi; } && rm vox1_test_wav.zip)
    # speaker folders look like id10270/<video>/<clip>.wav; link vox1/wav to their parent
    spk=$(find vox1/_extract -type d -name 'id1*' -print -quit)
    [ -n "$spk" ] || die "no idXXXXX speaker folders in the VoxCeleb1 archive"
    ln -s "$(cd "$(dirname "$spk")" && pwd)" vox1/wav
  fi
  if [ ! -s vox1/veri_test2.txt ]; then
    wget -q -O vox1/veri_test2.txt "$VERI_URL" || wget -q -O vox1/veri_test2.txt "$VERI_URL2" \
      || die "could not download veri_test2.txt"
  fi
  python3 - <<'PY'
import os
rows = [l.split() for l in open("vox1/veri_test2.txt") if l.strip()]
if not rows or len(rows[0]) != 3 or rows[0][0] not in "01":
    raise SystemExit("ERROR: veri_test2.txt is not a VoxCeleb trial list")
files = {r[1] for r in rows} | {r[2] for r in rows}
missing = [f for f in files if not os.path.exists(os.path.join("vox1/wav", f))]
print(f"Vox1-O: {len(rows)} trials (cleaned list = 37611), {len(files) - len(missing)}/{len(files)} files present")
if missing:
    raise SystemExit(f"ERROR: {len(missing)} trial files missing, e.g. {missing[:3]}")
PY
fi

for part in test-clean train-clean-100; do
  key=libri_test; gb=2
  [ "$part" = train-clean-100 ] && { key=libri_train100; gb=8; }
  if have $key && [ ! -d "LibriSpeech/$part" ]; then
    need_gb $gb
    fetch_tgz . "$LIBRI_BASE/$part.tar.gz" "$LIBRI_BASE2/$part.tar.gz"
    [ -d "LibriSpeech/$part" ] || die "LibriSpeech/$part missing after extraction"
  fi
  if have $key; then
    n=$(find "LibriSpeech/$part" -name '*.flac' | wc -l)
    echo "LibriSpeech $part: $n utterances (expected 2620 test-clean / 28539 train-clean-100)"
  fi
done
df -h . | tail -1
echo "done: $ROOT"
