#!/usr/bin/env bash
# Resumable download of one BACH shard per class from the Hugging Face mirror
# (1aurent/BACH, CC BY-NC-ND 4.0). The mirror's shards are sorted by class, so
# these four give ~27 images each of Benign, InSitu, Invasive and Normal.
# Safe to re-run: curl -C - continues a partial file.
cd "$(dirname "$0")/.." || exit 1
mkdir -p data/external && cd data/external || exit 1
mkdir -p BACH
BASE="https://huggingface.co/datasets/1aurent/BACH/resolve/main/data"
for shard in \
  train-00000-of-00015-45961313a94a1995.parquet \
  train-00005-of-00015-dbcf221bf12fae2a.parquet \
  train-00009-of-00015-991e02adb7338e11.parquet \
  train-00013-of-00015-9d1c485eacb1d332.parquet; do
  # HTTPS only, including on the CDN redirect the mirror answers with.
  until curl -sSL --fail --proto =https --proto-redir =https -C - \
      --retry 50 --retry-delay 30 --retry-all-errors \
      -o "BACH/$shard.part" "$BASE/$shard"; do
    echo "$(date '+%F %T') retrying $shard"; sleep 60
  done
  mv "BACH/$shard.part" "BACH/$shard"
  echo "$(date '+%F %T') done $shard"
done
echo "$(date '+%F %T') all shards downloaded"
