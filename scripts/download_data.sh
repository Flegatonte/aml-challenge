#!/usr/bin/env bash
set -euo pipefail

# simple helper to download the official aml competition data
# requires: kaggle cli configured with ~/.kaggle/kaggle.json

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

mkdir -p "${DATA_DIR}"

echo "[download_data] downloading aml-competition files into ${DATA_DIR}"
kaggle competitions download -c aml-competition -p "${DATA_DIR}"

cd "${DATA_DIR}"
echo "[download_data] unzipping archive..."
unzip -o aml-competition.zip
rm -f aml-competition.zip

echo "[download_data] done. available files:"
ls -lh
