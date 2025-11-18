#!/usr/bin/env bash
set -euo pipefail

# download official AML competition data using the Kaggle CLI
# prerequisite: ~/.kaggle/kaggle.json must exist and have correct permissions (600)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

mkdir -p "${DATA_DIR}"

echo "[download_data] downloading aml-competition files into ${DATA_DIR}"
kaggle competitions download -c aml-competition -p "${DATA_DIR}"

echo "[download_data] unzipping archive..."
cd "${DATA_DIR}"
unzip -q -o aml-competition.zip
rm -f aml-competition.zip

echo "[download_data] done. available files:"
ls -lh
