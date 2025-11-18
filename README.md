cross-modal alignment pipeline for the aml latent-space retrieval challenge (text → image)

this repository implements a compact 3-stage pipeline for cross-modal alignment between text and image embeddings.
the objective is to learn a projector that maps text embeddings into the image space while preserving semantic structure, enabling robust nearest-neighbor retrieval.

### step 1 — cosine baseline

a simple sanity-check baseline:

    loads raw embeddings from train.npz
    resolves dimensionality mismatch via pca on the larger modality
    l2-normalizes both sides
    evaluates cosine retrieval (mrr, recall@k)

this establishes the “no-learning” reference point to verify data integrity and initial separability.

### step 2 — affine + residual mlp projector

the core alignment model:

    builds exact (caption, image) training pairs
    fits a closed-form ridge regression map 𝑥 ↦ 𝑦
    initializes a torch projector (affine or affine + residual mlp)
    fine-tunes using a mix of mse + cosine loss
    writes all artifacts needed for test-time inference (scalers, w, checkpoint, validation embeddings)

this design is inspired by standard cross-modal alignment techniques (e.g., cca-style linear heads, lightweight adapters, and contrastive fine-tuning).

### step 3 — test-time export (submission)

generates the final leaderboard submission:

    loads test text embeddings
    standardizes them using train µ/σ
    applies either:
        - the affine ridge projection, or
        - the fine-tuned projector checkpoint
    reads official test ids
    writes the csv:

        id,embedding
        1,"[0.12, -0.03, ...]"
        2,"[-0.10, 0.67, ...]"

each row corresponds to one test caption and its predicted image-space embedding.

### usage

the dataset must not be stored in the repo due to size limits, you must download it via the kaggle api.

1. configure kaggle (first time only)

    place your kaggle token here:

    ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json

2. run the download script

    ./scripts/download_data.sh

this automatically populates:

data/train.npz
data/test.clean.npz
data/test_captions.txt

### setup
git clone https://github.com/Flegatonte/aml-challenge.git
cd aml-challenge

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

### running the pipeline
run the full workflow (baseline → training → submission)
    
    python main.py

run individual steps

    python main.py --skip-step1     # skip baseline
    python main.py --skip-step2     # skip training
    python main.py --skip-step3     # skip submission generation


