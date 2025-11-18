# main.py
# simple entry point for the retrieval-zoom project. runs the full pipeline:
#   1) cosine baseline (step1)
#   2) affine + residual mlp training (step2)
#   3) test-time export of embeddings for submission (step3)
#
# the idea is to keep the code close to how you would actually use it during the challenge: first get a sanity-check baseline, 
# then train the projector, then generate the csv for the leaderboard.

import argparse
import importlib
import sys
from pathlib import Path

# make sure we can import from src/ and see config.py
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import CFG  # noqa: E402


def run_step1():
    """
    run the cosine baseline (step1).

    this uses the raw text and image embeddings from train.npz, aligns their dimensionality with a simple pca projection
    (on the larger modality), l2-normalizes both sides and evaluates plain cosine retrieval.

    conceptually this is the “no learning” reference point: we only rely on the geometry of the pre-trained encoders.
    """
    mod = importlib.import_module("step1_cosine_baseline")
    mod.main()


def run_step2():
    """
    run the affine + residual mlp training (step2).

    this step implements the core alignment model between text and image embeddings. it:

    - builds exact (caption, image) pairs from train.npz
    - fits a closed-form ridge map x → y (text → image) in the least-squares sense
    - uses that solution to initialize a torch projector (affine or affine + small residual mlp)
    - fine-tunes the map with a loss that mixes regression and cosine alignment, as suggested in cross-modal
      alignment literature (e.g. mazzocco et al. 2023, vec2vec-style linear heads)

    the artifacts written here (affine_val_queries, affine_gallery, affine_scalers, W_ridge, projector checkpoint, etc.) are then
    reused by the test-time export step.
    """
    mod = importlib.import_module("step2_affine")
    mod.main()


def run_step3():
    """
    run the test-time export (step3).

    this step takes:

    - the test text embeddings (e.g. test.clean.npz)
    - the scaling statistics learned on train (sx_mean / sx_scale)
    - either the closed-form affine map W + b, or the fine-tuned
      projector checkpoint (affine / resmlp)

    and produces a csv in the format required by the challenge:

        id,embedding
        1,"[0.12, -0.03, 0.99, ...]"
        2,"[-0.10, -0.60, 0.70, ...]"

    ids are read from the official captions file, so that the submission matches exactly the leaderboard protocol.
    """
    mod = importlib.import_module("step3_test")
    mod.main()


def parse_args():
    parser = argparse.ArgumentParser(
        description="retrieval-zoom pipeline driver (baseline → training → submission)"
    )
    parser.add_argument(
        "--skip-step1",
        action="store_true",
        help="skip cosine baseline evaluation (step1)",
    )
    parser.add_argument(
        "--skip-step2",
        action="store_true",
        help="skip affine / resmlp training (step2)",
    )
    parser.add_argument(
        "--skip-step3",
        action="store_true",
        help="skip test export / submission generation (step3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[main] using seed={CFG.SEED}")

    if not args.skip_step1:
        print("\n[main] === step1: cosine baseline ===")
        run_step1()
    else:
        print("\n[main] >>> skipping step1 (cosine baseline)")

    if not args.skip_step2:
        print("\n[main] === step2: affine + residual mlp training ===")
        run_step2()
    else:
        print("\n[main] >>> skipping step2 (training)")

    if not args.skip_step3:
        print("\n[main] === step3: test export / submission csv ===")
        run_step3()
    else:
        print("\n[main] >>> skipping step3 (submission export)")

    print("\n[main] done.")


if __name__ == "__main__":
    main()
