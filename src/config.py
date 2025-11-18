# config.py — single source of truth for all pipeline parameters.
# this file centralizes every setting used across data loading, training, csls re-ranking, and final submission. keeping everything
# here makes experiments reproducible and avoids silent drift in hyperparameters across different steps of the pipeline.

from types import SimpleNamespace as NS

# choose which step2 run will be referenced by downstream stages
# this acts as a pointer to a specific trained projector.
RUN = "artifacts/affine_alpha_0.03"

# overwrite the run if using the mlp-enhanced projector.
RUN = "artifacts/affine_resmlp_alpha_0.03"   # ← this is the correct one for the resmlp experiments

CFG = NS(
    SEED = 42,  # fixed seed for full reproducibility across numpy, sklearn, torch

    PATHS = NS(
        TRAIN_NPZ     = "data/train/train.npz",          # training embeddings (text + images)
        TEST_NPZ      = "data/test/test.clean.npz",     # test-time text embeddings
        TEST_IDS_CSV  = "data/test/captions.txt",  # official id list for leaderboard alignment
        ARTIFACTS     = "artifacts",               # root folder for all outputs
    ),

    # step1: cosine baseline (optional sanity check)
    STEP1 = NS(
        DEVICE     = "cuda",
        VAL_FRAC   = 0.10,    # fraction of samples reserved for validation
        EVAL_BATCH = 2048,    # evaluation minibatch size for mrr/recall
        OUT_DIR    = "artifacts/cosine_baseline",
    ),

    # step2: main training stage (affine mapping or affine + residual mlp)
    STEP2 = NS(
        DEVICE        = "cuda",
        VAL_FRAC      = 0,           # validation split not used here (step2 handles eval explicitly)
        RIDGE_ALPHA   = 3e-2,        # ridge regularization coefficient for closed-form initialization
        EPOCHS        = 100,         # number of fine-tuning epochs for the projector
        BATCH_SIZE    = 4096,        # large batches stabilize cosine/mse joint loss
        LR            = 5e-4,        # adamw base learning rate
        WEIGHT_DECAY  = 5e-3,        # mild l2 penalty improves generalization
        EVAL_BATCH    = 2048,
        SAVE_RUN_SUBDIR = True,      # store each run in a separate directory for versioning

        # mlp configuration: these control the optional residual nonlinearity
        USE_MLP     = True,     # true = use residual mlp head; false = pure affine map
        MLP_HIDDEN  = 1024,     # hidden width inside the mlp blocks
        MLP_LAYERS  = 2,        # number of residual mlp layers
        MLP_DROPOUT = 0.15,     # dropout for regularization

        # initialization + loss objectives
        # "wp" = whitened-procrustes (aligned with the theory from 2311.00664)
        INIT        = "wp",          # weight init strategy: "ridge" or "wp"
        LOSS        = "info_nce",    # retrieval-oriented objective (contrastive)
        TEMP        = 0.07,          # temperature parameter used in infoNCE
        MIX_MSE     = 0.3,           # if using "mix", weight of mse term
        MIX_COS     = 0.7,           # if using "mix", weight of cosine embedding loss

        # training quality-of-life parameters
        GRAD_ACC    = 1,      # gradient accumulation steps
        AMP         = True,   # automatic mixed precision for faster training on cuda

        # anti-hubness mechanisms (optional preprocessing)
        # helps mitigate dominance of high-frequency vectors in nearest-neighbor retrieval.
        POWER_NORM_P   = 0.5,    # exponent for power normalization
        REMOVE_TOP_PCS = 2,      # number of top principal components to drop

        # self-learning loop (mnn + csls refinement)
        # disabled here, but this implements iterative pseudo-pair generation +
        # refitting (common in bilingual lexicon induction literature).
        SELF_LEARN = NS(
            ENABLE       = False,
            ITERS        = 2,
            K_CSLS       = 50,
            RG_SUBSET    = 100_000,
            MIN_PAIRS    = 50_000,
            EPOCHS_REFIT = 6,
        ),
    ),

    # step3 used as configuration for test-time export.
    step3 = NS(
        OUT_DIR     = "artifacts/test_pred",
        CSV_NAME    = "submission.csv",

        TEST_NPZ    = "data/test.clean.npz",
        TEST_IDS_CSV= "data/captions.txt",

        # mode 1: explicit affine mapping using W + bias
        USE_AFFINE_WB = False,
        SCALERS_NPZ   = f"{RUN}/affine_scalers.npz",
        W_NUMPY       = f"{RUN}/W_ridge.npy",
        BIAS_NUMPY    = f"{RUN}/bias_zero.npy",

        # mode 2: load full projector (affine or resmlp) and apply forward()
        USE_MODEL_PT  = True,
        MODEL_PT      = f"{RUN}/resmlp_projector.pt",
        HEAD          = "resmlp",
        DEVICE        = "cuda",
    ),

    # evaluation config (mrr @ top-k) used for ablation/testing
    EVAL = NS(
        QUERIES    = f"{RUN}/affine_val_queries.npy",
        GALLERY    = f"{RUN}/affine_gallery.npy",
        GT         = f"{RUN}/val_gt.npy",

        TOPK       = 100,
        BATCH_SIZE = 100,
        TRIALS     = 20,

        USE_CSLS   = False,
        CSLS_DIR   = "artifacts/csls_high",
    ),
)
