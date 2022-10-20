rsync --exclude ".vscode" \
    --exclude "datasets/" \
    --include "data_scripts/subsample/algos/iterative.py" \
    --exclude "data_scripts/subsample/algos/" \
    --exclude "data_scripts/subsample/sample_scfg/" \
    --exclude "experiments/outputs/" \
    --include "generation/*/" \
    --include "generation/scfg/utils.py" \
    --exclude "generation/" \
    --exclude  "language/" \
    --exclude  "legacy/" \
    --exclude "notebooks" \
    --exclude "repos/" \
    --exclude "server/" \
    --exclude "**/__pycache__/***" \
    --include "*" -avzh . ../structural-diversity

rsync -avzh uci-ava-s0:/home/shivag5/Documents/research/comp-gen/structural-diversity/analysis/notebooks ../structural-diversity/analysis/

rm ../structural-diversity/analysis/notebooks/results.ipynb
