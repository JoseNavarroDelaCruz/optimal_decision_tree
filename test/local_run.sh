#!/bin/bash

NUM_CORES=20
START_IDX=1
END_IDX=140
SEED="1"

PROJECT_DIR="/Users/navarrodelacruz/Documents/GitHub/optimal_decision_tree"
JULIA_PATH="$HOME/julia-1.7.2/bin"
MPI_PATH="$(dirname $(which mpiexec))"

export PATH="$JULIA_PATH:$MPI_PATH:$PATH"

OUTPUT_DIR="$PROJECT_DIR/outputs"
mkdir -p "$OUTPUT_DIR"

DATASET_NAME="banknote"
DATASET_FOLDER="$PROJECT_DIR/data/$DATASET_NAME"

for ((i=START_IDX; i<=END_IDX; i++)); do
    dataset_file="${DATASET_NAME}_${i}.${DATASET_NAME}"
    dataset="$DATASET_FOLDER/$dataset_file"
    output_file="$OUTPUT_DIR/info-${dataset_file}-sd${SEED}-2-CMS-${NUM_CORES}.out"

    echo ">>> Running: $dataset with seed $SEED"

    mpiexec -n $NUM_CORES julia "$PROJECT_DIR/test/test.jl" 2 CF+MILP+SG "$SEED" par "$dataset" > "$output_file"

    echo ">>> Done: $dataset"
    echo ""
done

echo "All jobs complete."
