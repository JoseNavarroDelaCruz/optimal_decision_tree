#!/bin/bash
#SBATCH --job-name=ODT_MultiData
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --array=1-2
#SBATCH --output=$HOME/optimal_decision_tree/outputs/info-%x-%a.out
#SBATCH --error=$HOME/optimal_decision_tree/outputs/info-%x-%a.err
#SBATCH --partition=amd_2021
#SBATCH --mail-type=ALL
#SBATCH --mail-user=navarrodelacruz@usf.edu

# Ensure output directory exists
mkdir -p $HOME/optimal_decision_tree/outputs

# Change to project directory
cd $HOME/optimal_decision_tree

# Debug working directory
echo "Current working directory: $(pwd)"
echo "Listing src directory:"
ls -l src/*.jl
echo "Listing test directory:"
ls -l test/*.jl
echo "Checking test.jl existence:"
ls -l test/test.jl

# Load Julia
export PATH=$HOME/julia-1.7.2/bin:$PATH

# Load OpenMPI 4.1.1
export PATH=$HOME/local/openmpi-4.1.1/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/openmpi-4.1.1/lib:$LD_LIBRARY_PATH

# Load CPLEX 20.1.0
export CPLEX_STUDIO_DIR201=$HOME/local/cplex201
export PATH=$CPLEX_STUDIO_DIR201/cplex/bin/x86-64_linux:$PATH
export LD_LIBRARY_PATH=$CPLEX_STUDIO_DIR201/cplex/lib/x86-64_linux:$LD_LIBRARY_PATH

# Debug environment
echo "Using Julia from: $(which julia)"
echo "Using mpiexec from: $(which mpiexec)"
echo "Using CPLEX from: $(which cplex)"

# Set dataset using SLURM array task ID
i=${SLURM_ARRAY_TASK_ID}
dataset="wall-following_${i}.wall-following"
echo "Running for dataset: $dataset"

# Set seed
seed="1"

# Define output file
output_file="$HOME/optimal_decision_tree/outputs/info-${dataset}-sd${seed}-2-CMS-${SLURM_NTASKS}.out"
echo "Running with seed: $seed - Output file: $output_file"

# Run the job with MPI
$HOME/local/openmpi-4.1.1/bin/mpiexec -n ${SLURM_NTASKS} $HOME/julia-1.7.2/bin/julia test/test.jl 2 CF+MILP+SG $seed par "$dataset" > "$output_file"

echo ">>> Job completed for dataset: $dataset with seed: $seed"