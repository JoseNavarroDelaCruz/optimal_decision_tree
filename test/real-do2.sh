#! /bin/bash
#SBATCH --job-name=ODT_MultiData
#SBATCH --nodes=1               # Adjust nodes if needed
#SBATCH --ntasks-per-node=128      # Adjust number of tasks per node
#SBATCH --time=24:00:00           # Adjust time as needed
#SBATCH --exclusive
#SBATCH --array=1-2           # Adjust the range as needed
#SBATCH --output=/home/n/navarrodelacruz/optimal_decision_tree/outputs/info-%x-%a.out
#SBATCH --error=/home/n/navarrodelacruz/optimal_decision_tree/outputs/info-%x-%a.err
#SBATCH --partition=amd_2021
#SBATCH --mail-type=ALL
#SBATCH --mail-user=navarrodelacruz@usf.edu


# === Ensure output directory exists
mkdir -p $HOME/optimal_decision_tree/outputs


cd ${SLURM_SUBMIT_DIR}

# Load necessary modules or set paths
# Julia
export PATH=$HOME/julia-1.7.2/bin:$PATH

# OpenMPI 4.1.1
export PATH=$HOME/local/openmpi-4.1.1/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/openmpi-4.1.1/lib:$LD_LIBRARY_PATH

# CPLEX 20.1.0
export CPLEX_STUDIO_DIR201=$HOME/local/cplex201
export PATH=$CPLEX_STUDIO_DIR201/cplex/bin/x86-64_linux:$PATH
export LD_LIBRARY_PATH=$CPLEX_STUDIO_DIR201/cplex/lib/x86-64_linux:$LD_LIBRARY_PATH

# === DEBUG ENVIRONMENT ===
echo "Using Julia from: $(which julia)"
echo "Using mpiexec from: $(which mpiexec)"
echo "Using CPLEX from: $(which cplex)"

# Set the value of `i` using the SLURM array task ID
i=${SLURM_ARRAY_TASK_ID}
dataset="wall-following_${i}.wall-following" # Define the dataset for the current iteration

# Debugging: Print the current dataset
echo "Running for dataset: $dataset"

# Set seed value (you can also loop through multiple seeds if needed)
seed="1"

# Define the output file name using the dataset and seed
output_file="info-${dataset}-sd${seed}-2-CMS-${SLURM_NTASKS}.out"
echo "Running with seed: $seed - Output file: $output_file"

# Run the job with MPI
#mpiexec -n ${SLURM_NTASKS} julia test/test.jl 2 CF+MILP+SG $seed par "$dataset" > "$output_file"
$HOME/local/openmpi-4.1.1/bin/mpiexec -n ${SLURM_NTASKS} $HOME/julia-1.7.2/bin/julia test.jl 2 CF+MILP+SG $seed par "$dataset"

echo ">>> Job completed for dataset: $dataset with seed: $seed"
