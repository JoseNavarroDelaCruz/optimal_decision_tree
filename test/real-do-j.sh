#! /bin/bash -l
#SBATCH --job-name=ODT_MultiData
#SBATCH --nodes=2               # Adjust nodes if needed
#SBATCH --ntasks-per-node=128      # Adjust number of tasks per node
#SBATCH --time=24:00:00           # Adjust time as needed
#SBATCH --exclusive
#SBATCH --output=info-%x-%a.out
#SBATCH --error=info-%x-%a.err
#SBATCH --array=1-2           # Adjust the range as needed
#SBATCH --partition=amd_2021
#SBATCH --mail-type=ALL
#SBATCH --mail-user=navarrodelacruz@usf.edu

# Initialize the module system (if not automatically done)
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh

# Load the MPI module (this will set the correct environment for OpenMPI)
module load mpi/openmpi/3.1.6

# Set the environment for Julia and CPLEX
export PATH=$HOME/julia-1.7.2/bin:$PATH
export PATH=$HOME/cplex/cplex/bin/x86-64_linux:$PATH
export LD_LIBRARY_PATH=$HOME/cplex/cplex/lib/x86-64_linux:$LD_LIBRARY_PATH

# Change to the directory from which the job was submitted
cd ${SLURM_SUBMIT_DIR}

# Debugging paths
which julia
which mpiexec
#which cplex

# Set the value of `i` using the SLURM array task ID
i=${SLURM_ARRAY_TASK_ID}

# Define the dataset for the current iteration
dataset="wall-following_${i}.wall-following"

# Debugging: Print the current dataset
echo "Running for dataset: $dataset"

# Set seed value (you can also loop through multiple seeds if needed)
seed="1"

# Define the output file name using the dataset and seed
output_file="info-${dataset}-sd${seed}-2-CMS-${SLURM_NTASKS}.out"
echo "Running with seed: $seed - Output file: $output_file"

# Run the job with MPI
mpiexec -n ${SLURM_NTASKS} julia test.jl 2 CF+MILP+SG $seed par "$dataset" > "$output_file"

echo ">>> Job completed for dataset: $dataset with seed: $seed"
