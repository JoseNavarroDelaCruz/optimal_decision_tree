#! /bin/bash
#SBATCH --job-name=ODT_MultiData
#SBATCH --nodes=2               # Adjust nodes if needed
#SBATCH --ntasks-per-node=8      # Adjust number of tasks per node
#SBATCH --mem=50GB
#SBATCH --time=6:30:00           # Adjust time as needed
#SBATCH --exclusive
#SBATCH --output=info-%x.out
#SBATCH --error=info-%x.err

cd ${SLURM_SUBMIT_DIR}

# Load necessary modules or set paths
export PATH=$HOME/julia-1.7.2/bin:$HOME/openmpi-3.1.6/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openmpi-3.1.6/lib:$LD_LIBRARY_PATH
export CPLEX_STUDIO_DIR=$HOME/cplex
export PATH=$CPLEX_STUDIO_DIR/cplex/bin/x86-64_linux:$PATH
export LD_LIBRARY_PATH=$CPLEX_STUDIO_DIR/cplex/lib/x86-64_linux:$LD_LIBRARY_PATH

# Debugging paths
which julia
which mpiexec
which cplex

#Manually set the value of `i`
i="_140"  # Change this manually; set i="" for original datasets

#Define datasets, automatically handling `i`
datasets=("banknote${i}.banknote" "contraceptive${i}.contraceptive" "ozone-eight_${i}.ozone-eight" "pendigits${i}.pendigits" "spambase${i}.spambase" "thyroid-ann${i}.thyroid-ann" \
          "body${i}.body" "glass${i}.glass" "ozone-one${i}.ozone-one" "seeds${i}.seeds" "statlog-german${i}.statlog-german" "wall-following${i}.wall-following" \
          "concrete${i}.concrete" "page-block${i}.csv" "small_toy${i}.small_toy" "statlog-lansat${i}.statlog-lansat" \
           )


# Debugging: Print datasets
echo "Using i = '${i}'"
echo "Processing datasets: ${datasets[@]}"

# Set seed values
seeds=("1" "2" "3" "4" "5")

# Loop through all datasets and run the job
for dataset in "${datasets[@]}"; do
    dataset_name=$(basename "$dataset")  # Extract filename
    echo "Running dataset: $dataset_name"

    # Run with MPI
    mpiexec -n ${SLURM_NTASKS} julia test/test.jl 2 CF+MILP+SG ${seeds[0]} par "$dataset" > "info-${dataset_name}-sd${seeds[0]}-2-CMS-${SLURM_NTASKS}.out"

    echo ">>> Job completed for dataset: $dataset_name"
done
