#! /bin/bash

# Define directories
HOME_REPO="$HOME/optimal_decision_tree"  # Your repository in /home
WORK_DIR="$WORK_BGFS/optimal_decision_tree"
LOG_DIR="$HOME/optimal_decision_tree/logs"  # Directory in /home to store logs

# Create the job directory in /work_bgfs
mkdir -p $WORK_DIR || { echo "ERROR: Failed to create $WORK_DIR"; exit 1; }
mkdir -p $LOG_DIR || { echo "ERROR: Failed to create $LOG_DIR"; exit 1; }

# Copy necessary files from home to work directory
if [ -d "$WORK_DIR" ]; then
    echo "Warning: $WORK_DIR already exists. Overwriting contents."
fi
cp -r "$HOME_REPO" "$WORK_DIR/.."

# Change to work directory
cd "$WORK_DIR" || { echo "ERROR: Failed to cd into $WORK_DIR"; exit 1; }


module load NiaEnv/2019b
#module load gcc/9.2.0 openmpi/4.0.3
module load intel/2019u4 intelmpi/2019u4
#module load ddt # load this module to prevent error of julia module loading error
module load julia/1.7.2
module load mycplex/20.1.0


cat <<EOT > data_augment.sh
#!/bin/bash
#SBATCH --job-name=data_augment
#SBATCH --output=$LOG_DIR/data_augment_%j.out
#SBATCH --error=$LOG_DIR/data_augment_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=6
#SBATCH --mem=64G
#SBATCH --array=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=navarrodelacruz@usf.edu
#SBATCH --partition=chbme_2018
#SBATCH --qos=preempt


# seeds number for multi-run
seeds=("1" "2" "3" "4" "5")

# 0-1 # large datasets
datasets=("seeds")

echo " "
echo " "
echo "Starting job for dataset: ${datasets[${SLURM_ARRAY_TASK_ID}]}"
echo "***************"

mpiexec -n ${SLURM_NTASKS} julia test/test.jl 2 CF+MILP+SG ${seeds[0]} par ${datasets[${SLURM_ARRAY_TASK_ID}]} > ${datasets[${SLURM_ARRAY_TASK_ID}]}-sd${seeds[0]}-2-CMS-${SLURM_NTASKS}.out
EOT

# Make the script executable
chmod +x data_augment.sh || { echo "ERROR: Failed to make data_augment.sh executable"; exit 1; }

# Submit the job
sbatch data_augment.sh || { echo "ERROR: Failed to submit job"; exit 1; }


julia test/test.jl 2 CF+MILP+SG ${seeds[0]} par ${datasets[0]} | tee ${datasets[0]}-sd${seeds[0]}-2-CMS-${SLURM_NTASKS}.out

sbatch data_augment.sh || { echo "ERROR: Failed to submit job"; exit 1; }