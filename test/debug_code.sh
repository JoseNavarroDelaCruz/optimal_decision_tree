#! /bin/bash
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=20
#SBATCH -t 0-8:30
#SBATCH --array=0,1
#SBATCH --output=info-%x-%a-2-CF+MILP+SG.out

# Change the following directory for your computer
cd "/c/Users/navarrodelacruz/OneDrive - University of South Florida/USF/PROJECTS/optimal_decision_tree" || { echo "Directory not found!"; exit 1; }

# seeds number for multi-run
seeds=("1" "2" "3" "4" "5")
# 0-1 # large datasets
#datasets=("wall-following")
datasets=("pendigits")

echo " "
echo " "
echo "Starting job for dataset: ${datasets[${SLURM_ARRAY_TASK_ID}]}"
echo "***************"


#mpiexec -n ${SLURM_NTASKS} julia test/test.jl 2 CF+MILP+SG ${seeds[0]} par ${datasets[${SLURM_ARRAY_TASK_ID}]} > ${datasets[${SLURM_ARRAY_TASK_ID}]}-sd${seeds[0]}-2-CMS-${SLURM_NTASKS}.out
julia test/test.jl 2 CF+MILP+SG ${seeds[0]} par ${datasets[0]} | tee ${datasets[0]}-sd${seeds[0]}-2-CMS-${SLURM_NTASKS}.out

:<<EOF
    # input argument of julia
    # args[1] maximum depth of the tree
    # args[2] lower bound method: base-dim, base-glb, CF+MILP+SG
    # args[3] random seed value
    # args[5] dataset name
EOF