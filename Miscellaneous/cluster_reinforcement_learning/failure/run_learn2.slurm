#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J qlearn # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 10080 # Runtime in minutes 
#SBATCH -p general # Partition to submit to 
#SBATCH --mem=10000 # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o qlearn_%A.out # Standard out goes to this file 
#SBATCH -e qlearn_%A.err # Standard err goes to this filehostname 

source new-modules.sh
module load python/3.6.3-fasrc01
conda create -n ody --clone="$PYTHON_HOME"
source activate ody
conda install matplotlib 
conda install tqdm

python learn2.py

source deactivate ody
