#!/bin/bash
#
#SBATCH --job-name=pybn_run
#SBATCH --output=res_pybn_run.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#

# Initialize virtual environment.
cd /home/mendez/PyBN
source pybn_env/bin/activate
python pybn_exec.py --base_values=2,3,4 \
                    --num_nodes='50-60,5'\
                    --var=5