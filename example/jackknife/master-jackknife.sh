#!/bin/sh

#SBATCH --no-requeue
#SBATCH --partition=spica

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --job-name=alcyone
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt

export PYTHONPATH="$PYTHONPATH:$HOME/opt/repositories/git-reps/alcyone.glucksfall"
export PYTHONPATH="$PYTHONPATH:$HOME/opt/repositories/git-reps/pleione.glucksfall"

python3 -m alcyone.jackknife --soft kasim \
--model ../model.kappa --data ../cold_stress_* --final 90 --steps 10 --syntax=3 \
--error CHISQ \
--seeds 0 1 2 \
--python /usr/bin/python3 --slurm spica
