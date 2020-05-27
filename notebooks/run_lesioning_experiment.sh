#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4      # cores requested
#SBATCH --partition=gpu
#SBATCH --job-name lesion
#SBATCH --mem=32G  # memory 
#SBATCH --output logfiles/sbatch-logfile-%j.txt  # send stdout to outfile
#SBATCH --time=04:00:00

python3 lesion_experiment.py $1 $2 $3 $4