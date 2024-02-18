#!/bin/bash
# Brandon Michaud
#
# disc_dual_a100_students
#SBATCH --partition=disc_dual_a100_students
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16
#SBATCH --mem=1G
#SBATCH --output=outputs/hw1_%j_stdout.txt
#SBATCH --error=outputs/hw1_%j_stderr.txt
#SBATCH --time=00:05:00
#SBATCH --job-name=hw1
#SBATCH --mail-user=brandondmichaud@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/cs5043-hw1

. /home/fagg/tf_setup.sh
conda activate tf

python hw1_base_skel.py --project 'hw1' --predict_dim 1 --rotation 10 --Ntraining 18 --lrate 0.0001 --activation_hidden 'elu' --activation_out 'linear' --hidden 100 10 --epochs 100 -vv
