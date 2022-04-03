#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --job-name=gan_rain_01
#SBATCH --output=output/test_interpolation/output_%x_%j.out
#SBATCH --error=error/test_interpolation/error_test_%x_%j.err

module load anaconda
source activate kietcdx
python test_interpolation.py --netED syn1400models/ED_state_272.pt
