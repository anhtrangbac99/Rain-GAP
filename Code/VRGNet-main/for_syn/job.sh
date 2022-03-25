#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --job-name=gan_rain_01
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
source activate kietcdx
python train_syn_joint.py --data_path data/rain1400/training/rainy_image --gt_path data/rain1400/training/ground_truth --resume 933 --niter 2000
