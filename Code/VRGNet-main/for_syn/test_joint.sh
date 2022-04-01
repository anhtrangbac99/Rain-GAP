#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --job-name=gan_rain_01
#SBATCH --output=output/test_joint/output_%x_%j.out
#SBATCH --error=error/test_joint/error_test_%x_%j.err

module load anaconda
source activate kietcdx
python test_syn_joint.py --netDerain syn1400models/DerainNet_state_500.pt --data_path ./data/rain1400/testing/rainy_image
