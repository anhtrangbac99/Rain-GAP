#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=q_ug24
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --job-name=style
#SBATCH --output=output/test/output_%x_%j.out
#SBATCH --error=error/test/error_%x_%j.err

module load anaconda
source activate kietcdx
python generate_samples.py --config configs/sample.yaml --generator_file checkpoints/rain/models/GAN_GEN_SHADOW_4_4.pth --output_dir image_test/

