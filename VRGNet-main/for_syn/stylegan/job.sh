#!/bin/bash
#SBATCH --partition=SCSEGPU_MSAI
#SBATCH --qos=q_msai24
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --job-name=style
#SBATCH --output=output/train/output_%x_%j.out
#SBATCH --error=error/train/error_%x_%j.err

module load anaconda
source activate kietcdx
python train.py --config configs/sample.yaml --generator_file checkpoints/rain/models/GAN_GEN_4_4.pth --gen_shadow_file checkpoints/rain/models/GAN_GEN_SHADOW_4_4.pth --discriminator_file checkpoints/rain/models/GAN_DIS_4_4.pth --gen_optim_file checkpoints/rain/models/GAN_GEN_OPTIM_4_4.pth --dis_optim_file checkpoints/rain/models/GAN_DIS_OPTIM_4_4.pth

