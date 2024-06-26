#!/bin/bash -l
#SBATCH --job-name=nnUNet
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH -o nnUNet.out
#SBATCH -e nnUNet.err
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=2

module load python
conda activate nnunet

export nnUNet_raw="./nnUNet_raw"
srun python nnunetv2/dataset_conversion/Dataset_Cityscapes.py source="/home/hpc/iwi5/iwi5192h/Dataset/Cityscapes"