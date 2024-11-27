#!/bin/bash -l
#SBATCH --job-name=nnUNet_cityscapes_106
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH -o nnUNet_cityscapes_106.out
#SBATCH -e nnUNet_cityscapes_106.err
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00

module load python
conda activate nnunet

export nnUNet_raw="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_results"

# srun nnUNetv2_train 100 2d 0
# srun nnUNetv2_train 102 2d 0
# srun nnUNetv2_train 101 2d 0
# srun nnUNetv2_train 103 2d 0
# srun nnUNetv2_train 104 2d 0
# srun nnUNetv2_train 105 2d 0
srun nnUNetv2_train 106 2d 0