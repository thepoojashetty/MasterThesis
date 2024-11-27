#!/bin/bash -l
#SBATCH --job-name=nnUNet_msscc_215
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH -o nnUNet_msscc_215.out
#SBATCH -e nnUNet_msscc_215.err
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00

module load python
conda activate nnunet

export nnUNet_raw="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_results"

# srun nnUNetv2_train 200 2d 0
# srun nnUNetv2_train 201 2d 0
# srun nnUNetv2_train 202 2d 0
# srun nnUNetv2_train 203 2d 0
# srun nnUNetv2_train 204 2d 0
# srun nnUNetv2_train 205 2d 0
# srun nnUNetv2_train 206 2d 0
# srun nnUNetv2_train 207 2d 0
# srun nnUNetv2_train 208 2d 0
# srun nnUNetv2_train 209 2d 0
# srun nnUNetv2_train 210 2d 0
# srun nnUNetv2_train 211 2d 0
# srun nnUNetv2_train 212 2d 0
# srun nnUNetv2_train 213 2d 0
# srun nnUNetv2_train 214 2d 0
srun nnUNetv2_train 215 2d 0