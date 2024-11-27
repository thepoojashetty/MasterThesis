#!/bin/bash -l
#SBATCH --job-name=nnUnet_cityscapes
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH -o nnUNet_cityscapes_pre.out
#SBATCH -e nnUNet_cityscapes_pre.err
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00

module load python
conda activate nnunet

export nnUNet_raw="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_results"

# srun nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 102 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 103 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 104 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 105 --verify_dataset_integrity
srun nnUNetv2_plan_and_preprocess -d 106 --verify_dataset_integrity