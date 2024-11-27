#!/bin/bash -l
#SBATCH --job-name=nnUNet
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH -o nnUNet.out
#SBATCH -e nnUNet.err
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=1:00:00

module load python
conda activate nnunet

export nnUNet_raw="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_results"

# srun nnUNetv2_plan_and_preprocess -d 200 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 201 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 202 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 203 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 204 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 205 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 206 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 207 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 208 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 209 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 210 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 211 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 212 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 213 --verify_dataset_integrity
# srun nnUNetv2_plan_and_preprocess -d 214 --verify_dataset_integrity
srun nnUNetv2_plan_and_preprocess -d 215 --verify_dataset_integrity
