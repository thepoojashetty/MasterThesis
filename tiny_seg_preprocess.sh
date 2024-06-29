#!/bin/bash -l
#SBATCH --job-name=nnUNet
#SBATCH --gres=gpu:v100:2
#SBATCH --partition=v100
#SBATCH -o nnUNet.out
#SBATCH -e nnUNet.err
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00

module load python
conda activate nnunet

export nnUNet_raw="/home/hpc/iwi5/iwi5192h/MasterThesis_nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/hpc/iwi5/iwi5192h/MasterThesis_nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/hpc/iwi5/iwi5192h/MasterThesis_nnUNet/nnUNet_results"
srun nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity