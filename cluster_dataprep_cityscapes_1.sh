#!/bin/bash -l
#SBATCH --job-name=nnUNet_cityscapes
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH -o nnUNet_cityscapes.out
#SBATCH -e nnUNet_cityscapes.err
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00

module load python
conda activate nnunet

export nnUNet_raw="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw"

# srun python nnunetv2/dataset_conversion/Dataset_Cityscapes.py location=cluster
# srun python nnunetv2/dataset_conversion/Dataset_Cityscapes.py location=cluster dataset_type=foggy
# srun python nnunetv2/dataset_conversion/Dataset_Cityscapes.py location=cluster dataset_type=rainy

#with vanilla settings: Diff_SSL_Augmented_Mean_Cityscapes_vanilla
# srun python nnunetv2/dataset_conversion/Dataset_Cityscapes.py location=cluster dataset_name=Dataset102_Vanilla
# srun python nnunetv2/dataset_conversion/Dataset_Cityscapes.py location=cluster dataset_type=foggy dataset_name=Dataset102_Vanilla
# srun python nnunetv2/dataset_conversion/Dataset_Cityscapes.py location=cluster dataset_type=rainy dataset_name=Dataset102_Vanilla

#with complete data
# srun python nnunetv2/dataset_conversion/Dataset_Cityscapes.py location=cluster dataset_name=Dataset101_Complete dataset_type=complete

#with uncond diff : Diff_Uncdiff_Augmented_Mean_Cityscapes_finetune
#we will be manually copying foggy and rainy data from Dataset100 and then we add the extra generated data
# srun python nnunetv2/dataset_conversion/Dataset_Cityscapes.py location=cluster dataset_name=Dataset103_UncDiff

#with simclr : SimCLR_cityscapes_finetune
#we will be manually copying foggy and rainy data from Dataset100 and then we add the extra generated data
srun python nnunetv2/dataset_conversion/Dataset_Cityscapes.py location=cluster dataset_name=Dataset104_simclr