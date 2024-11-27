#!/bin/bash -l
#SBATCH --job-name=nnUNet
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH -o nnUNet.out
#SBATCH -e nnUNet.err
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00

module load python
conda activate nnunet

export nnUNet_raw="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw"
#200
# srun python nnunetv2/dataset_conversion/Dataset_MultiScannerSCC.py location=cluster nclass=two_with_ignore
# srun python nnunetv2/dataset_conversion/Dataset_MultiScannerSCC.py location=cluster test_only=True scanner=cs2 nclass=two_with_ignore

#201: for complete training, 20% p1000, rest 80% cs2, we will copy rest of data from 200
# srun python nnunetv2/dataset_conversion/Dataset_MultiScannerSCC.py location=cluster scanner=cs2 nclass=two_with_ignore dataset_name=Dataset201_Complete

#202: vanilla training nearby, dataset_name=Dataset202_NearbyVanilla

#203: vanilla training mp, dataset_name=Dataset203_MPVanilla

#204: uncond diff nearby, dataset_name=Dataset204_UncondDiffNearby

#205: uncond diff mp, dataset_name=Dataset205_UncondDiffMP

#206: simclr nearby, dataset_name=Dataset206_SimclrNearby

#207: simclr mp, dataset_name=Dataset207_SimclrMP