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
export nnUNet_preprocessed="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_results"

# dataset_name="Dataset100_Cityscapes"
# dataset_num=100

# dataset_name="Dataset101_Complete"
# dataset_num=101

# dataset_name="Dataset102_Vanilla"
# dataset_num=102

# dataset_name="Dataset103_UncDiff"
# dataset_num=103

# dataset_name="Dataset104_simclr"
# dataset_num=104

# dataset_name="Dataset105_fmUncondDiff" 
# dataset_num=105

dataset_name="Dataset106_fmSimclr"
dataset_num=106

imagesTs="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw/$dataset_name/imagesTs"
outputFolder="/home/woody/iwi5/iwi5192h/nnUNet/output_images/$dataset_name/imagesPr"
inputFoggyTs="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw/$dataset_name/imagesTs_foggy"
outputFoggyFolder="/home/woody/iwi5/iwi5192h/nnUNet/output_images/$dataset_name/imagesFoggyPr"
inputRainyTs="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw/$dataset_name/imagesTs_rainy"
outputRainyFolder="/home/woody/iwi5/iwi5192h/nnUNet/output_images/$dataset_name/imagesRainyPr"

srun nnUNetv2_predict -i $inputRainyTs -o $outputRainyFolder -d $dataset_num -c 2d -f 0