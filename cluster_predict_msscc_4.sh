#!/bin/bash -l
#SBATCH --job-name=nnUNet
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH -o nnUNet_pred.out
#SBATCH -e nnUNet_pred.err
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=1:00:00

module load python
conda activate nnunet

# dataset_name="Dataset200_MultiscannerSCC"
# dataset_num=200
# dataset_name="Dataset201_Complete"
# dataset_num=201

# dataset_name="Dataset202_NearbyVanilla"
# dataset_num=202

# dataset_name="Dataset203_MPVanilla"
# dataset_num=203

# dataset_name="Dataset204_UncondDiffNearby"
# dataset_num=204

# dataset_name="Dataset205_UncondDiffMP"
# dataset_num=205

# dataset_name="Dataset206_SimclrNearby"
# dataset_num=206

# dataset_name="Dataset207_SimclrMP"
# dataset_num=207

# dataset_name="Dataset208_Simclr2Nearby"
# dataset_num=208

# dataset_name="Dataset209_Simclr2MP"
# dataset_num=209

# dataset_name="Dataset210_fmUncondDiffNearby"
# dataset_num=210

# dataset_name="Dataset211_fmUncondDiffMP"
# dataset_num=211

# dataset_name="Dataset212_fmSimclrNearby"
# dataset_num=212

# dataset_name="Dataset213_fmSimclrMP"
# dataset_num=213

# dataset_name="Dataset214_fmSimclr2Nearby"
# dataset_num=214

dataset_name="Dataset215_fmSimclr2MP"
dataset_num=215

export nnUNet_raw="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_results"
imagesTs_cs2="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw/$dataset_name/imagesTs_cs2"
outputFolder_cs2="/home/woody/iwi5/iwi5192h/nnUNet/output_images/$dataset_name/imagesPr_cs2"
imagesTs_p1000="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw/$dataset_name/imagesTs_p1000"
outputFolder_p1000="/home/woody/iwi5/iwi5192h/nnUNet/output_images/$dataset_name/imagesPr_p1000"

srun nnUNetv2_predict -i $imagesTs_cs2 -o $outputFolder_cs2 -d $dataset_num -c 2d -f 0