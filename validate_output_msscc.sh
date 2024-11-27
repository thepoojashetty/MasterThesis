#!/bin/bash -l

module load python
conda activate nnunet

export nnUNet_raw="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_results"

# dataset_name="Dataset200_MultiscannerSCC"
# dataset_name="Dataset201_Complete"
# dataset_name="Dataset202_NearbyVanilla"
# dataset_name="Dataset203_MPVanilla"
# dataset_name="Dataset204_UncondDiffNearby"
# dataset_name="Dataset205_UncondDiffMP"
# dataset_name="Dataset206_SimclrNearby"
# dataset_name="Dataset207_SimclrMP"
# dataset_name="Dataset208_Simclr2Nearby"
# dataset_name="Dataset209_Simclr2MP"
# dataset_name="Dataset210_fmUncondDiffNearby"
# dataset_name="Dataset211_fmUncondDiffMP"
# dataset_name="Dataset212_fmSimclrNearby"
# dataset_name="Dataset213_fmSimclrMP"
# dataset_name="Dataset214_fmSimclr2Nearby"
dataset_name="Dataset215_fmSimclr2MP"

actual_seg_output_path_cs2="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw/$dataset_name/labelsTs_cs2"
predicted_seg_output_path_cs2="/home/woody/iwi5/iwi5192h/nnUNet/output_images/$dataset_name/imagesPr_cs2"
actual_seg_output_path_p1000="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw/$dataset_name/labelsTs_p1000"
predicted_seg_output_path_p1000="/home/woody/iwi5/iwi5192h/nnUNet/output_images/$dataset_name/imagesPr_p1000"

python validate_msscc.py actual_seg_output=$actual_seg_output_path_cs2 predicted_seg_output=$predicted_seg_output_path_cs2