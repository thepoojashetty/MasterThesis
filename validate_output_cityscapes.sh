#!/bin/bash -l

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

dataset_name="Dataset106_fmSimclr"

actual_seg_output_path_clear="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw/$dataset_name/labelsTs"
predicted_seg_output_path_clear="/home/woody/iwi5/iwi5192h/nnUNet/output_images/$dataset_name/imagesPr"
actual_seg_output_path_foggy="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw/$dataset_name/labelsTs_foggy"
predicted_foggy_seg_output_path="/home/woody/iwi5/iwi5192h/nnUNet/output_images/$dataset_name/imagesFoggyPr"
actual_seg_output_path_rainy="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw/$dataset_name/labelsTs_rainy"
predicted_rainy_seg_output_path="/home/woody/iwi5/iwi5192h/nnUNet/output_images/$dataset_name/imagesRainyPr"

python validate_cityscapes.py actual_seg_output=$actual_seg_output_path_clear predicted_seg_output=$predicted_seg_output_path_clear