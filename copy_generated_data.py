import os
import numpy as np
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

nnUNet_raw="/home/woody/iwi5/iwi5192h/nnUNet/nnUNet_raw"
source = "/home/woody/iwi5/iwi5192h/Datasets/syn_data"

#cityscapes: actual train-1019- we ll take two times this
#total images in train with complete training : 4328
#so count = 4328-1622= 2706
count =  2706 #train_ratio=1, num_train= 1622+2706=4328
num_train = 4328

# dataset_name = "Dataset102_Vanilla" # ratio =1
# source_name = "Predict_Diff_SSL_Augmented_Mean_Cityscapes_vanilla"

# dataset_name = "Dataset103_UncDiff"
# source_name = "Predict_Diff_Uncdiff_Augmented_Mean_Cityscapes_finetune"

# dataset_name = "Dataset104_simclr" 
# source_name = "Predict_SimCLR_cityscapes_sameaug_finetune"

# dataset_name = "Dataset105_fmUncondDiff" 
# source_name = "Predict_fm_Diff_Uncdiff_Augmented_Mean_Cityscapes_finetune"

dataset_name = "Dataset106_fmSimclr"
source_name = "Predict_fm_SimCLR_cityscapes_sameaug_finetune"

##################################################
#msscc
# count = 1200
# num_train = 1750

# dataset_name = "Dataset202_NearbyVanilla"
# source_name = "Predict_Diff_Nearby_Mean_MSSCC_vanilla"

# dataset_name = "Dataset203_MPVanilla"
# source_name = "Predict_Diff_MP_Linear_MSSCC_vanilla"

# dataset_name = "Dataset204_UncondDiffNearby"
# source_name = "Predict_Diff_Uncdiff_Nearby_Mean_MSSCC_finetune"

# dataset_name = "Dataset205_UncondDiffMP"
# source_name = "Predict_Diff_Uncdiff_MP_Linear_MSSCC_finetune"

# dataset_name = "Dataset206_SimclrNearby"
# source_name = "Predict_SimCLR_MSSCC_Nearby_finetune"

# dataset_name = "Dataset207_SimclrMP"
# source_name = "Predict_SimCLR_MSSCC_MP_linear_finetune"

# dataset_name = "Dataset208_Simclr2Nearby"
# source_name = "Predict_SimCLR2_MSSCC_Nearby_finetune"

# dataset_name = "Dataset209_Simclr2MP"
# source_name = "Predict_SimCLR2_MSSCC_MP_linear_finetune"

# dataset_name = "Dataset210_fmUncondDiffNearby"
# source_name = "Predict_fm_Diff_Uncdiff_Nearby_Mean_MSSCC_finetune"

# dataset_name = "Dataset211_fmUncondDiffMP"
# source_name = "Predict_fm_Diff_Uncdiff_MP_Linear_MSSCC_finetune"

# dataset_name = "Dataset212_fmSimclrNearby"
# source_name = "Predict_fm_SimCLR_MSSCC_Nearby_finetune"

# dataset_name = "Dataset213_fmSimclrMP"
# source_name = "Predict_fm_SimCLR_MSSCC_MP_linear_finetune"

# dataset_name = "Dataset214_fmSimclr2Nearby"
# source_name = "Predict_fm_SimCLR2_MSSCC_Nearby_finetune"

# dataset_name = "Dataset215_fmSimclr2MP"
# source_name = "Predict_fm_SimCLR2_MSSCC_MP_linear_finetune"

imagestr = os.path.join(nnUNet_raw, dataset_name, 'imagesTr')
labelstr = os.path.join(nnUNet_raw, dataset_name, 'labelsTr')

source_imgs = os.path.join(source, source_name, 'imgs')
source_segs = os.path.join(source, source_name, 'segs')

total=3000
choices = np.random.choice(total,count,replace=False)
for ind in choices:
    num_str = str(ind).zfill(5)
    inp_img_path = os.path.join(source_imgs, f"img_{num_str}_0000.png")
    inp_seg_path = os.path.join(source_segs, f"img_{num_str}.png")
    out_img_path = os.path.join(imagestr, f"img_{num_str}_0000.png")
    out_seg_path = os.path.join(labelstr, f"img_{num_str}.png")
    shutil.copy(inp_img_path, out_img_path)
    shutil.copy(inp_seg_path, out_seg_path)

generate_dataset_json(os.path.join(nnUNet_raw,dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'car': 1},
                            num_train, '.png', dataset_name=dataset_name)
# generate_dataset_json(os.path.join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'tumour': 1, 'ignore': 2},
#                             num_train, '.png', dataset_name=dataset_name)
