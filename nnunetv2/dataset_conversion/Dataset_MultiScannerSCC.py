import shutil
from batchgenerators.utilities.file_and_folder_operations import *

import sys
# sys.path.append("/home/hpc/iwi5/iwi5192h/nnUNet")
sys.path.append("/Users/poojashetty/Documents/MasterThesis/Code/nnUNet")

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
import numpy as np

import hydra
from omegaconf import DictConfig

from ms_scc.slide_container import SlideContainer
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import glob

np.random.seed(42)

def load_slides(cfg):
    train_files = []
    valid_files = []
    test_files = []

    datadir = os.path.join(cfg.location.msscc.data_dir,cfg.location.msscc.source)
    annotation_file = os.path.join(datadir, cfg.location.msscc.annotation_file)
    patch_size = cfg.patch_size
    label_dict = cfg.nclass.label_dict
    down_factor = cfg.down_factor
    csv_path = os.path.join(cfg.location.msscc.data_split_csv_path,cfg.data_split_csv)

    slides = pd.read_csv(csv_path, delimiter=";")
    for index, row in tqdm(slides.iterrows()):
        image_file = Path(glob.glob("{}/{}/{}_{}.{}".format(datadir,cfg.scanner.name,row["Slide"], cfg.scanner.name,cfg.scanner.data.file_extension), recursive=True)[0])
        # image_file = Path(glob.glob("{}/{}_{}.tif".format(datadir,row["Slide"],cfg.scanner.name), recursive=True)[0])
        if row["Dataset"] == "train" and row["Annotation"] == "no":
            train_files.append(SlideContainer(image_file, annotation_file, down_factor, patch_size, patch_size, label_dict = label_dict))
        elif row["Dataset"] == "val":
            valid_files.append(SlideContainer(image_file, annotation_file, down_factor, patch_size, patch_size, label_dict = label_dict))
        elif row["Dataset"] == "test":
            test_files.append(SlideContainer(image_file, annotation_file, down_factor, patch_size, patch_size, label_dict = label_dict, test_only=True))
        else:
            pass

    return train_files, valid_files, test_files

def load_single_slide(cfg,file_name):
    datadir = os.path.join(cfg.location.msscc.data_dir,cfg.location.msscc.source)
    annotation_file = os.path.join(datadir, cfg.location.msscc.annotation_file)
    patch_size = cfg.patch_size
    label_dict = cfg.nclass.label_dict
    down_factor = cfg.down_factor

    image_file = Path(glob.glob("{}/{}/{}_{}.{}".format(datadir,cfg.scanner.name,file_name, cfg.scanner.name,cfg.scanner.data.file_extension), recursive=True)[0])

    return SlideContainer(image_file, annotation_file, down_factor, patch_size, patch_size, label_dict = label_dict)

# this test_only is for all test images
def get_item(slide_container, test_only=False, excluded_val=0):
    xmin, ymin = slide_container.get_new_train_coordinates()
    if test_only:
        xmin = 0
        ymin = 0
    patch = slide_container.get_patch(xmin, ymin)
    y_patch = slide_container.get_y_patch(xmin, ymin, excluded_val)
    return (patch, y_patch)

@hydra.main(config_path="../../conf", config_name="config_ms_scc")
def main(cfg:DictConfig) -> None:
    # imagests = join(nnUNet_raw, cfg.dataset_name, 'imagesTs'+'_'+cfg.scanner.name)
    # labelsts = join(nnUNet_raw, cfg.dataset_name, 'labelsTs'+'_'+cfg.scanner.name)

    # maybe_mkdir_p(imagests)
    # maybe_mkdir_p(labelsts)

    # #generate images and labels for train and val
    # #sample images and then store it in nUnet format
    # train_files, valid_files, test_files = load_slides(cfg)
    # print("Number of train files: ", len(train_files))

    # for slide in test_files:
    #     # for i in range(cfg.patches_per_slide):
    #     x_patch, y_patch = get_item(slide,test_only=True,excluded_val=cfg.nclass.excluded)
    #     io.imsave(join(imagests, f"{slide.file.stem}_0000.png"), x_patch,check_contrast=False)
    #     io.imsave(join(labelsts, f"{slide.file.stem}.png"), y_patch,check_contrast=False)
    
    # if(not cfg.test_only):
        # imagestr = join(nnUNet_raw, cfg.dataset_name, 'imagesTr')
        # labelstr = join(nnUNet_raw, cfg.dataset_name, 'labelsTr')

        # maybe_mkdir_p(imagestr)
        # maybe_mkdir_p(labelstr)

        # for slide in train_files+valid_files:
        # for slide in train_files:
        #     for i in range(cfg.patches_per_slide):
        #         x_patch, y_patch = get_item(slide,excluded_val=cfg.nclass.excluded)
        #         io.imsave(join(imagestr, f"{slide.file.stem}_{i}_0000.png"), x_patch,check_contrast=False)
        #         io.imsave(join(labelstr, f"{slide.file.stem}_{i}.png"), y_patch,check_contrast=False)

        # num_train = len(train_files) * cfg.patches_per_slide + len(valid_files) * cfg.patches_per_slide
        # num_train = 1750

        # generate_dataset_json(join(nnUNet_raw, cfg.dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'tumour': 1, 'ignore': 2},
        #                     num_train, '.png', dataset_name=cfg.dataset_name)

        # generate_dataset_json(join(nnUNet_raw, cfg.dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'tissue': 1, 'tumour': 2, 'ignore': 3},
        #                     num_train, '.png', dataset_name=cfg.dataset_name)
        
        ##################################
        #for report
        # slide = load_single_slide(cfg,"scc_02") #cs2
        # slide = load_single_slide(cfg,"scc_06") #p1000
        # slide = load_single_slide(cfg,"scc_15") #p1000
        slide = load_single_slide(cfg,"scc_24") #p1000

        imagestr = "/Users/poojashetty/Documents/MasterThesis/Dataset/test_img_diff/msscc/samples/p1000_3"
        for i in range(cfg.patches_per_slide):
                x_patch, y_patch = get_item(slide,excluded_val=cfg.nclass.excluded)
                io.imsave(join(imagestr, f"{slide.file.stem}_{i}_0000.png"), x_patch,check_contrast=False)
                # io.imsave(join(labelstr, f"{slide.file.stem}_{i}.png"), y_patch,check_contrast=False)

if __name__ == "__main__":
    main()
