import shutil
from batchgenerators.utilities.file_and_folder_operations import *

# import sys
# sys.path.append("/Users/poojashetty/Documents/MasterThesis/Code/nnUNet")

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
import numpy as np

import hydra
from omegaconf import DictConfig

def process_seg_image(seg_image_path,cfg):
    seg = io.imread(seg_image_path)
    for id in cfg.void_classes:
        seg[seg == id] = 0
    for id in cfg.valid_classes:
        seg[seg == id] = 1
    return seg

def convert_data_to_nnUnet_format(image_source,label_source,image_dest,label_dest,split,city_list,suffix=".png",cfg=None,keep_prob=1):
        count = 0
        split_label_source= os.path.join(label_source,split)
        for root, dirs, files in os.walk(image_source):
            for file in files:
                if file.endswith(suffix):
                    city_name = root.split("/")[-1]
                    inp_seg_file_name = "_".join(file.split("_")[:3]) + "_gtFine_labelIds.png"
                    inp_seg_file_path = os.path.join(split_label_source,city_name,inp_seg_file_name)
                    if os.path.exists(inp_seg_file_path) and (city_name in cfg.val_cities or (np.random.random() <= keep_prob and city_name in city_list)):
                        count += 1
                        inp_image_file_path = os.path.join(root,file)
                        out_image_file_path = os.path.join(image_dest,"_".join(file.split("_")[:3])+"_0000.png")
                        out_seg_file_path = os.path.join(label_dest,"_".join(file.split("_")[:3])+".png")
                        seg = process_seg_image(inp_seg_file_path,cfg)
                        io.imsave(out_seg_file_path, seg, check_contrast=False)
                        shutil.copy(inp_image_file_path, out_image_file_path)
                    else:
                        continue
        return count

@hydra.main(config_path="../../conf", config_name="config_cityscapes")
def main(cfg:DictConfig) -> None:
    if cfg.dataset_type == "clear":
        convert_cityscapes_to_nnUnet_format(cfg)
    elif cfg.dataset_type == "foggy":
        convert_foggytestdata_to_nnUnet_format(cfg)
    elif cfg.dataset_type == "rainy":
        convert_rainytestdata_to_nnUnet_format(cfg)
    elif cfg.dataset_type == "complete":
        convert_completecityscapes_to_nnUnet_format(cfg)

def convert_completecityscapes_to_nnUnet_format(cfg:DictConfig):
    imagests = join(nnUNet_raw, cfg.dataset_name, 'imagesTs')
    labelsts = join(nnUNet_raw, cfg.dataset_name, 'labelsTs')

    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelsts)

    source = cfg.location.cityscapes.source
    
    test_source = join(source, cfg.val_rel_path)
    label_source = join(source, cfg.label_rel_path) #without split
    
    #clear test data
    convert_data_to_nnUnet_format(test_source,label_source,imagests,labelsts,"val",cfg.test_cities,".png",cfg)

    #i will copy the rest
    #foggy test data
    # convert_foggytestdata_to_nnUnet_format(cfg)
    # #rainy test data
    # convert_rainytestdata_to_nnUnet_format(cfg)

    imagestr = join(nnUNet_raw, cfg.dataset_name, 'imagesTr')
    labelstr = join(nnUNet_raw, cfg.dataset_name, 'labelsTr')

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    #clear train data
    train_source = join(source, cfg.train_rel_path)
    # returns a list of tuple of train image paths and corr label paths
    num_train_clear = convert_data_to_nnUnet_format(train_source,label_source,imagestr,labelstr,"train",cfg.anno_cities,".png",cfg,keep_prob=cfg.train_split_ratio)

    #foggy train data
    # foggy_source = join(source, cfg.train_foggy_rel_path)
    num_train_foggy = convert_foggytraindata_to_nnUnet_format(train_source,label_source,imagestr,labelstr,"train",cfg.unanno_cities,".png",cfg)

    #rainy train data
    rainy_source = join(source, cfg.train_rainy_rel_path)
    num_train_rainy = convert_rainytraindata_to_nnUnet_format(rainy_source,label_source,imagestr,labelstr,"train",cfg.unanno_cities,num_train_foggy,".png",cfg)

    num_train = num_train_clear + num_train_foggy + num_train_rainy

    generate_dataset_json(join(nnUNet_raw, cfg.dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'car': 1},
                        num_train, '.png', dataset_name=cfg.dataset_name)
    
def convert_rainytraindata_to_nnUnet_format(train_source,label_source,image_dest,label_dest,split,city_list,size,suffix=".png",cfg=None):
    split_label_source= os.path.join(label_source,split)
    sample_list = []
    i = 0
    for root, dirs, files in os.walk(train_source):
        for file in files:
            if file.endswith(suffix):
                city_name = root.split("/")[-1]
                if city_name in city_list:
                    inp_seg_file_name = "_".join(file.split("_")[:3]) + "_gtFine_labelIds.png"
                    inp_seg_file_path = os.path.join(split_label_source,city_name,inp_seg_file_name)
                    out_seg_file_path = os.path.join(label_dest,"_".join(file.split("_")[:3])+f"_{i}.png")
                    seg = process_seg_image(inp_seg_file_path,cfg)
                    # io.imsave(out_seg_file_path, seg, check_contrast=False)

                    inp_image_file_path = os.path.join(root,file)
                    out_image_file_path = os.path.join(image_dest,"_".join(file.split("_")[:3])+f"_{i}_0000.png")
                    # shutil.copy(inp_image_file_path, out_image_file_path)
                    sample_list.append((out_seg_file_path, seg,inp_image_file_path,out_image_file_path))
                    i += 1

    # Randomly sample from the list
    sampled_indices = np.random.choice(len(sample_list), size, replace=False)
    for idx in sampled_indices:
        out_seg_file_path, seg, inp_image_file_path, out_image_file_path = sample_list[idx]
        io.imsave(out_seg_file_path, seg, check_contrast=False)
        shutil.copy(inp_image_file_path, out_image_file_path)

    return size
    
def convert_foggytraindata_to_nnUnet_format(train_source,label_source,image_dest,label_dest,split,city_list,suffix=".png",cfg=None):
    foggy_file_endings = ["_leftImg8bit_foggy_beta_0.01.png","_leftImg8bit_foggy_beta_0.02.png","_leftImg8bit_foggy_beta_0.005.png"]
    split_label_source= os.path.join(label_source,split)
    count = 0

    source = cfg.location.cityscapes.source

    for root, dirs, files in os.walk(train_source):
        for file in files:
            if file.endswith(suffix):
                city_name = root.split("/")[-1]
                if city_name in city_list:
                    count += 1
                    inp_seg_file_name = "_".join(file.split("_")[:3]) + "_gtFine_labelIds.png"
                    inp_seg_file_path = os.path.join(split_label_source,city_name,inp_seg_file_name)
                    out_seg_file_path = os.path.join(label_dest,"_".join(file.split("_")[:3])+".png")
                    seg = process_seg_image(inp_seg_file_path,cfg)
                    io.imsave(out_seg_file_path, seg, check_contrast=False)

                    ending = np.random.choice(foggy_file_endings)
                    foggy_file_name = "_".join(file.split("_")[:3]) + ending
                    inp_image_file_path = os.path.join(source,cfg.train_foggy_rel_path,city_name,foggy_file_name)
                    out_image_file_path = os.path.join(image_dest,"_".join(file.split("_")[:3])+"_0000.png")
                    shutil.copy(inp_image_file_path, out_image_file_path)
    return count

def convert_cityscapes_to_nnUnet_format(cfg:DictConfig):
    imagests = join(nnUNet_raw, cfg.dataset_name, 'imagesTs')
    labelsts = join(nnUNet_raw, cfg.dataset_name, 'labelsTs')

    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelsts)

    source = cfg.location.cityscapes.source

    test_source = join(source, cfg.val_rel_path)
    label_source = join(source, cfg.label_rel_path) #without split
    # test_source = join(cfg.source, cfg.test_rel_path) # we are not using test set as they are not labeled
    #test images-- we will consider val as test
    convert_data_to_nnUnet_format(test_source,label_source,imagests,labelsts,"val",cfg.test_cities,".png",cfg)
    
    if not cfg.test_only:
        train_source = join(source, cfg.train_rel_path)

        imagestr = join(nnUNet_raw, cfg.dataset_name, 'imagesTr')
        labelstr = join(nnUNet_raw, cfg.dataset_name, 'labelsTr')

        maybe_mkdir_p(imagestr)
        maybe_mkdir_p(labelstr)
        # returns a list of tuple of train image paths and corr label paths
        num_train = convert_data_to_nnUnet_format(train_source,label_source,imagestr,labelstr,"train",cfg.anno_cities,".png",cfg,keep_prob=cfg.train_split_ratio)
    
        generate_dataset_json(join(nnUNet_raw, cfg.dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'car': 1},
                            num_train, '.png', dataset_name=cfg.dataset_name)

def convert_foggytestdata_to_nnUnet_format(cfg:DictConfig=None):
    foggy_file_endings = ["_leftImg8bit_foggy_beta_0.01.png","_leftImg8bit_foggy_beta_0.02.png","_leftImg8bit_foggy_beta_0.005.png"]

    imagests_foggy = join(nnUNet_raw, cfg.dataset_name, 'imagesTs_foggy')
    labelsts_foggy = join(nnUNet_raw, cfg.dataset_name, 'labelsTs_foggy')

    maybe_mkdir_p(imagests_foggy)
    maybe_mkdir_p(labelsts_foggy)

    source = cfg.location.cityscapes.source
    test_source = join(source, cfg.val_rel_path)
    split_label_source = join(source, cfg.label_rel_path,"val")
    suffix = ".png"

    for root, dirs, files in os.walk(test_source):
        for file in files:
            if file.endswith(suffix):
                city_name = root.split("/")[-1]
                inp_seg_file_name = "_".join(file.split("_")[:3]) + "_gtFine_labelIds.png"
                inp_seg_file_path = os.path.join(split_label_source,city_name,inp_seg_file_name)
                out_seg_file_path = os.path.join(labelsts_foggy,"_".join(file.split("_")[:3])+".png")
                seg = process_seg_image(inp_seg_file_path,cfg)
                io.imsave(out_seg_file_path, seg, check_contrast=False)

                ending = np.random.choice(foggy_file_endings)
                foggy_file_name = "_".join(file.split("_")[:3]) + ending
                inp_image_file_path = os.path.join(source,cfg.val_foggy_rel_path,city_name,foggy_file_name)
                out_image_file_path = os.path.join(imagests_foggy,"_".join(file.split("_")[:3])+"_0000.png")
                shutil.copy(inp_image_file_path, out_image_file_path)

def convert_rainytestdata_to_nnUnet_format(cfg:DictConfig=None):
    imagests_rainy = join(nnUNet_raw, cfg.dataset_name, 'imagesTs_rainy')
    labelsts_rainy = join(nnUNet_raw, cfg.dataset_name, 'labelsTs_rainy')

    maybe_mkdir_p(imagests_rainy)
    maybe_mkdir_p(labelsts_rainy)

    source = cfg.location.cityscapes.source
    test_source = join(source, cfg.val_rainy_rel_path)
    split_label_source = join(source, cfg.label_rel_path,"val")
    suffix = ".png"

    i = 0
    for root, dirs, files in os.walk(test_source):
        for file in files:
            if file.endswith(suffix):
                city_name = root.split("/")[-1]
                inp_seg_file_name = "_".join(file.split("_")[:3]) + "_gtFine_labelIds.png"
                inp_seg_file_path = os.path.join(split_label_source,city_name,inp_seg_file_name)
                out_seg_file_path = os.path.join(labelsts_rainy,"_".join(file.split("_")[:3])+f"_{i}.png")
                seg = process_seg_image(inp_seg_file_path,cfg)
                io.imsave(out_seg_file_path, seg, check_contrast=False)

                inp_image_file_path = os.path.join(root,file)
                out_image_file_path = os.path.join(imagests_rainy,"_".join(file.split("_")[:3])+f"_{i}_0000.png")
                shutil.copy(inp_image_file_path, out_image_file_path)
                i += 1
    
if __name__ == "__main__":
    main()
