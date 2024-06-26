import shutil
from batchgenerators.utilities.file_and_folder_operations import *

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

def convert_data_to_nnUnet_format(image_source,label_source,image_dest,label_dest,split,suffix=".png",cfg=None,keep_prob=1):
        count = 0
        split_label_source= os.path.join(label_source,split)
        for root, dirs, files in os.walk(image_source):
            for file in files:
                if file.endswith(suffix):
                    city_name = root.split("/")[-1]
                    inp_seg_file_name = "_".join(file.split("_")[:3]) + "_gtFine_labelIds.png"
                    inp_seg_file_path = os.path.join(split_label_source,city_name,inp_seg_file_name)
                    if os.path.exists(inp_seg_file_path) and np.random.random() <= keep_prob:
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


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg:DictConfig) -> None:
    imagestr = join(nnUNet_raw, cfg.dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, cfg.dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, cfg.dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, cfg.dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(cfg.source, cfg.train_rel_path)
    val_source = join(cfg.source, cfg.val_rel_path)
    test_source = join(cfg.source, cfg.test_rel_path)
    label_source = join(cfg.source, cfg.label_rel_path) #without split

    # returns a list of tuple of train image paths and corr label paths
    num_train = convert_data_to_nnUnet_format(train_source,label_source,imagestr,labelstr,"train",".png",cfg,keep_prob=cfg.train_split_ratio)
    num_val = convert_data_to_nnUnet_format(val_source,label_source,imagestr,labelstr,"val",".png",cfg)
    num_train += num_val
    num_test = convert_data_to_nnUnet_format(test_source,label_source,imagests,labelsts,"test",".png",cfg)

    generate_dataset_json(join(nnUNet_raw, cfg.dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'car': 1},
                          num_train, '.png', dataset_name=cfg.dataset_name)

if __name__ == "__main__":
    main()
