import hydra
from omegaconf import DictConfig
from validate_utils import load_images_from_folder, dice_coefficient
import numpy as np

@hydra.main(config_path="conf", config_name="config_cityscapes")
def main(cfg: DictConfig):
    with open("dice_scores.txt", "w",encoding='utf-8') as f:
        actual_images = load_images_from_folder(cfg.actual_seg_output)
        predicted_images = load_images_from_folder(cfg.predicted_seg_output)
        
        dice_scores = {}
        for filename, actual_img in actual_images.items():
            predicted_img = predicted_images.get(filename)
            if predicted_img is not None:
                dice = dice_coefficient(actual_img, predicted_img, cfg.nclass.classes_to_include)
                dice_scores[filename] = dice
            else:
                print(f'Prediction for {filename} not found.')
        
        for filename, dice in dice_scores.items():
            print(f'{filename}: Dice Coefficient = {dice:.4f}')
            #store in file
            f.write(f'{filename}: Dice Coefficient = {dice:.4f}\n')
        
        mean_dice = np.mean(list(dice_scores.values()))
        print(f'Mean Dice Coefficient: {mean_dice:.4f}')
        f.write("*****************************************")
        f.write(f'Mean Dice Coefficient: {mean_dice:.4f}\n')

if __name__ == "__main__":
    main()
