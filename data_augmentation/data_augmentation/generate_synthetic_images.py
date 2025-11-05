import random

#from cv2 import _OUTPUT_ARRAY_DEPTH_MASK_16U
import numpy as np
import cv2
from random import seed
from random import randint
import time
import os
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join as ospj
import yaml
import argparse

# Function to remap mask values to their class value
# input: mask with 0,255 values where 0 is background
# output: mask with 0,label values
def remapLabel(img, pos_label, neg_label):
    imgP = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if imgP[i, j] == 255:
                imgP[i, j] = pos_label
            elif imgP[i, j] != 0:
                imgP[i, j] = neg_label

    return img

# Apply random patch to the background image in random
# position and with random rotation
def randomPatchImg(fg_img_path, fg_mask_path, material, bg_img, bg_label, crop_area=None):
    fg_img = Image.open(fg_img_path)
    fg_mask = Image.open(fg_mask_path)

    # Crop the original image to have the background size
    if crop_area is not None:
        fg_img = fg_img.crop(crop_area)
        fg_mask = fg_mask.crop(crop_area)
    
    fg_img = fg_img.convert("RGBA")
    fg_mask = fg_mask.convert("L")
    bg_img = bg_img.convert("RGBA")
    
    ### ERODE THE MASK ###

    mask_array = np.array(fg_mask)

    # Define the kernel for erosion (2x2 square kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Perform erosion using OpenCV's erode function
    eroded_mask_array = cv2.erode(mask_array, kernel, iterations=2)

    # Convert the eroded mask back to a PIL Image
    fg_mask = Image.fromarray(eroded_mask_array)

    r, g, b, a = fg_img.split()
    new_a = fg_mask
    fg_img = Image.merge("RGBA", (r, g, b, new_a)) 
    assert fg_img.size[:2] == fg_mask.size[:2] and bg_img.size[:2] == bg_label.size[:2]
    x_res = fg_img.size[0] - 50
    y_res = fg_img.size[1] - 50
    x_offset = randint(50, x_res)
    y_offset = randint(50, y_res)
    angle_offset = randint(0, 360)

    ### RGB IMAGE ###

    fg_img = fg_img.crop(fg_img.getbbox())
    fg_img = fg_img.rotate(angle_offset, resample=Image.Resampling.BICUBIC, expand=True)

    # Get the dimensions of the foreground image after rotation
    fg_width, fg_height = fg_img.size

    # Calculate the center of the foreground image (taking rotation into account)
    center_x = x_offset - fg_width // 2
    center_y = y_offset - fg_height // 2
    
    #Paste the patch on background image
    Image.Image.paste(bg_img, fg_img, (center_x, center_y), fg_img)

    ### LABEL ###

    temp_mask = np.array(fg_mask)
    temp_mask[temp_mask>0] = material
    fg_mask = Image.fromarray(temp_mask).convert('L')
    fg_mask = fg_mask.crop(fg_mask.getbbox())
    fg_mask = fg_mask.rotate(angle_offset, resample=Image.Resampling.NEAREST, expand=True)
    
    # Paste the mask on 0 background
    Image.Image.paste(bg_label, fg_mask, (center_x, center_y), fg_img)

    return bg_img, bg_label

# Funtion to explore where the iage and mask path are
def find_rgb_folders(root_directory,image_folder_name,mask_folder_name):
    rgb_folders = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if image_folder_name in dirnames and mask_folder_name in dirnames:
            rgb_folders.append(os.path.join(dirpath))
    return rgb_folders

# Returns the full path of each image and its mask
def get_patch_full_path(root,image_folder_name,mask_folder_name):
    rgb = []
    labels = []
    
    directories = find_rgb_folders(root,image_folder_name,mask_folder_name)
    for dir in directories:
        rgb_path = os.path.join(dir, image_folder_name)
        mask_path = os.path.join(dir, mask_folder_name)
        for root, _, files in os.walk(rgb_path):  # Walk through directory and subdirectories
            for file in files:
                if file.lower().endswith('.png'):
                    rgb.append(os.path.join(rgb_path, file))  # Add full path to the list
        
        for root, _, files in os.walk(mask_path):  # Walk through directory and subdirectories
            for file in files:
                if file.lower().endswith('.png'):
                    labels.append(os.path.join(mask_path, file))  # Add full path to the list
    return rgb, labels

# Returns the full path of each image and its mask and material (label)
def get_path_lists(root,material,image_folder_name,mask_folder_name):
    data_list = [] 
    rgb, labels = get_patch_full_path(root,image_folder_name,mask_folder_name)
    for path_rgb, path_label in zip(rgb, labels):
        data_list.append([path_rgb, path_label,material]) 
    return data_list


def main():

    ### SETUP ###

    # Read the configuration file
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument("--config", type=str,
                        default='/home/toffanin/wastebot/segmentation_server/config/generate_synthetic_images_with_annotations_orbbec_config.yaml')
    args = parser.parse_args()
    yaml_config_file = args.config
    with open(yaml_config_file, "r") as file:
        data = yaml.safe_load(file) 

    input_arguments = data['input_arguments']
    class_distribution = data['class_distribution']
    data_size = {}
    data_size['training'] = input_arguments['size_training']
    data_size['validation'] = input_arguments['size_validation']
    #data_size = input_arguments['size']
    out_root  = input_arguments['out_dir']
    bg_path = input_arguments['background_dir']
    camera_idx = input_arguments['camera_idx']
    mean = input_arguments['mean_obj']
    std = input_arguments['std_obj']
    repetition = input_arguments['repetition']
    vis = input_arguments['visualize']
    image_folder_name = input_arguments['image_folder_name']
    mask_folder_name = input_arguments['mask_folder_name']
    val_percent = input_arguments['validation_percentage']

    classes = list(class_distribution.keys())

    # Read the probabilities of appearance for each class and
    # ensure they sum to 1
    probabilities = [class_distribution[cls]['prob'] for cls in classes]
    sum_probabilities = sum(probabilities)
    probabilities = [num / sum_probabilities for num in probabilities]


    # Seed random number generator
    rseed = int(time.time())
    np.random.seed(rseed)
    random.seed(rseed)
    print("RANDOM SEED: ", rseed)

    ### LOAD DATA ###

    # Load the [mask image material] list of paths and store the 
    # amount of images present for that class
    img_multi_lists = {}
    img_multi_lists['training'] = {}
    if val_percent > 0 :
            img_multi_lists['validation'] = {}
    for key in class_distribution:
        img_list = []
        for root in class_distribution[key]['root_data']:
            img_list.extend(get_path_lists(root,class_distribution[key]['label'],image_folder_name,mask_folder_name))
        random.shuffle(img_list)

        # Calculate split index
        split_index = int(len(img_list) * (1-val_percent))
        # Split the list
        train_list = img_list[:split_index]
        val_list = img_list[split_index:]
        img_multi_lists['training'][key] = {'img_list' : train_list, 'size' : len(train_list)}
        if val_percent > 0 :
            img_multi_lists['validation'][key] = {'img_list' : val_list, 'size' : len(val_list)}
    
    # Load background images and pre process them based on their indexes
    background_dict = {}
    for dirpath, _, filenames in os.walk(bg_path):
        for i,file in enumerate(filenames):
                image_path = os.path.join(dirpath, file)
                bg_image = Image.open(image_path)

                # Based on the camera image, decide how to 
                # pre process the background   
                if camera_idx == 0 :
                    area = (280, 58, 845, 623) #(280, 58, 845, 507)
                elif camera_idx == 1:
                    enhancer = ImageEnhance.Brightness(bg_image) 
                    factor = 2.0 
                    bg_image = enhancer.enhance(factor)
                    area = (61, 0, 1770, 1200)
                if camera_idx == 2 :
                    area = (0, 0, 1280, 720)
                bg_image = bg_image.crop(area)
                #bg_image = ImageOps.flip(bg_image)
                bg_image = np.array(bg_image)
                background_dict[i] = bg_image
    
    for set in img_multi_lists:
        
        # Create output folders
        out_root_set = os.path.join(out_root,set)
        out_rgb_root = os.path.join(out_root_set, "rgb")
        out_mask_root = os.path.join(out_root_set, "mask")
        list_folder = os.path.join(out_root, "lists")
        stats_path = os.path.join(out_root_set, "stats.yaml")

        os.makedirs(out_rgb_root, exist_ok=True)
        os.makedirs(out_mask_root, exist_ok=True)
        os.makedirs(list_folder, exist_ok=True)

        print("LOADED DATA: ")
        img_lists = {}
        
        img_lists = img_multi_lists[set]
        for key in img_lists:
            print(key, " : " ,img_lists[key]['size']," items")

        ### GENERATE SYNTHETIC IMAGES ###

        print("GENERATING IMAGES: ")
        data_list = []

        # Define statistics
        stats = {}
        for key in img_lists:
            stats["img_with_"+key] = 0
        stats["avg_obj_per_image"] = 0
        stats["total_images"] = 0
        stats["empty_images"] = 0

        # Loop for image generation
        for i in tqdm(range(data_size[set])):
            # Random muber of objects in the scene
            obj_count = max(0, int(np.random.normal(mean, std)))
            obj_list = []
            conveyor_belt = background_dict[random.randrange(0, len(background_dict))]

            if obj_count == 0:
                stats["empty_images"] += 1
            # Define two variables to store the already
            # chosen indexes in an image and the presence 
            # of a class in that image
            for key in img_lists:
                img_lists[key]['already_chosen_idx'] = []
                img_lists[key]['presence'] = 0

            for _ in range(obj_count):
                # Randomly choose a class based on the class probabilities
                chosen_class = random.choices(classes, probabilities)[0]
                while img_lists[chosen_class]['size'] == len(img_lists[chosen_class]['already_chosen_idx']):
                    chosen_class = random.choices(classes, probabilities)[0]
                random_int = random.randrange(0, img_lists[chosen_class]['size'])
                if not repetition:
                    while random_int in img_lists[chosen_class]['already_chosen_idx']:
                        random_int = random.randrange(0, img_lists[chosen_class]['size'])
                    img_lists[chosen_class]['already_chosen_idx'].append(random_int)
                obj_list.append(img_lists[chosen_class]['img_list'][random_int])
                img_lists[chosen_class]['presence'] = 1

            for key in img_lists:
                stats['img_with_'+key] += img_lists[key]['presence']
            stats["avg_obj_per_image"] += obj_count
            stats["total_images"] +=1

            indexes = np.random.RandomState(rseed).permutation(len(obj_list))


            conv_height = conveyor_belt.shape[0]
            conv_width = conveyor_belt.shape[1]
            alpha = np.ones((conv_height, conv_width, 1), np.uint8) + 255
            out_img = Image.fromarray(np.concatenate((conveyor_belt, alpha), axis=2))
            out_mask = Image.fromarray(np.zeros((conv_height, conv_width), np.uint8))

            for j in indexes:
                out_img, out_mask = randomPatchImg(obj_list[j][0], obj_list[j][1], obj_list[j][2], out_img, out_mask, crop_area = area)

            ### VISUALIZE THE IMAGE ####

            if vis:
                fig = plt.figure()
                fig.add_subplot(2, 1, 1)
                plt.imshow(out_img)
                fig.add_subplot(2, 1, 2)
                plt.imshow(out_mask)
                plt.show()

            out_img = out_img.convert("RGB")

            data_tuple = "{:06d}.png".format(i)

            ### STORE THE SYNTHETIC IMAGE ###

            out_img.save(ospj(out_rgb_root, data_tuple))
            out_mask.save(ospj(out_mask_root, data_tuple))
            data_list.append(data_tuple)

        ### STORE STATS ###

        stats["avg_obj_per_image"] = stats["avg_obj_per_image"] / data_size[set]
        print(stats)

        with open(stats_path, "w") as file:
            yaml.dump(stats, file, default_flow_style=False)

        # Write data_list to file
        # with open(ospj(list_folder, "data_list.txt"), "w") as f:
        #     f.writelines("%s\n" % item for item in data_list)

        # Split in train and val lists
        # train_list = data_list[:int(len(data_list)*0.8)]
        # val_list = data_list[int(len(data_list)*0.8):]

        # Write train_list to file
        with open(ospj(list_folder, set+".txt"), "w") as f:
            for item in data_list:
                f.writelines("%s\n" % item)

        # # Write val_list to file
        # with open(ospj(list_folder, "val.txt"), "w") as f:
        #     for item in val_list:
        #         f.writelines("%s\n" % item)

if __name__ == "__main__":
    main()