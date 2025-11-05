# Usage example
import random
import time
import math
import os
import numpy as np
from grasp_planner.grasp_planner import GraspPlanner, store_image_with_point, store_image_with_gripper_bbox

# Support visualization function
def colorize_mask(mask):
    """
    Converts a labeled mask to a BGR color image for visualization.
    Value 0 is mapped to black. Other values get random distinct colors.
    
    Parameters:
        mask (np.ndarray): 2D array of integer class labels
    
    Returns:
        color_mask (np.ndarray): 3D BGR image for visualization
    """
    unique_values = np.unique(mask)
    
    # Create a color map (label -> color)
    colormap = {}
    for val in unique_values:
        if val == 0:
            colormap[val] = (0, 0, 0)  # Black for background
        else:
            # Random color (in BGR for OpenCV)
            colormap[val] = tuple(np.random.randint(0, 256, size=3).tolist())

    # Create a blank color image
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Apply color mapping
    for val, color in colormap.items():
        color_mask[mask == val] = color

    return color_mask

def main():

    ### PARAMETERS SETTING ###

    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # set bounding box dimension
    gripper_width_x = 0.3 #meters (optionally set it to pixels if depth of point is set to None)
    gripper_height_y = 0.15 #meters (optionally set it to pixels if depth of point is set to None)

    # set focal length 
    f_x = 500 #pixels
    f_y = 500 #pixels

    # decide object importance
    desired_object_importance = 1.0 # min = 0 max 1 (default is o.5 which means obstacle avoidance and object gripping have the same importance)

    # assign a random depth to the point or give None if you want fixed size gripper dimensions
    depth_point = 0.4
    
    # decide 2d point chosing criterion 
    criterion = "bigger_area" # other options are: # min_depth or max_depth ( they require a depth map )

    # Depth map of the image 
    depth = None

    # Desired class to grasp ( if None all the grasping points found are returned)
    desired_object_class = 2

    # decide the angle rad interval to compute the orientation
    angle_rad_interval=math.pi / 18

    # decide how to evaluate the score of an orientation
    scoring_type= "rotated_boxes" #"three_parts_mask"  #"contact_points+three_parts"

    # How amany oriented masks do you mant at most
    max_grasping_masks = 3

    # Percentage of the mask belonging to gripper fingers used in scoring_type= "three_parts_mask"
    gripper_fingers_percentage = 0.15

    ### EXAMPLE INPUT CREATION ###

    # create the panoptic map
    class_labels = [2,5,6]
    number_of_masks = 15
    image_size = 1500
    min_mask_size = 50
    max_mask_size = 200

    SEED = random.randint(2,150)#13 #33
    print("random seed: ",SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for i in range(number_of_masks):
        mask_class = class_labels[random.randint(0, len(class_labels)-1)]
        side = random.randint(min_mask_size, max_mask_size)
        
        # pick top-left corner (ensuring square fits in image)
        x = random.randint(0, image_size - side)
        y = random.randint(0, image_size - side)

        # fill the square
        mask[y:y+side, x:x+side] = mask_class

    visualization_mask = colorize_mask(mask)

    # create a GraspPlanner object
    grasper = GraspPlanner()

    start = time.time()
    # get the 2D grasping points
    points = grasper.find_grasping_points(mask, desired_class = desired_object_class, criterion = criterion, depth = depth)
    points = grasper.find_grasping_points(mask, desired_class = desired_object_class, criterion = criterion, depth = depth)
    if len(points) > 0:
        point2D = points[0]
        print("time to find all grasping points: ", time.time() - start)
        # find orientation
        if len(points) > 0:
            start = time.time()
            grasping_bboxes = grasper.find_point_orientation(mask = mask, point2d = point2D,
                                    gripper_width_x = gripper_width_x, gripper_height_y = gripper_height_y, 
                                    scoring_type=scoring_type, max_grasping_masks=max_grasping_masks,
                                    desired_object_importance = desired_object_importance,
                                    PCA_importance = 1.0, depth_point=depth_point, 
                                    f_x = f_x, f_y = f_y,
                                    angle_rad_interval=angle_rad_interval, 
                                    gripper_fingers_percentage=gripper_fingers_percentage,
                                    min_object_size = 200)
            print("time to find the orientation: ", time.time() - start)

            # show results
            if grasping_bboxes:
                store_image_with_point(visualization_mask, point2D, os.path.join(output_folder,"image_with_grasping_point.png"))
                for index, bbox in enumerate(grasping_bboxes):
                    rotated_gripper_bbox = grasper.rotate_mask(bbox[2], bbox[1], (int(point2D[0]), int(point2D[1])))
                    visualization_mask = store_image_with_gripper_bbox(visualization_mask, rotated_gripper_bbox, os.path.join(output_folder,"image_with_bbox.png"), color = (100+50*index,255-40*index,0))
    #                                                                                                                                                                                    ___________
    #                                                                                                                                                                              /|   /          /              
    #                                                                                                                                         ___ .--.                           /  |  |          |
    #                                                                                                      /|                                /   V __ \                        /    |  |          |      ________
    #  _____ _           _                 _ _    __       _ _        _                                   / |                                |/| /\\ \|                    _ /_____ |  \___________\    |    _  |
    # |_   _| |__   __ _| |_   ___    __ _| | |  / _| ___ | | | _____| |              ____               /  |\                                 \_\ \\                       \______\|________||_________/   |_| | 
    #   | | | '_ \ / _` | __| / __|  / _` | | | | |_ / _ \| | |/ / __| |           .'      \            /   | \                                ____| \_______                \                                 /
    #   | | | | | | (_| | |_  \__ \ | (_| | | | |  _| (_) | |   <\__ \_|          /      /\/          _/____|__\__                         ___/               \_______        `.    0   0   0   0   0   0     |
    #   |_| |_| |_|\__,_|\__| |___/  \__,_|_|_| |_|  \___/|_|_|\_\___(_) _ _ ____(      '._______ _ _ \_________.'_ _ __ ____________ __ _/                           '.._ _ __ `. __________________________/
    return 0

if __name__ == "__main__":
    main()
