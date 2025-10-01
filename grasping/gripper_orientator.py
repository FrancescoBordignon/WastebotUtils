import cv2
import numpy as np
import math
from shapely.geometry import box
from shapely.affinity import rotate


class GripperOrientator:
    """
    Computes the optimal orientation of a robotic gripper relative to an object mask,
    aiming to avoid obstacles detected in a scene.
    It can be used also to determine the 2D grasping point given the mask of the wanted object using
    erode_undtil_one_pixel

    Attributes:
        focal_length_x = focal length of the camera in pixels (optional: necesssary only if depth is used)
        focal§_length_y = focal length of the camera in pixels (optional: necesssary only if depth is used)
        edge_length (float): Fixed gripper length in the same measurement unit of the depth map ( if depth is not available this has fixed dimension in pixels)
        edge_height (float): Fixed gripper height in the same measurement unit of the depth map ( if depth is not available this has fixed dimension in pixels)
        object_importance(float): the importance of the cossision with the object that needs to be gripped.(default 0 means it is important as any other object,
        -1 is the minimum importancew( not important at all) while the maximum value can be any)
    """

    def __init__(self, 
                 edge_length, edge_height,
                 focal_length_x =None , focal_length_y =None,
                 object_importance = 0):
        
        self.focal_length_x = focal_length_x
        self.focal_length_y = focal_length_y

        self.edge_length_x = edge_length
        self.edge_height_y = edge_height

        self.object_importance = object_importance

    def find_orientation(self, mask, point2d, panoptic_map, depth_point=None,
                         angle_rad_inerval=math.pi / 18, initial_guess_importance = 1):
        """
        Finds the optimal gripper orientation around a given grasping point.

        Args:
            mask (np.ndarray): Binary mask (1-channel) of the target object.
            point2d (tuple): Pixel coordinates (x, y) of the grasping point.
            panoptic_map (list[dict]): List of instance dicts, each with
                'segmentation' as a boolean mask (e.g., SAM output).
            depth_point (float, optional): Depth at grasp point (needed if aperture scaling factors are used).
            angle_rad_inerval (float): Angle step (in radians) for searching orientations.
            initial_guess_importance(float): this number identifies how important ti sthe initial guessed orientation (found with PCA) 
            0 (or lower) = max importance 
            1 = same importance as other guesses, 
            >1 less important than other guesses

        Returns:
            tuple:
                - (best_vector, opposite_vector) as unit vectors in camera plane.
                - best_bounding_box (np.ndarray): Pixel coordinates of gripper edges.
        """
        

        # Combine all panoptic instance masks into one obstacle mask
        mask_obstacles = np.zeros(mask.shape, dtype=bool)
        for instance in panoptic_map:
            mask_obstacles |= instance['segmentation']

        
        
        # PCA to find the main object axis
        contours, _ = cv2.findContours((mask.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        data_pts = contour.reshape(-1, 2).astype(np.float32)
        _, eigenvectors = cv2.PCACompute(data_pts, mean=np.array([]))
        direction = eigenvectors[0]
        init_angle_rad = math.atan2(direction[1], direction[0])

        # Initial bounding box based on PCA angle
        # Create the bounding box of the gripper
        edges = self.find_gripper_edges(point2d, depth_point, init_angle_rad)   
        mask_gripper = np.zeros(mask_obstacles.shape, dtype=np.uint8)
        cv2.fillPoly(mask_gripper, [edges], color=1)
        best_score = initial_guess_importance*self.compute_score(mask_gripper, mask, mask_obstacles)
        best_angle_rad = init_angle_rad
        best_bounding_box = edges

        # Search for orientation with minimal obstacle overlap
        for angle_offset in np.arange(-math.pi / 2, math.pi / 2, angle_rad_inerval):
            angle_rad = init_angle_rad + angle_offset
            edges = self.find_gripper_edges(point2d, depth_point, angle_rad)
            mask_gripper = np.zeros(mask_obstacles.shape, dtype=np.uint8)
            cv2.fillPoly(mask_gripper, [edges], color=1)
            score = self.compute_score(mask_gripper, mask, mask_obstacles)
            if score < best_score:
                best_score = score
                best_angle_rad = angle_rad
                best_bounding_box = edges

        vec_camera = np.array([np.cos(best_angle_rad), np.sin(best_angle_rad), 0])

        return (vec_camera, -vec_camera), best_bounding_box

    def compute_score(self, mask_gripper, mask, mask_obstacles):
        """
        Computes score = (area of gripper mask overlapping obstacles) + object_importance * area of gripper mask overlapping gripping object
        Lower is better.
        """
        bool_gripper_mask = mask_gripper > 0
        intersection_all = bool_gripper_mask[mask_obstacles]
        intersection_object = bool_gripper_mask[mask]
        return np.sum(intersection_all) + (max(self.object_importance, -1) * np.sum(intersection_object))


    def find_gripper_edges(self, point_camera_frame, depth_point, angle_rad):
        """
        Generates the rotated bounding box for the gripper at a given angle.

        Args:
            point_camera_frame (tuple): Center point (x, y) in pixels.
            depth_point (float): Depth at center point (if scaling factors are used).
            angle_rad (float): Rotation angle in radians.

        Returns:
            np.ndarray: Coordinates of rotated gripper polygon vertices.
        """
        x_center, y_center = point_camera_frame

        # Determine gripper dimensions
        if depth_point:
            edge_length = (self.edge_length_x / depth_point) * self.focal_length_x
            edge_height = (self.edge_height_y / depth_point) * self.focal_length_y
        else:
            edge_length = self.edge_length_x
            edge_height = self.edge_height_y

        # Create and rotate rectangle
        bbox = box(x_center - edge_length / 2, y_center - edge_height / 2,
                   x_center + edge_length / 2, y_center + edge_height / 2)
        rotated_bbox = rotate(bbox, angle_rad, origin=(x_center, y_center), use_radians=True)
        return np.array(rotated_bbox.exterior.coords, dtype=np.int32)

    def erode_until_one_pixel(self, mask, erosion_kernel_size = 3):
        """
        Generates a grasping point eroding the mask until one pixel remains
        
        Args:
            mask (bool): boolean mask of the wanted object
            erosion_kernel_size (int): kernel size for erosion

        Returns:
            x,y (int): tuple containing the coordinates of the centered grasping point
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
        border_thickness = 10
        if mask.dtype != np.uint8:
            mask = (mask.astype(np.uint8)) * 255

        # padding necessary to avoid finding a point on the edge
        current = np.pad(mask.copy(), pad_width=border_thickness, mode='constant', constant_values=0)
        
        while np.count_nonzero(current) > 0:
            prev = current.copy()
            current = cv2.erode(current, kernel)
        coords = np.argwhere(prev > 0)
        return (coords[0][1] - border_thickness, coords[0][0] - border_thickness)


def store_image_with_orientation(original_image, orientation, filename):
    """
    Saves an image with a drawn orientation vector.
    """
    image = original_image.copy()
    norm = np.linalg.norm(orientation)
    if norm == 0:
        return
    orientation = orientation / norm
    center = (image.shape[1] // 2, image.shape[0] // 2)
    end_point = (int(center[0] + orientation[0] * 50),
                 int(center[1] + orientation[1] * 50))
    cv2.arrowedLine(image, center, end_point, (255, 0, 0), 2)
    cv2.imwrite(filename, image)
    print(f"Image saved with orientation at {filename}")


def store_image_with_point(original_image, point, filename):
    """
    Saves an image with a marked point.
    """
    image = original_image.copy()
    cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
    cv2.imwrite(filename, image)
    print(f"Image saved with point at {filename}")


def store_image_with_gripper_bbox(original_image, gripper_bbox, filename):
    """
    Saves an image with the gripper bounding box drawn.
    """
    image = original_image.copy()
    cv2.polylines(image, [gripper_bbox.astype(np.int32)], isClosed=True,
                  color=(0, 255, 0), thickness=2)
    cv2.imwrite(filename, image)
    print(f"Image saved with gripper bounding box at {filename}")


# Usage example
import random

def main():
    # set bounding box dimension
    edge_length = 0.2 #meters
    edge_height = 0.1 #meters

    # set focal length 
    focal_length_x = 500 #pixels
    focal_length_y = 500 #pixels

    # decide object importance
    object_importance = 1 # min = -1 max can be any default is 0 (means every obj has the same importance)
    
    # create the panoptic map
    number_of_masks = 5
    image_size = 512
    min_mask_size = 50
    max_mask_size = 200
    panoptic_map = []
    visualization_panoptic_map = np.zeros((image_size, image_size), dtype=bool)
    for i in range(number_of_masks):
        mask = np.zeros((image_size, image_size), dtype=bool)
        side = random.randint(min_mask_size, max_mask_size)
        
        # pick top-left corner (ensuring square fits in image)
        x = random.randint(0, image_size - side)
        y = random.randint(0, image_size - side)

        if i == 0:
            # get the mask of the wanted object
            mask_wanted_obj = mask

        # fill the square
        mask[y:y+side, x:x+side] = True
        visualization_panoptic_map |= mask
        panoptic_map.append({'segmentation' : mask})

    visualization_panoptic_map = (visualization_panoptic_map.astype(np.uint8)) * 255
    # # Show using OpenCV
    # cv2.imshow("Random Panoptic Map", visualization_panoptic_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # decide the angle rad interval to compute the orientation
    angle_rad_inerval=math.pi / 16

    # create a Gripper Orientator object
    orientator = GripperOrientator(focal_length_x, focal_length_y, edge_length, edge_height, object_importance)

    # get the 2D gripping point
    point2D = orientator.erode_until_one_pixel(mask_wanted_obj,erosion_kernel_size = 3)

    # find orientation
    orientations, best_bounding_box = orientator.find_orientation(mask_wanted_obj, point2D, panoptic_map, depth_point=0.4, angle_rad_inerval = angle_rad_inerval, initial_guess_importance = 1)
    # show results
    visualization_panoptic_map = cv2.merge([visualization_panoptic_map, visualization_panoptic_map, visualization_panoptic_map])
    visualization_panoptic_map[mask_wanted_obj] = [155,155,0]
    store_image_with_orientation(visualization_panoptic_map, orientations[0], "output/image_with_orientation.png")
    store_image_with_gripper_bbox(visualization_panoptic_map, best_bounding_box, "output/image_with_bbox.png")
    store_image_with_point(visualization_panoptic_map, point2D, "output/image_with_grasping_point.png")

    # 
    #  _____ _           _                 _ _    __       _ _        _ 
    # |_   _| |__   __ _| |_   ___    __ _| | |  / _| ___ | | | _____| |
    #   | | | '_ \ / _` | __| / __|  / _` | | | | |_ / _ \| | |/ / __| |
    #   | | | | | | (_| | |_  \__ \ | (_| | | | |  _| (_) | |   <\__ \_|
    #   |_| |_| |_|\__,_|\__| |___/  \__,_|_|_| |_|  \___/|_|_|\_\___(_)

    return 0

if __name__ == "__main__":
    main()
