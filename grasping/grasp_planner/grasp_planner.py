# Import required libraries
import cv2
import numpy as np
import math
from shapely.geometry import box
from shapely.affinity import rotate
from scipy.ndimage import label

class GraspPlanner:
    """
    Class to compute optimal 2D grasping points and orientations for a robotic gripper.

    Main features:
    - Detect grasping points using erosion-based centroids.
    - Score and rank points using multiple criteria.
    - Determine best gripper orientation to minimize collision.

    Attributes:
        grasping_criterion: List of supported criteria for scoring grasp points:
            - "bigger_area": prefer larger object areas
            - "depth_max": prefer deeper points
            - "depth_min": prefer shallower points
    """
    def __init__(self):
        self.grasping_criterion = ["bigger_area", "depth_max", "depth_min"]

    def find_grasping_points(self, mask, desired_class=None, criterion="bigger_area", depth=None):
        """
        Find all potential grasping points in the mask for a desired class.

        Args:
            mask (np.ndarray): segmentation mask, where each class is an integer value
            desired_class (int or None): target class to grasp; if None, all classes
            criterion (str): scoring method
            depth (np.ndarray or None): depth map if using depth-based scoring

        Returns:
            np.ndarray: Nx4 array with grasping points [x, y, class, score]
        """
        # Sanity checks
        if criterion not in self.grasping_criterion:
            raise ValueError(f"{criterion} is an invalid criterion. Choose among: {self.grasping_criterion}")
        if mask is None:
            raise ValueError("Mask is None")
        if len(np.unique(mask)) < 2:
            return []

        # Convert boolean mask to uint8
        if mask.dtype == bool:
            mask = mask.astype(np.uint8)

        # Store grasping points [x, y, class, score]
        grasping_points = []

        # Kernel size for erosion-based point detection
        erosion_kernel_size = max(3, int(max(mask.shape) / 300))

        # Class selection: either a specific class or all unique non-zero classes
        class_labels = [desired_class] if desired_class is not None else np.unique(mask)

        # Process each class individually
        for class_label in class_labels:
            if class_label != 0:  # Skip background
                # Separate spatially disconnected instances
                labeled_mask = self.__separate_instances(mask == class_label)
                instance_labels = np.unique(labeled_mask)

                for instance_label in instance_labels:
                    if instance_label != 0:
                        instance_mask = labeled_mask == instance_label
                        gp_x, gp_y = self.__erode_until_one_pixel(instance_mask, erosion_kernel_size)

                        # Scoring the grasp point using the chosen criterion
                        if criterion == "bigger_area":
                            score = (math.atan(np.count_nonzero(instance_mask)) + math.pi / 2) / math.pi
                        elif criterion == "depth_max":
                            if depth is not None:
                                score = (math.atan(depth[gp_y, gp_x]) + math.pi / 2) / math.pi
                            else:
                                raise ValueError("depth_max criterion selected but depth not provided")
                        elif criterion == "depth_min":
                            if depth is not None:
                                if depth[gp_y, gp_x] != 0:
                                    score = (math.atan(1 / depth[gp_y, gp_x]) + math.pi / 2) / math.pi
                                else:
                                    score = 0.5
                            else:
                                raise ValueError("depth_min criterion selected but depth not provided")

                        # Store result
                        grasping_points.append([gp_x, gp_y, class_label, score])

        # Sort grasping points by score (descending)
        grasping_points = np.array(grasping_points)
        if len(grasping_points) == 0:
            return grasping_points
        grasping_points = grasping_points[grasping_points[:, 3].argsort()[::-1]]

        return grasping_points

    def find_point_orientation(self, mask, point2d, gripper_width_x, gripper_height_y,
                               depth_point=None, f_x=None, f_y=None, angle_rad_inerval=math.pi / 18,
                               initial_guess_importance=1.0, desired_object_importance=1.0):
        """
        Determine the best gripper orientation for a given grasping point.

        Args:
            mask (np.ndarray): segmentation mask
            point2d (tuple): (x, y, ...) coordinates of the grasping point
            gripper_width_x, gripper_height_y (float): physical or pixel gripper size
            depth_point (float): depth at the grasping point
            f_x, f_y (int): focal lengths for scaling if depth is used
            angle_rad_inerval (float): angular step for orientation search
            initial_guess_importance (float): weight for PCA initial guess
            desired_object_importance (float): weight for penalizing overlap with desired object

        Returns:
            tuple: ((best_vector, opposite_vector), bounding_box)
        """
        # Sanity checks
        if gripper_width_x <= 0 or gripper_height_y <= 0:
            raise ValueError("Gripper size must be > 0")
        if depth_point is not None:
            if f_x is None or f_y is None:
                raise ValueError("Depth given without focal lengths")
            if depth_point <= 0:
                raise ValueError("Depth must be > 0")
            if f_x <= 0 or f_y <= 0:
                raise ValueError("Focal lengths must be > 0")

        # Extract object class mask for the given point
        mask_object = mask == mask[int(point2d[1]), int(point2d[0])]

        # PCA to get the main orientation of the object
        contours, _ = cv2.findContours((mask_object.astype(np.uint8)) * 255,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        data_pts = contour.reshape(-1, 2).astype(np.float32)
        _, eigenvectors = cv2.PCACompute(data_pts, mean=np.array([]))
        direction = eigenvectors[0]
        init_angle_rad = math.atan2(direction[1], direction[0])

        # Resize image to speed up computation
        original_mask_shape = mask.shape[:2]
        scale = 256 / min(original_mask_shape)
        new_w = int(round(original_mask_shape[0] * scale))
        new_h = int(round(original_mask_shape[1] * scale))

        # Resize masks and convert to boolean
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_object = cv2.resize(mask_object.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)

        # Compute gripper mask using initial orientation
        edges = self.__find_gripper_edges(point2d[:2], gripper_width_x, gripper_height_y,
                                          depth_point, init_angle_rad, f_x, f_y)
        mask_gripper = np.zeros(original_mask_shape, dtype=np.uint8)
        cv2.fillPoly(mask_gripper, [edges], color=1)
        mask_gripper = cv2.resize(mask_gripper, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Compute initial score
        best_score = initial_guess_importance * self.__compute_orientation_score(
            mask_gripper, mask_object, mask, desired_object_importance)
        best_angle_rad = init_angle_rad
        best_bounding_box = edges

        # Test other angles
        for angle_offset in np.arange(-math.pi / 2,  math.pi / 2 + angle_rad_inerval, angle_rad_inerval):
            angle_rad = init_angle_rad + angle_offset
            edges = self.__find_gripper_edges(point2d[:2], gripper_width_x, gripper_height_y,
                                              depth_point, angle_rad, f_x, f_y)
            mask_gripper = np.zeros(original_mask_shape, dtype=np.uint8)
            cv2.fillPoly(mask_gripper, [edges], color=1)
            mask_gripper = cv2.resize(mask_gripper, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            score = self.__compute_orientation_score(mask_gripper, mask_object, mask, desired_object_importance)

            if score < best_score:
                best_score = score
                best_angle_rad = angle_rad
                best_bounding_box = edges

        vec_camera = np.array([np.cos(best_angle_rad), np.sin(best_angle_rad), 0])
        return (vec_camera, -vec_camera), best_bounding_box

    def __compute_orientation_score(self, mask_gripper, mask_object, mask_obstacles, desired_object_importance):
        """
        Score = overlap with all obstacles + overlap with desired object * importance
        Lower is better.
        """
        desired_object_importance = desired_object_importance - 1
        bool_gripper_mask = mask_gripper > 0
        intersection_all = bool_gripper_mask[mask_obstacles]
        intersection_object = bool_gripper_mask[mask_object]
        return np.sum(intersection_all) + (max(desired_object_importance, -1) * np.sum(intersection_object))

    def __find_gripper_edges(self, point_camera_frame, gripper_width_x, gripper_height_y,
                              depth_point, angle_rad, f_x, f_y):
        """
        Returns 4-corner polygon of gripper rotated at angle_rad around the point.
        """
        x_center, y_center = point_camera_frame

        if depth_point is not None:
            edge_length = (gripper_width_x / depth_point) * f_x
            edge_height = (gripper_height_y / depth_point) * f_y
        else:
            edge_length = gripper_width_x
            edge_height = gripper_height_y

        bbox = box(x_center - edge_length / 2, y_center - edge_height / 2,
                   x_center + edge_length / 2, y_center + edge_height / 2)
        rotated_bbox = rotate(bbox, angle_rad, origin=(x_center, y_center), use_radians=True)
        edges = np.array(rotated_bbox.exterior.coords, dtype=np.int32)[:4]
        return np.array(edges, dtype=np.int32).reshape((-1, 1, 2))

    def __erode_until_one_pixel(self, mask, erosion_kernel_size=3):
        """
        Iteratively erodes the mask until only one point remains, used as grasp center.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
        border_thickness = 10
        if mask.dtype != np.uint8:
            mask = (mask.astype(np.uint8)) * 255
        current = np.pad(mask.copy(), pad_width=border_thickness, mode='constant', constant_values=0)
        while np.count_nonzero(current) > 0:
            prev = current.copy()
            current = cv2.erode(current, kernel)
        coords = np.argwhere(prev > 0)
        return (coords[0][1] - border_thickness, coords[0][0] - border_thickness)

    def __separate_instances(self, boolean_mask, connectivity=1):
        """
        Separates spatially disconnected components in a mask using connected components.
        """
        labeled_mask, num_instances = label(boolean_mask, structure=None if connectivity == 1 else np.ones((3, 3)))
        return labeled_mask

def store_image_with_orientation(original_image, orientation, filename = None):
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
    if filename is not None:
        cv2.imwrite(filename, image)
        print(f"Image saved with orientation at {filename}")
    return image

def store_image_with_point(original_image, point, filename = None):
    """
    Saves an image with a marked point.
    """
    image = original_image.copy()
    cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
    if filename is not None:
        cv2.imwrite(filename, image)
        print(f"Image saved with point at {filename}")
    return image

def store_image_with_gripper_bbox(original_image, gripper_bbox, filename = None, color=(0, 255, 0)):
    """
    Saves an image with the gripper bounding box drawn.
    """
    image = original_image.copy()
    cv2.polylines(image, [gripper_bbox.astype(np.int32)], isClosed=True,
                  color=color, thickness=2)
    if filename is not None:
        cv2.imwrite(filename, image)
        print(f"Image saved with gripper bounding box at {filename}")
    return image

def store_image_with_points(original_image, points, filename = None, color = (0, 255, 0)):
    """
    Saves an image with all the given grasping points (given as x y)
    """
    image = original_image.copy()
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, color, -1)
    if filename is not None:
        cv2.imwrite(filename, image)
        print(f"Image with points saved at {filename}")
    return image