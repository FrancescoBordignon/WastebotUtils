import cv2
import numpy as np
import math
from shapely.geometry import box
from shapely.affinity import rotate


class GripperOrientator:
    """
    Computes the optimal orientation of a robotic gripper relative to an object mask,
    aiming to avoid obstacles detected in a scene.

    Attributes:
        short_fingers_aperture_factor (float): Aperture scaling factor for short fingers.
        long_fingers_aperture_factor (float): Aperture scaling factor for long fingers.
        edge_length (float): Fixed gripper length (optional).
        edge_height (float): Fixed gripper height (optional).
    """

    def __init__(self, short_fingers_aperture_factor=None, long_fingers_aperture_factor=None,
                 edge_length=None, edge_height=None):
        self.short_fingers_aperture_factor = short_fingers_aperture_factor
        self.long_fingers_aperture_factor = long_fingers_aperture_factor
        self.edge_length = edge_length
        self.edge_height = edge_height

    def find_orientation(self, mask, point2d, panoptic_map, depth_point=None,
                         angle_rad_inerval=math.pi / 18):
        """
        Finds the optimal gripper orientation around a given grasping point.

        Args:
            mask (np.ndarray): Binary mask (1-channel) of the target object.
            point2d (tuple): Pixel coordinates (x, y) of the grasping point.
            panoptic_map (list[dict]): List of instance dicts, each with
                'segmentation' as a boolean mask (e.g., SAM output).
            depth_point (float, optional): Depth at grasp point (needed if aperture scaling factors are used).
            angle_rad_inerval (float): Angle step (in radians) for searching orientations.

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
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        data_pts = contour.reshape(-1, 2).astype(np.float32)
        _, eigenvectors = cv2.PCACompute(data_pts, mean=np.array([]))
        direction = eigenvectors[0]
        init_angle_rad = math.atan2(direction[1], direction[0])

        # Initial bounding box based on PCA angle
        edges = self.find_gripper_edges(point2d, depth_point, init_angle_rad)
        mask_gripper = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(mask_gripper, [edges], color=1)
        best_score = self.compute_score(mask_gripper, mask_obstacles)
        best_angle_rad = init_angle_rad
        best_bounding_box = edges

        # Search for orientation with minimal obstacle overlap
        for angle_offset in np.arange(-math.pi / 2, math.pi / 2, angle_rad_inerval):
            angle_rad = init_angle_rad + angle_offset
            edges = self.find_gripper_edges(point2d, depth_point, angle_rad)
            mask_gripper = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(mask_gripper, [edges], color=1)

            score = self.compute_score(mask_gripper, mask_obstacles)
            if score < best_score:
                best_score = score
                best_angle_rad = angle_rad
                best_bounding_box = edges

        vec_camera = np.array([np.cos(best_angle_rad), np.sin(best_angle_rad), 0])

        return (vec_camera, -vec_camera), best_bounding_box

    def compute_score(self, mask_gripper, mask_obstacles):
        """
        Computes score = (area of gripper mask overlapping obstacles) / (total gripper area).
        Lower is better.
        """
        intersection = mask_gripper[mask_obstacles]
        return np.sum(intersection) / np.sum(mask_gripper)

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
        if self.edge_height and self.edge_length:
            edge_length = self.edge_length
            edge_height = self.edge_height
        elif self.short_fingers_aperture_factor and self.long_fingers_aperture_factor:
            edge_length = self.short_fingers_aperture_factor * depth_point * 2
            edge_height = self.long_fingers_aperture_factor * depth_point * 2

        # Create and rotate rectangle
        bbox = box(x_center - edge_length / 2, y_center - edge_height / 2,
                   x_center + edge_length / 2, y_center + edge_height / 2)
        rotated_bbox = rotate(bbox, angle_rad, origin=(x_center, y_center), use_radians=True)
        return np.array(rotated_bbox.exterior.coords, dtype=np.int32)


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
def main():

    #  _____ _           _                 _ _    __       _ _        _ 
    # |_   _| |__   __ _| |_   ___    __ _| | |  / _| ___ | | | _____| |
    #   | | | '_ \ / _` | __| / __|  / _` | | | | |_ / _ \| | |/ / __| |
    #   | | | | | | (_| | |_  \__ \ | (_| | | | |  _| (_) | |   <\__ \_|
    #   |_| |_| |_|\__,_|\__| |___/  \__,_|_|_| |_|  \___/|_|_|\_\___(_)

    return 0

if __name__ == "__main__":
    main()
