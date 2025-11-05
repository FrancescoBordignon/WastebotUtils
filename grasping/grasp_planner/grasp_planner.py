# Import required libraries
import cv2
import numpy as np
import math
from shapely.geometry import box
from shapely.affinity import rotate
from scipy.ndimage import label
import heapq

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

    def find_point_orientation(
        self,
        mask,
        point2d,
        gripper_width_x,
        gripper_height_y,
        scoring_type="rotated_boxes",
        max_grasping_masks=1,
        desired_object_importance=0.5,
        PCA_importance=0.0,
        depth_point=None,
        f_x=None,
        f_y=None,
        angle_rad_interval=math.pi / 18,
        gripper_fingers_percentage=0.15,
        min_object_size = 200
    ):
        """
        Determines the optimal 2D orientation for a robotic gripper to grasp an object at a given point.

        The function rotates the gripper mask around a specified 2D point (`point2d`) and evaluates each orientation
        based on the selected `scoring_type`. Higher scores represent better grasping configurations.

        Args:
            mask (np.ndarray): HxW Binary segmentation mask. Should distinguish between the target object and obstacles.
            point2d (tuple): Coordinates (x, y, ...) of the target grasping point on the mask.
            gripper_width_x (float): Width of the gripper (in pixels or physical units).
            gripper_height_y (float): Height of the gripper (in pixels or physical units).
            scoring_type (str): Scoring method to evaluate grasp orientation:
                - "three_parts_mask": Splits the gripper mask into palm and fingers (based on `gripper_fingers_percentage`).
                                    Penalizes finger overlap with the object and obstacle.
                                    Score = desired_object_importance * %palm-overlap-with-object - 
                                            (1 - desired_object_importance) * %fingers-and-palm-overlap-with-obstacles
                - "contact_points+three_parts": Same as "three_parts_mask", but also considers how many good contact points
                                                fall within the gripper area. a good contact point is a point on the border
                                                of the mask that is not confining with a obstacle
                - "rotated_boxes": Evaluates overlap as:
                                1 - (desired_object_importance * %gripper-overlapping-object +
                                    (1 - desired_object_importance) * %gripper-overlapping-obstacles)
            max_grasping_masks (int): Maximum number of top-scoring masks (orientations) to return.
            desired_object_importance (float): Weight [0, 1] determining how much to prioritize the object vs. obstacles in scoring.
            PCA_importance (float): value between 0 and 1 determines how important is the alignement of the gripper wrt the PCA of an instance
            depth_point (float, optional): Depth value at `point2d`. Used to scale gripper size appropriately if provided.
            f_x (int, optional): Horizontal focal length (in pixels) for depth scaling.
            f_y (int, optional): Vertical focal length (in pixels) for depth scaling.
            angle_rad_interval (float): Angular step (in radians) for rotating the gripper and evaluating orientation.
            gripper_fingers_percentage (float): Ratio of the gripper mask allocated to fingers (vs. palm) for scoring purposes.
            min_object_size (int): pixels of minimum spickable object size

        Returns:
            best_masks List[Tuple[float, float, np.ndarray]]: A list of up to `max_grasping_masks` elements, each being a tuple:
                - score (float): Score of the orientation.
                - angle_rad (float): Angle (in radians) of the tested orientation. In opencv reference frame counter clockwise
                - original_size_mask (np.ndarray): HxW original mask oriented with angle 0
                - point2d (tuple): Coordinates (x, y) of the target grasping point on the mask. 
                or None if nothing found
                
        """
        # Sanity checks
        desired_object_importance = min(desired_object_importance,1.0)
        desired_object_importance = max(desired_object_importance,0.0)
        PCA_importance = min(PCA_importance,1.0)
        PCA_importance = max(PCA_importance,0.0)
        gripper_fingers_percentage = min(gripper_fingers_percentage,0.5)
        gripper_fingers_percentage = max(gripper_fingers_percentage,0.0)
        if gripper_width_x <= 0 or gripper_height_y <= 0:
            raise ValueError("Gripper size must be > 0")
        if depth_point is not None:
            if f_x is None or f_y is None:
                raise ValueError("Depth given without focal lengths")
            if depth_point <= 0:
                raise ValueError("Depth must be > 0")
            if f_x <= 0 or f_y <= 0:
                raise ValueError("Focal lengths must be > 0")
        if angle_rad_interval <= 0:
            raise ValueError("angle_rad_interval must be > 0")

        # Extract object class mask for the given point
        x,y = int(point2d[0]), int(point2d[1])

        # Take wanted object mask
        mask_obj_bool = self.__extract_instance_from_point(mask, (x,y))
        mask_obj = mask_obj_bool.astype(np.uint8)

        if np.count_nonzero(mask_obj) < min_object_size:
            return None

        # Take the collision mask
        mask_collisions = ((~mask_obj_bool) & (mask > 0)).astype(np.uint8)
        #print("time to np where: ", time.time() - start)

        # Compute dimension of the gripper based on the point depth if given, else keep pixels gripper dimensions
        if depth_point is not None:
            edge_width, edge_height = self.__compute_gripper_bbox_scale(depth_point, gripper_width_x, gripper_height_y, f_x, f_y,)
        else:
            edge_width = gripper_width_x
            edge_height = gripper_height_y
        

        # If the point is too close to the border just discard it TODO implement custom logic
        min_dist_from_edge  = max(edge_width//2, edge_height//2)
        if (x < min_dist_from_edge or x > mask.shape[1] - min_dist_from_edge or
            y < min_dist_from_edge or y > mask.shape[0] - min_dist_from_edge):
            return None


        #start = time.time()
        # Draw mask depending on scoring_type
        if scoring_type == "three_parts_mask" or scoring_type == "contact_points+three_parts":
            # mask divided in three and asign to each pixel a value 1,2,3 --> 1 = finger1, 2 = palm, 3 = finger2
            side_width = int(edge_width * gripper_fingers_percentage)
            mask_gripper_original = self.__generate_split_rectangular_mask(image_size = mask.shape, center = (x,y), rect_width = edge_width, rect_height = edge_height, side_width = side_width)
        elif scoring_type == "rotated_boxes":
            mask_gripper_original = self.__create_centered_rectangle_mask(image_shape = mask.shape[:2], center = (x,y), width = edge_width, height = edge_height, value=1)
        
        # Resize masks to speed up computation
        original_mask_shape = mask.shape[:2]
        scale = 128 / min(original_mask_shape)
        new_w = int(round(original_mask_shape[1] * scale))
        new_h = int(round(original_mask_shape[0] * scale))
        x_new = x * (new_w / original_mask_shape[1])
        y_new = y * (new_h / original_mask_shape[0])
        mask_collisions = cv2.resize(mask_collisions, (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_obj = cv2.resize(mask_obj.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_gripper = cv2.resize(mask_gripper_original.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        best_masks = [] 

        # Compute PCA if required the distance of the guessed angle from the PCA angle will be used to compute the score
        # of a given orientation
        if PCA_importance > 0.0:
            initial_guess_angle = self.__compute_PCA(mask_obj)
        else:
            initial_guess_angle = None
        if scoring_type == "three_parts_mask" or scoring_type == "contact_points+three_parts":
            palm_area = np.sum(mask_gripper == 2)
            fingers_area = fingers_area = ((mask_gripper == 1) | (mask_gripper == 3)).sum()
        
        if scoring_type == "contact_points+three_parts":
            contour_points = self.sample_contour_points(mask_obj, step=10, label=None) 
            # For each point check if it overlaps (more than 40% an object from amother mask)
            good_points = self.check_contact_points_overlapping(mask_collisions, contour_points, kernel_size=10, threshold=0.4)
            tot_points = len(good_points)
        if scoring_type == "rotated_boxes":
            gripper_area = np.sum(mask_gripper>0)
        # Rotate them and compute score
        for angle_offset in np.arange(-math.pi / 2,  math.pi / 2 + angle_rad_interval, angle_rad_interval):

            # Rotate the mask
            angle_rad = angle_offset
            rotated_gripper_mask = self.rotate_mask(mask_gripper, angle_rad, center_point = (x_new,y_new))

            # Compute Score (always between 0 and 1) greater values correspond to better values
            if scoring_type == "rotated_boxes":
                score = self.compute_orientation_score(mask_obj, mask_collisions, rotated_gripper_mask, desired_object_importance, gripper_area)
            
            elif scoring_type == "three_parts_mask":
                score = self.evaluate_score_three_layers(mask_obj,mask_collisions,rotated_gripper_mask, desired_object_importance, palm_area, fingers_area)
            elif scoring_type == "contact_points+three_parts":
                if tot_points > 0:
                    score = (self.count_points_inside_mask(rotated_gripper_mask, good_points)/tot_points)*self.evaluate_score_three_layers(mask_obj, mask_collisions, rotated_gripper_mask, desired_object_importance, palm_area, fingers_area)
                else:
                    score = 0.0
            if score > 0.0:
            # Adjust score based on pca (if required) (best score when pca is aligned with short gripper edge)
                if initial_guess_angle is not None:
                    alignement_score = self.__angular_similarity_90_deg(initial_guess_angle + (np.pi/2), angle_rad)
                    score = (score+ PCA_importance*alignement_score)/(1+alignement_score)
                entry = (score, angle_rad, mask_gripper_original, point2d)
                if len(best_masks) < max_grasping_masks:
                    heapq.heappush(best_masks, entry)
                else:
                    # Only keep if better than the worst in heap (min-heap stores smallest score at top)
                    if score > best_masks[0][0]:
                        heapq.heappushpop(best_masks, entry)

        # best mask is at index 0
        best_masks = sorted(best_masks, reverse=True)  
        
        return best_masks
    
    ### UTILS ###
    
    def __get_rotated_rectangle_corners(self, center, width, height, angle_rad):
        """
        Calculate the corner points of a rotated rectangle in OpenCV's coordinate system.
        
        Args:
            center (tuple): (cx, cy) coordinates of the rectangle center.
            width (float): Width of the rectangle.
            height (float): Height of the rectangle.
            angle_rad (float): Rotation angle in radians (CCW in OpenCV frame).

        Returns:
            np.ndarray: 4x1x2 array of corner points (int32) in OpenCV format.
        """
        cx, cy = center
        w = width / 2
        h = height / 2
        angle_rad = -angle_rad # counter clockwise in opencv format

        # Define corners in OpenCV reference frame (+Y is down)
        corners = np.array([
            [-w,  h],  # top-left
            [ w,  h],  # top-right
            [ w, -h],  # bottom-right
            [-w, -h]   # bottom-left
        ])

        # Rotation matrix (standard 2D rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])

        # Rotate and translate
        rotated_corners = corners @ R.T + np.array([cx, cy])

        # Return in OpenCV contour format
        return rotated_corners.astype(np.int32).reshape((-1, 1, 2))

    def count_points_inside_mask(self, mask, points):
        """
        Counts how many (x, y) points fall inside a non-zero region of a mask.

        Args:
            mask (np.ndarray): 2D binary or labeled mask (0 = background).
            points (list of tuples): List of (x, y) coordinates.

        Returns:
            int: Number of points inside the mask.
            list: List of booleans, True if point is inside.
        """
        h, w = mask.shape
        inside_flags = []

        for x, y in points:
            x = int(round(x))
            y = int(round(y))

            if 0 <= x < w and 0 <= y < h:
                inside = mask[y, x] > 0  # Note: OpenCV uses (x, y), NumPy uses [row, col] = [y, x]
            else:
                inside = False

            inside_flags.append(inside)

        count_inside = sum(inside_flags)
        return count_inside

    def check_contact_points_overlapping(self, mask, points, kernel_size=10, threshold=0.4):
        """
        Checks whether the given points are inside the mask based on coverage in a kernel.

        Args:
            mask (np.ndarray): 2D binary mask (0/1 or 0/255).
            points (list of tuples): List of (x, y) points.
            kernel_size (int): Size of the square window (must be even for centering).
            threshold (float): Fraction of mask coverage required to consider "inside".

        Returns:
            List[bool]: True if the mask covers more than `threshold` inside the window around each point.
        """
        assert mask.ndim == 2, "Mask must be 2D"
        h, w = mask.shape
        half_k = kernel_size // 2

        # Normalize mask to 0 or 1
        bin_mask = (mask > 0).astype(np.uint8)

        results = []

        for (x, y) in points:
            # Convert to int just in case
            x, y = int(x), int(y)

            # Define window bounds, clipping to image size
            x1 = max(x - half_k, 0)
            x2 = min(x + half_k, w)
            y1 = max(y - half_k, 0)
            y2 = min(y + half_k, h)

            window = bin_mask[y1:y2, x1:x2]
            total_pixels = window.size
            mask_pixels = np.count_nonzero(window)

            ratio = mask_pixels / total_pixels if total_pixels > 0 else 0
            if (ratio <= threshold):
                results.append((x, y))

        return results
    
    def sample_contour_points(self, mask, step=10, label=None):
        """
        Samples points every `step` pixels along the contour of a binary or labeled mask.

        Args:
            mask (np.ndarray): 2D binary or labeled mask.
            step (int): Sampling step in pixels along the contour length.
            label (int or None): If mask is labeled, extract contour only of this label.
        
        Returns:
            List of (x, y) tuples: Sampled points along the contour.
        """
        assert mask.ndim == 2, "Mask must be 2D"

        # If labeled mask and label is specified, extract binary mask of that label
        if label is not None:
            mask_bin = (mask == label).astype(np.uint8)
        else:
            mask_bin = (mask > 0).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return []

        # Choose the longest contour (assuming that's the object border)
        contour = max(contours, key=lambda c: cv2.arcLength(c, closed=True))

        # Flatten and convert to list of (x, y)
        contour_points = contour[:, 0, :]  # shape: (N, 2)

        # Sample every `step` points
        sampled = contour_points[::step]

        return [tuple(pt) for pt in sampled]
    
    def __compute_PCA(self,mask_object):
        # Computes PCA of a mask and returns the angle
        # in opencv coordinates counter clockwise
        contours, _ = cv2.findContours((mask_object.astype(np.uint8)) * 255,
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        data_pts = contour.reshape(-1, 2).astype(np.float32)
        _, eigenvectors = cv2.PCACompute(data_pts, mean=np.array([]))
        direction = eigenvectors[0]
        angle_rad = -math.atan2(direction[1], direction[0])
        return angle_rad

    def __extract_instance_from_point(self,mask, point):
        """
        Given a multi-class mask and a point, extract the connected object (as a binary mask)
        to which the point belongs.

        Args:
            mask (np.ndarray): 2D array of class IDs (H x W), dtype = int or uint8.
            point (tuple): (x, y) coordinates.

        Returns:
            np.ndarray: Binary mask (bool) of the connected object containing the point.
        """
        x, y = point
        class_id = mask[y, x]  # Note: mask[y, x] due to image coordinate order

        # Create a mask for floodFill with 2 extra pixels in each dimension (required by OpenCV)
        h, w = mask.shape
        floodfill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Make a copy of the mask to flood fill into (cv2 modifies it)
        temp_mask = (mask == class_id).astype(np.uint8)  # 1 where class matches, 0 elsewhere

        # Flood fill starting from the point, setting connected region to 255
        flooded = temp_mask.copy()
        cv2.floodFill(flooded, floodfill_mask, seedPoint=(x, y), newVal=255)

        # Extract only the flooded region (was 1, now 255)
        instance_mask = (flooded == 255)

        return instance_mask

    def __angular_similarity_90_deg(self,a, b):
        # a, b should be in radians
        # Compute angular distance in range [0, π/2]
        diff = abs((a - b + (np.pi/2)) % (np.pi) - np.pi/2)
        # Normalize: 0 distance → 1 = 100% similarity, π/2 distance → 0 = 0% similarity
        return 1 - (diff / (np.pi/2))

    
    def rotate_mask(self,mask, angle_radians, center_point):
        """
        Rotates a 2D mask around a specific point by a given angle in radians,
        using OpenCV's coordinate system (origin at top-left, y-axis down).

        Args:
            mask (np.ndarray): 2D binary or labeled mask (H, W).
            angle_radians (float): Rotation angle in radians (positive = counter-clockwise).
            center_point (tuple): (x, y) point to rotate around.

        Returns:
            np.ndarray: Rotated mask.
        """
        assert mask.ndim == 2, "Mask must be 2D (H, W)"
        h, w = mask.shape

        # Convert radians to degrees
        angle_degrees = np.degrees(angle_radians)

        # Compute rotation matrix
        rot_mat = cv2.getRotationMatrix2D(center_point, angle_degrees, scale=1.0)

        # Rotate the mask
        rotated_mask = cv2.warpAffine(
            mask,
            rot_mat,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )

        return rotated_mask

    def evaluate_score_three_layers(self, mask_obj, mask_collisions, mask_gripper, desired_object_importance, palm_area, fingers_area):
        intersection_palm_object = np.count_nonzero(mask_obj[mask_gripper==2])
        object_area = np.count_nonzero(mask_obj)
        intersection_all_collision = np.count_nonzero(mask_collisions[mask_gripper>0])
        intersection_fingers_object = np.count_nonzero(mask_obj & ((mask_gripper == 1) | (mask_gripper == 3)))
                                  
        if palm_area == 0:
            intersection_palm_object_score = 0.0
        else:
            intersection_palm_object_score = max(intersection_palm_object / palm_area, intersection_palm_object / object_area)
        if fingers_area == 0:
            intersection_fingers_object_score = 0.0
        else:
            intersection_fingers_object_score = intersection_fingers_object / fingers_area
        intersection_all_collision_score = (intersection_all_collision)/ (palm_area + fingers_area)

        #if intersection of fingers with parlm is even slightly high, return zero
        if intersection_fingers_object_score > 0.1:
            return 0
        obstacles_avoidance_inportance = 1 - desired_object_importance
        return max(intersection_palm_object_score*desired_object_importance - intersection_all_collision_score*obstacles_avoidance_inportance, 0)

    def __generate_split_rectangular_mask(self, image_size, center, rect_width, rect_height, side_width):
        """
        Generate a labeled rectangular mask with 3 horizontal regions.
        
        Args:
            image_size (tuple): (height, width) of the full mask.
            center (tuple): (x, y) center of the rectangle.
            rect_width (int): total width of the rectangle.
            rect_height (int): total height of the rectangle.
            side_width (int): width of left and right labeled parts.
        
        Returns:
            np.ndarray: mask of shape (H, W) with labels 0 (background), 1 (left), 3 (center), 2 (right).
        """
        H, W = image_size
        mask = np.zeros((H, W), dtype=np.uint8)

        cx, cy = center

        # Compute bounds of the rectangle
        x2 = int(cx + rect_width // 2)
        x1 = int(cx - rect_width // 2)
        y1 = int(cy - rect_height // 2)
        y2 = int(cy + rect_height // 2)

        # Clamp to image boundaries
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)

        # Define inner region divisions
        left_x2 = min(x1 + side_width, x2)
        right_x1 = max(x2 - side_width, x1)
        
        # Draw left part (label 1)
        if left_x2 > x1:
            mask[y1:y2, x1:left_x2] = 1

        # Draw right part (label 2)
        if x2 > right_x1:
            mask[y1:y2, right_x1:x2] = 3

        # Draw center part (label 3)
        if right_x1 > left_x2:
            mask[y1:y2, left_x2:right_x1] = 2

        return mask

    def __create_centered_rectangle_mask(self, image_shape, center, width, height, value=1):
        """
        Creates a binary mask with a filled rectangle centered at a given point.

        Args:
            image_shape (tuple): (height, width) of the mask to create.
            center (tuple): (x, y) center of the rectangle.
            width (int): Width of the rectangle in pixels.
            height (int): Height of the rectangle in pixels.
            value (int): Pixel value to fill in the rectangle (default 1).

        Returns:
            np.ndarray: Binary mask with the rectangle.
        """
        h_img, w_img = image_shape
        mask = np.zeros((h_img, w_img), dtype=np.uint8)

        cx, cy = center
        half_w = width // 2
        half_h = height // 2

        # Top-left and bottom-right corners
        x1 = int(max(cx - half_w, 0))
        y1 = int(max(cy - half_h, 0))
        x2 = int(min(cx + half_w, w_img - 1))
        y2 = int(min(cy + half_h, h_img - 1))
        # Draw filled rectangle
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=value, thickness=-1)

        return mask

    def compute_orientation_score(self, mask_object, mask_obstacles, mask_gripper, desired_object_importance, gripper_area):
        # Directly compute overlaps using logical_and to avoid allocating multiple masks
        intersection_obstacles = np.logical_and(mask_gripper, mask_obstacles)
        intersection_object = np.logical_and(mask_gripper, mask_object)

        # Count overlapping pixels efficiently (non-zero entries)
        obstacle_overlap = np.count_nonzero(intersection_obstacles)/gripper_area
        object_overlap = np.count_nonzero(intersection_object)/gripper_area
        obstacole_avoidance_percentage = 1-desired_object_importance
        return 1 - (obstacle_overlap*obstacole_avoidance_percentage + object_overlap*desired_object_importance)

    def __find_gripper_edges(self, point_camera_frame, gripper_width_x, gripper_height_y,
                              depth_point, angle_rad, f_x, f_y):
        """
        Returns 4-corner polygon of gripper rotated at angle_rad around the point.
        """
        x_center, y_center = point_camera_frame
        if depth_point is not None:
            edge_length, edge_height = self.__compute_gripper_bbox_scale(depth_point, gripper_width_x, gripper_height_y, f_x, f_y,)
        else:
            edge_length = gripper_width_x
            edge_height = gripper_height_y

        bbox = box(x_center - edge_length / 2, y_center - edge_height / 2,
                   x_center + edge_length / 2, y_center + edge_height / 2)
        rotated_bbox = rotate(bbox, angle_rad, origin=(x_center, y_center), use_radians=True)
        edges = np.array(rotated_bbox.exterior.coords, dtype=np.int32)[:4]
        return np.array(edges, dtype=np.int32).reshape((-1, 1, 2))
    
    def __compute_gripper_bbox_scale(self, depth_point, gripper_width_x, gripper_height_y, f_x, f_y,):
        return (gripper_width_x / depth_point) * f_x, (gripper_height_y / depth_point) * f_y

    def __find_gripper_edges_fast(self, point_camera_frame, gripper_width_x, gripper_height_y,
                               depth_point, angle_rad, f_x, f_y):
        x_center, y_center = point_camera_frame

        if depth_point is not None:
            edge_length = (gripper_width_x / depth_point) * f_x
            edge_height = (gripper_height_y / depth_point) * f_y
        else:
            edge_length = gripper_width_x
            edge_height = gripper_height_y

        # Define corners relative to center (before rotation)
        dx = edge_length / 2
        dy = edge_height / 2
        corners = np.array([
            [-dx, -dy],
            [ dx, -dy],
            [ dx,  dy],
            [-dx,  dy]
        ])

        # Rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]])

        # Rotate and translate to center
        rotated = corners @ R.T + np.array([x_center, y_center])

        return rotated.astype(np.int32).reshape((-1, 1, 2))

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
    gripper bbox can either be a 0,n mask or a polygon
    """
    image = original_image.copy()
    if isinstance(gripper_bbox, np.ndarray) and gripper_bbox.ndim == 2 and gripper_bbox.dtype in [np.uint8, bool, int]:
        gripper_bbox = gripper_bbox.astype(np.uint8)
        contours, _ = cv2.findContours(gripper_bbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        polygon = largest.squeeze()
        if polygon.ndim == 1:
            polygon = polygon[np.newaxis, :]
        # convert polygon to np.ndarray (int32) instead of list
        gripper_bbox = np.array(polygon, dtype=np.int32)

    cv2.polylines(image, [gripper_bbox], isClosed=True,
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
