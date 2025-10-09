# 🤖 GraspPlanner

## 📄 Overview 

The `GraspPlanner` class provides tools to compute optimal 2D grasping points and orientations for robotic grippers based on segmentation masks and (optionally) depth maps.

### 🔍 Key Features

- **Grasp Point Detection**: Uses morphological erosion to detect stable grasp points within segmented objects.
- **Scoring and Ranking**: Scores grasp points using one of three criteria:
  - `bigger_area`: Prefer larger object regions.
  - `depth_max`: Prefer deeper regions (requires depth map).
  - `depth_min`: Prefer shallower regions (requires depth map).
- **Orientation Estimation**: Finds the best gripper orientation rotating the gripper bbox around the grasp point (at centre) using PCA and collision-aware scoring.

---


## 📦 Installation 
Install dependencies using `pip`:

```bash
pip install opencv-python numpy shapely scipy
```

## 🛠️ Usage
1. Initialize the Grasp Planner
```bash
from grasp_planner import GraspPlanner
planner = GraspPlanner()
```
2. 🎯 Find grasping points
```bash
points = planner.find_grasping_points(
    mask=segmentation_mask,
    desired_class=1,                  # Optional: target object class
    criterion="bigger_area",          # Options: "bigger_area", "depth_max", "depth_min"
    depth=depth_map                   # Optional: required for depth-based scoring
)
```
It returns an Nx4 np.array containing points in this format [x,y,class,score] sorted by score in descending order. The scoring depends on criterion and is between 0 and 1.


3. 🧭 Compute Best Gripper Orientation

Computes the best gripper orientation to grasp an object at a specific 2D point in a segmentation mask.
Evaluates different angles by rotating a virtual gripper mask and scoring overlap with the object and obstacles.
Supports several scoring methods, optional PCA alignment, and depth-based size scaling.
```bash
best_masks = planner.find_point_orientation(
    mask=segmentation_mask,             # 2D binary mask (HxW, np.uint8). Foreground is the object to grasp (non-zero), 
                                        # background is zero. Can include obstacles in non-zero values too.
    point2d=(x, y),                      # Grasp point (x, y) on the mask. Only this point is needed.
    
    gripper_width_x=0.4,                # Gripper width (in meters or pixels, depending on whether depth_point is given).
    gripper_height_y=0.2,               # Gripper height (same unit as width).

    scoring_type="rotated_boxes",       # Gripper orientation evaluation method:
                                        #   - "rotated_boxes": Simple overlap-based score
                                        #   - "three_parts_mask": Finger/palm overlap penalties
                                        #   - "contact_points+three_parts": Adds contact point quality to scoring

    max_grasping_masks=1,               # Maximum number of top orientations to return (sorted by score descending).

    desired_object_importance=0.5,      # Trade-off between object and obstacle overlap [0 (obstacles) to 1 (object)].
    PCA_importance = 0.1,               # value between 0 and 1 determines how important is the alignement of the gripper wrt the PCA of an                                  
                                        # instance If True, favors alignment with the object’s PCA direction.

    depth_point=0.4,                    # Optional: Depth at grasp point (same unit as gripper size). If None, width/height are in pixels.
    f_x=500,                            # Optional: Camera intrinsics (fx) for scaling gripper size from depth.
    f_y=500,                            # Optional: Camera intrinsics (fy) for scaling gripper size from depth.

    angle_rad_interval=math.pi / 18,    # Angular resolution of the rotation search (smaller = finer search).
    gripper_fingers_percentage=0.15     # Fraction of the gripper allocated to fingers vs. palm (for scoring).
)
```
It returns
```bash
best_masks: List[Tuple[float, float, np.ndarray]]: A list of up to         
                                                   max_grasping_masks  elements, each being a tuple:
                - score (float): Score of the orientation.
                - angle_rad (float): Angle (in radians) of the tested orientation. In opencv reference frame counter clockwise
                - original_size_mask (array): HxW original mask oriented with angle 0
```               

gripper_bbox: (polygon) rotated bounding box of the gripper 
## 🖼️ Visualization Utilities
Draw a Grasp Point
```bash
image_with_points = store_image_with_point(image, points=[(x, y),(x2,y2)], filename="point.png")
```
Draw Gripper Orientation
```bash
image_with_orientation = store_image_with_orientation(image, orientation=vector, filename="orientation.png")
```
Draw Gripper Bounding Box
```bash
image_with_gripperbbox = store_image_with_gripper_bbox(image, gripper_bbox=(polygon or mask), filename="bbox.png")
```
## Example
There is a testing example where parameters can be tuned that generates random segmentation masks and generates grasping point + generates orientation. To run
```bash
cd grasping
python -m examples.test_grasp_planner
```
## 📝 Notes
- Segmentation mask must use integer class labels. Class 0 is treated as background. The classes can also not be contiguous like [1,2,3,4] but also [56,32,8] just make sure classes stay in the range 8-bit unsigned int
- Depth-based orientation scoring requires valid depth_point, f_x, and f_y.
- Depth-based grasping point detection can be done with "depth_max", "depth_min" criteria but a valid depth map must be given
- To find orientation, the 2D point can also just be [x,y] any other field is ignored [x,y,...etc...]
### 👤 Authors
Francesco Bordignon 