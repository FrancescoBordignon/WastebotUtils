# 3d_tools
## 🎥 depth_reprojector
### 📄 Overview 
This is a c++ class  that, given a depth map, and the cameras intrinsics and extrinsics projects the depth map from camera 1 image plane to camera 2 image plane. The class is also binded using pybind in order to be usable in a python program.
### 📦 Installation  
- Install pybind 
```bash
pip install pybind11
```
- Build 
``` bash
cd  3d_tools/
python3 setup.py build_ext --inplace
```
- The class is now ready to be used in a python program with
``` bash
from depth_reprojector import DepthReprojector, Size
```
### 🛠️ Usage
```bash
from depth_reprojector import DepthReprojector, Size

reprojector = DepthReprojector(
    K1 = np.array([
        [751.4, 0.0, 644.3],
        [0.0, 751.2, 342.1],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64), 
    K2 = np.array([
        [2323.0, 0.0, 959.1],
        [0.0, 2322.7, 580.5],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64), 
    # Camera matrices, 1 refers to the source camera while 2 to the destination camera
    dim1 = Size(1280, 720),
    dim2 = Size(1280, 720),  
    # Dimesions of the images [width, height]
    dist1 = np.array([[0.08, -0.1, 0.0007, -0.0001, 0.04]], dtype=np.float64), 
    dist2 = np.zeros([1, 5], dtype=np.float64), 
    # Distortion coefficients in OPENCV format (optional)
    R = np.array([
        [ 0.9,  0.03 ,  0.01],
        [-0.03,  0.9,  0.1],
        [-0.006, -0.1 ,  0.9]
    ], dtype=np.float64), # Rotation matrix in meters (default: identity matrix)
    T = np.array([
        [ 0.09], # x
        [-0.07], # y
        [ 0.05]  # z
    ], dtype=np.float64) # Translation vector in meters (default: 0 0 0)
)
depth2 = reprojector.reprojectDepth(
    depth1 # souce depth map
    )
```
Then run with
```bash
cd 3d_tools
python3 -m your_script
```
### 🖼️ Example
A usage example can be foud in 3d_tools/examples. Change the parameters and the path to your input depth map. Then
```bash
cd 3d_tools
python3 -m examples.test_reprojector 
```
### 📝 Notes
- If you can't see the depth ensure R T and camera matrices are correctly set and that the distortion coefficients are in opencv format
- R T dist1 and dist2 are optional defalut values are the identity matrix 0 0 0 trnslation and 0 distortion
- T must be in the same measurement unit as the depth values (eg millimeters)
- The running time with 1200 x 1200 images on intel i5 processor is arround 35ms for the function reprojectDepth allowing for real time computation. If the function is used directly in a c++ program the inference time should be arround 27 ms in the same conditions
## 🎯 TODO
- For now giving dist2 doesn't have any effect, ideally the function distortImage2 should be implemented souch that it distorts the output depth map in order to have a depth map distorted as if it is seen from the distorted camera2 lens. For now image2 (the output depth map) is simpy undistorted which fits most cases.

## 👤 Authors
Francesco Bordignon
