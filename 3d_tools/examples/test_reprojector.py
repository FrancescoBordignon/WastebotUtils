from depth_reprojector import DepthReprojector, Size
import numpy as np
import cv2
import time


def main():
    depth_path = "/home/ecobot-3/Desktop/datasets/weight_test_set_IWAMO/depth0/000037.png"
    depth1 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth1 = depth1.astype(np.float64)
    
    K1 = np.array([
        [751.4531860351562, 0.0, 644.3215942382812],
        [0.0, 751.22265625, 342.1682434082031],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    K2 = np.array([
        [2323.054401884099, 0.0, 959.1673606011286],
        [0.0, 2322.786050857415, 580.5408611613744],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist1 = np.array([[0.08185149799312205, -0.1103452261532463, 0.0007392139314295011, -0.0001122657490592643, 0.04707396660353352]], dtype=np.float64)
    dist2 = np.zeros((1, 5), dtype=np.float64)
    R = np.array([
        [ 0.99948576,  0.0300562 ,  0.01117276],
        [-0.03139111,  0.98823577,  0.14968185],
        [-0.00654245, -0.1499556 ,  0.98867108]
    ], dtype=np.float64)

    T = np.array([
        [ 96.49281],
        [-78.45502],
        [ 52.30814]
    ], dtype=np.float64)

    dim1 = Size(1280, 720) #width, height
    dim2 = Size(1920, 1200) # width, height


    reprojector = DepthReprojector(
        K1, K2, dim1,dim2, R = R, T = T
    )
    start = time.time()
    depth2 = reprojector.reprojectDepth(depth1)
    print("required_time: ", time.time() -start)
    depth2_vis = cv2.normalize(depth2, None, 0, 255, cv2.NORM_MINMAX)
    depth2_vis = depth2_vis.astype(np.uint8)

    cv2.namedWindow("Reprojected Depth", cv2.WINDOW_NORMAL)  # Make window resizable
    cv2.imshow("Reprojected Depth", depth2_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
