#ifndef DEPTHREPROJECTOR_H
#define DEPTHREPROJECTOR_H

#include <iostream>
#include <vector>
#include <optional>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

class DepthReprojector {
private:
    cv::Mat K1;
    cv::Mat K2;
    cv::Mat dist1;
    cv::Mat dist2;
    cv::Mat R;
    cv::Mat T;
    cv::Size dim2;
    cv::Size dim1;
    cv::Mat map1, map2, map3, map4;
    cv::Mat rectified_image_1;
    cv::Mat depth2_raw;

    Eigen::Matrix3d K1_eigen;
    Eigen::Matrix3d K2_eigen;
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d T_eigen;
    Eigen::MatrixXd depth_1_eigen;
    Eigen::MatrixXd depth1_scaled;
    Eigen::Matrix2Xd uv1_matrix;   
    Eigen::VectorXd z_vector;        
    Eigen::Matrix3Xd uvz_points_matrix;
    Eigen::Matrix3d K1_eigen_inverse;
    Eigen::Matrix3Xd uv1_homogeneous; 
    Eigen::Matrix3Xd points2_3d;

    std::vector<Eigen::Vector3d> temp_points; 
    int num_points;

    void rectifyImage1(cv::Mat& image);

    void distortImage2(cv::Mat& image);

public:
    // Constructor
    DepthReprojector(
        const cv::Mat& K1,
        const cv::Mat& K2,
        const cv::Size& dim1,
        const cv::Size& dim2,
        const cv::Mat& dist1 = cv::Mat(),
        const cv::Mat& dist2 = cv::Mat(),
        const cv::Mat& R = cv::Mat(),
        const cv::Mat& T = cv::Mat()
    );

    cv::Mat reprojectDepth(cv::Mat& depth1_raw, double scale_factor);
};

#endif // DEPTHREPROJECTOR_H
