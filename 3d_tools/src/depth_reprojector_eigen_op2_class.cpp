#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <optional>
#include "depth_reprojector_eigen_op2_class.hpp"


// compile with g++ -O3 -march=native -ffast-math -DNDEBUG     depth_reprojector_eigen_op1_class.cpp -o depth_rep     `pkg-config --cflags --libs opencv4`     -I /usr/include/eigen3
// execution time: 79 ms on: Intel® Core™ i5-8279U × 8 16GB RAM

// Constructor
DepthReprojector::DepthReprojector(        
            const cv::Mat& K1,
            const cv::Mat& K2,
            const cv::Size& dim1,
            const cv::Size& dim2,
            const cv::Mat& dist1, 
            const cv::Mat& dist2,
            const cv::Mat& R,
            const cv::Mat& T
            )
{
    // Convert and store camera matrices as double precision
    cv::Mat K1_double, K2_double;
    K1.convertTo(K1_double, CV_64F);
    K2.convertTo(K2_double, CV_64F);

    // Store original inputs (if needed)
    this->K1 = K1_double;
    this->K2 = K2_double;
    this->dim1 = dim1;
    this->dim2 = dim2;
    
    // Convert R and T if non-empty
    if (!R.empty()) {
        R.convertTo(this->R, CV_64F);
    } else {
        this->R = cv::Mat::eye(3, 3, CV_64F); // Identity by default
    }

    if (!T.empty()) {
        T.convertTo(this->T, CV_64F);
    } else {
        this->T = cv::Mat::zeros(3, 1, CV_64F); // Zero translation by default
    }

    cv::cv2eigen(K1_double, K1_eigen);
    cv::cv2eigen(K2_double, K2_eigen);
    if (!dist1.empty() && cv::countNonZero(dist1)>0) {
        this->dist1 = dist1.clone();
        cv::initUndistortRectifyMap(this->K1, this->dist1, cv::Mat(), this->K1, this->dim1, CV_32FC1, this->map1, this->map2);
    } else {
        this->dist1 = cv::Mat();
    }

    if (!dist2.empty() && cv::countNonZero(dist2)>0) {
        this->dist2 = dist2.clone();
    } else {
        this->dist2 = cv::Mat();
    }
    if (!this->R.empty()) {
        cv::cv2eigen(this->R, this->R_eigen);
    } else {
        this->R_eigen = Eigen::Matrix3d::Identity();
    }
    if (!this->T.empty()) {
        CV_Assert(this->T.rows == 3 && this->T.cols == 1 && this->T.type() == CV_64F);
        cv::cv2eigen(this->T, this->T_eigen);
    } else {
        this->T_eigen = Eigen::RowVector3d::Zero();
    }
    this->K1_inverse_eigen = K1_eigen.inverse();

    // create the 4x4 Reprojection matrix
    this->RepMat_eigen = Eigen::Matrix4d::Zero();
    this->RepMat_eigen.block<3,3>(0,0) = this-> R_eigen;
    this->RepMat_eigen.block<3,1>(0,3) = this->T_eigen;
    this->RepMat_eigen(3,3) = 1.0;

    Eigen::Matrix4d  K1_inverse_eigen_new = Eigen::Matrix4d::Zero();
    K1_inverse_eigen_new.block<3,3>(0,0) = this->K1_inverse_eigen;
    K1_inverse_eigen_new(3,3) = 1.0;

    this->RepMat_eigen = this-> RepMat_eigen * K1_inverse_eigen_new;
    
    Eigen::Matrix4d  K2_eigen_new = Eigen::Matrix4d::Zero();
    K2_eigen_new.block<3,3>(0,0) = this->K2_eigen;
    K2_eigen_new(3,3) = 1.0;
    this->RepMat_eigen = K2_eigen_new * this->RepMat_eigen;
    
    num_points = this->dim1.height * this->dim1.width;
    this->points1_eigen.resize(4, num_points);
    this->points1_eigen.setOnes();
    this->points2_eigen.resize(4, num_points);
    this->depth_1_eigen.resize(this->dim1.height, this->dim1.width);
    this->depth2_raw = cv::Mat::zeros(dim2, CV_64FC1); 

}
void DepthReprojector::rectifyImage1(cv::Mat& image)
{
    cv::remap(image, this->rectified_image_1, this->map1, this->map2, cv::INTER_NEAREST);
    cv::cv2eigen(this->rectified_image_1, this->depth_1_eigen);
}

void DepthReprojector::distortImage2(cv::Mat& image)
{
    std::cout<<" distortImage2 TODO "<<std::endl;
}

cv::Mat DepthReprojector::reprojectDepth(cv::Mat& depth1_raw)
{
    //Re initialize variables 
    this->depth2_raw = cv::Mat::zeros(depth2_raw.size(), depth2_raw.type());

    // If the distortion coefficients were given undistort the first image
    if(!this->dist1.empty())
    {   
        rectifyImage1(depth1_raw);
    }
    else
    {
        cv::cv2eigen(depth1_raw, this->depth_1_eigen);
    }

    // Get not 0 or nan values from the depth map
    int idx = 0;
    double z = 0.0;
    for (int v = 0; v < this->depth_1_eigen.rows(); ++v) {
        for (int u = 0; u < this->depth_1_eigen.cols(); ++u) {
            z = this->depth_1_eigen(v, u);
            
            if (z > 0.0 && !std::isnan(z)) {
                this->points1_eigen(0, idx) = u*z;
                this->points1_eigen(1, idx) = v*z;
                this->points1_eigen(2, idx) = z;
                ++idx;
            }
        }
    }
    
    // Reproject from first camera to second camera plane
    this->points2_eigen.noalias() = this->RepMat_eigen * this->points1_eigen;

    // Transsform to homogeneous (pixel) coordinates
    this->points2_eigen.topRows(2).array().rowwise() /= this->points2_eigen.row(2).array();
    
    //reconstruct the image as cv::Mat
    int u, v;
    for (int i = 0; i < idx; ++i) {
        u = static_cast<int>(std::round(this->points2_eigen(0, i)));
        v = static_cast<int>(std::round(this->points2_eigen(1, i)));
        z = this->points2_eigen(2, i);  // depth
        if (u >= 0 && u < this->depth2_raw.cols && v >= 0 && v < this->depth2_raw.rows && !std::isnan(z)) {
            this->depth2_raw.at<double>(v, u) = z;
        }
    }
    
    // If the distortion coefficients were given distort the second image
    if(!this->dist2.empty())
    {   
        distortImage2(this->depth2_raw);
    }

    return this->depth2_raw;
}