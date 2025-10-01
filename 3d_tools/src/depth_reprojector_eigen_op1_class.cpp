#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <optional>
#include "depth_reprojector_eigen_op1_class.hpp"


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
    this->K1_eigen_inverse = K1_eigen.inverse();
    

    num_points = this->dim1.height * this->dim1.width;
    this->temp_points.reserve(num_points);
    this->uvz_points_matrix.resize(3, num_points); 
    this->uv1_homogeneous.resize(3, num_points);
    this->uv1_homogeneous.row(2).setOnes();
    this->uv1_matrix.resize(2, num_points);
    this->z_vector.resize(num_points);
    this->points2_3d.resize(3, num_points);
    this->depth_1_eigen.resize(this->dim1.height, this->dim1.width);
    this->depth1_scaled.resize(this->dim1.height, this->dim1.width);
    this->depth2_raw = cv::Mat::zeros(dim2, CV_64FC1); 

}
void DepthReprojector::rectifyImage1(cv::Mat& image)
{
    cv::remap(image, this->rectified_image_1, this->map1, this->map2, cv::INTER_NEAREST);
    cv::cv2eigen(this->rectified_image_1, this->depth_1_eigen);
}

void DepthReprojector::distortImage2(cv::Mat& image)
{
    std::cout<<" distortImage2 TODO"<<std::endl
}

cv::Mat DepthReprojector::reprojectDepth(cv::Mat& depth1_raw, double scale_factor)
{
    //Re initialize variables 
    this->z_vector.setZero();
    this->uv1_homogeneous.topRows(2).setZero(); 
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


    // If scale_factor scale the depth to meters millimiters etc

    this->depth1_scaled = this->depth_1_eigen * scale_factor;

    // Get not 0 or nan values from the depth map
    int idx = 0;
    for (int v = 0; v < this->depth1_scaled.rows(); ++v) {
        for (int u = 0; u < this->depth1_scaled.cols(); ++u) {
            double z = this->depth1_scaled(v, u);
            if (z > 0.0 && !std::isnan(z)) {
                this->uv1_homogeneous(0, idx) = u;
                this->uv1_homogeneous(1, idx) = v;
                this->z_vector(idx) = z;
                ++idx;
            }
        }
    }

    

    // Step 2: Unproject to normalized camera coordinates
    Eigen::Matrix3Xd rays = this->K1_eigen_inverse * this->uv1_homogeneous;  // 3xN //11ms

    // Step 3: Scale by depth
    Eigen::Matrix3Xd points1_3d = rays.array().rowwise() * this->z_vector.transpose().array(); //11ms

    //Project in new space
    this->points2_3d.noalias() = this->R_eigen * points1_3d;
    this->points2_3d.colwise() += this->T_eigen; //both ops 15ms
    //Project in new camera plane
    Eigen::Matrix3Xd projected = this->K2_eigen * this->points2_3d;  // 3xN

    // Normalize by third row (homogeneous to pixel coordinates)
    Eigen::Matrix3Xd projected_uv(3, projected.cols());
    projected_uv.row(0) = projected.row(0).array() / projected.row(2).array();  // u'
    projected_uv.row(1) = projected.row(1).array() / projected.row(2).array();  // v'
    projected_uv.row(2) = projected.row(2);
    
    //reconstruct the image as cv::Mat
    int u, v;
    double z;
    for (int i = 0; i < projected_uv.cols(); ++i) {
        u = static_cast<int>(std::round(projected_uv(0, i)));
        v = static_cast<int>(std::round(projected_uv(1, i)));
        z = projected_uv(2, i);  // depth

        if (u >= 0 && u < this->depth2_raw.cols && v >= 0 && v < this->depth2_raw.rows && z > 0.0f && !std::isnan(z)) {
            this->depth2_raw.at<double>(v, u) = z;
        }
    }
    
    // If the distortion coefficients were given distort the second image
    if(!this->dist1.empty())
    {   
        distortImage2(this->depth2_raw);
    }

    return this->depth2_raw;
}