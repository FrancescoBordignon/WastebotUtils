#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "depth_reprojector_eigen_op1_class.hpp"

namespace py = pybind11;

// Convert numpy array to cv::Mat (assumes double)
cv::Mat numpy_to_mat(const py::array& arr) {
    py::buffer_info info = arr.request();
    int rows = info.shape[0];
    int cols = info.shape[1];
    int type = CV_64F;  // assumes double
    return cv::Mat(rows, cols, type, const_cast<void*>(info.ptr)).clone();  // clone to own memory
}

PYBIND11_MODULE(depth_reprojector, m) {
    py::class_<cv::Size>(m, "Size")
        .def(py::init<int, int>())
        .def_readwrite("width", &cv::Size::width)
        .def_readwrite("height", &cv::Size::height);

    py::class_<DepthReprojector>(m, "DepthReprojector")
        .def(py::init([](py::array K1,
                         py::array K2,
                         cv::Size dim1,
                         cv::Size dim2,
                         py::array dist1 = py::array(),
                         py::array dist2 = py::array(),
                         py::array R = py::array(),
                         py::array T = py::array()) {

            cv::Mat K1_bind = numpy_to_mat(K1);
            cv::Mat K2_bind = numpy_to_mat(K2);

            // Handle optional dist1
            cv::Mat dist1_bind;
            if (dist1.size() == 0) {
                dist1_bind = cv::Mat();
            } else {
                cv::Mat temp = numpy_to_mat(dist1);
                dist1_bind = (cv::countNonZero(temp) == 0) ? cv::Mat() : temp;
            }

            // Handle optional dist2
            cv::Mat dist2_bind;
            if (dist2.size() == 0) {
                dist2_bind = cv::Mat();
            } else {
                cv::Mat temp = numpy_to_mat(dist2);
                dist2_bind = (cv::countNonZero(temp) == 0) ? cv::Mat() : temp;
            }

            // Default R: 3x3 identity
            cv::Mat R_bind;
            if (R.size() == 0) {
                R_bind = cv::Mat::eye(3, 3, CV_64F);
            } else {
                R_bind = numpy_to_mat(R);
            }

            // Default T: 3x1 zero vector
            cv::Mat T_bind;
            if (T.size() == 0) {
                T_bind = cv::Mat::zeros(3, 1, CV_64F);
            } else {
                T_bind = numpy_to_mat(T);
            }

            return new DepthReprojector(K1_bind, K2_bind, dim1, dim2, dist1_bind, dist2_bind, R_bind, T_bind);
        }),
        py::arg("K1"),
        py::arg("K2"),
        py::arg("dim1"),
        py::arg("dim2"),
        py::arg("dist1") = py::array(),
        py::arg("dist2") = py::array(),
        py::arg("R") = py::array(),  // Optional: defaults to identity
        py::arg("T") = py::array())  // Optional: defaults to zero vector

        .def("reprojectDepth", [](DepthReprojector& self, py::array depth, double scale_factor) {
            cv::Mat depth_bind = numpy_to_mat(depth);
            cv::Mat result = self.reprojectDepth(depth_bind, scale_factor);

            return py::array_t<double>(
                { result.rows, result.cols },
                { result.step[0], result.step[1] },
                result.ptr<double>()
            );
        });

    m.def("print_ok", []() { std::cout << "binding works\n"; });
}
