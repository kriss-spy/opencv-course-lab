// verify opencv installation
// use this to compile
// g++ -o verify-opencv verify-opencv.cpp `pkg-config --cflags --libs opencv4` && ./verify-opencv && rm ./verify-opencv
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(0, 0, 255));
    if (image.empty())
    {
        std::cerr << "Failed to create image. OpenCV is not installed properly." << std::endl;
        return -1;
    }
    std::cout << "OpenCV " << CV_VERSION << " is installed properly." << std::endl;
    return 0;
}