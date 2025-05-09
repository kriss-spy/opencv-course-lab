// task1
// 使用基于种子标记的分水岭算法（OpenCV自带watershed）对输入图像进行过分割
// 用户输入图像和整数K，要求程序自动计算K个随机种子点，确保各种子点之间的距离均 > (M*N/K)^0.5（参考泊松圆盘采样+贪心策略）
// 然后让程序在原图中标出各种子点的位置及编号，并采用半透明+随机着色的方式给出分水岭算法的可视化结果。

// solution
// adopt general framework of cv2-watershed.cpp
// add features:
// user friendly cli
// input img path (with size of MxN) and int k,
// compute random seeds of number k, while keeping min distance of seeds larger than (M*N/K)0.5)
// display image covered with seeds, print seed numbering
// run watershed

// how to compile and run task1
// cd opencv-course-lab/opencv-watershed/solutions
// make
// ./build/task1

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "watershed_utils.h" // Include the utility functions
#include "sample.h"
using namespace cv;
using namespace std;

// Global variables
Mat marker_mask, markers, img0, img, img_gray, wshed;
Point prev_pt(-1, -1);
NextStep task1_next_step;

const int k_min = 1;
const int k_max = 5000; // TODO choose proper values for k range
// theoretical max for 600x600 image, about 100000?

// TODO automatic stress test
int main(int argc, char **argv)
{

    print_welcome();
    print_task1_help();

    int k = 0;
    // default_temperature = 0.01;
    // double default_sigma = 1.02;
    vector<Point> seeds;

    task1_next_step = INPUT_IMAGE;
    string default_image = "image/fruits.jpg";
    img0 = get_image(default_image);

    // Print image size for logging
    // std::cout << "Loaded image: " << filepath << std::endl;
    std::cout << "Image size: " << img0.cols << "x" << img0.rows << " pixels" << std::endl;
    // std::cout << "Total area: " << img0.cols * img0.rows << " pixels" << std::endl;

    RNG rng(getTickCount());

    task1_next_step = INPUT_K;
    k = get_k(k_min, k_max);

    task1_next_step = INPUT_TEMP;
    double temperature = get_temperature(0, 1);

    // task1_next_step = INPUT_SIGMA;
    // double sigma = get_sigma(1, 2, default_sigma);

    // Create windows
    namedWindow("image", 1);
    namedWindow("watershed transform", 1);

    // Initialize images
    img = img0.clone();
    img_gray = img0.clone();
    wshed = img0.clone();
    marker_mask = Mat::zeros(img.size(), CV_32SC1);
    markers = Mat::zeros(img.size(), CV_32SC1);
    cvtColor(img, marker_mask, COLOR_BGR2GRAY);
    cvtColor(marker_mask, img_gray, COLOR_GRAY2BGR);

    imshow("image", img);
    imshow("watershed transform", wshed);
    // setMouseCallback("image", on_mouse, 0);
    task1_next_step = GENERATE_SEEDS;

    // Main loop
    for (;;)
    {
        int c = waitKey(0);

        if (c == 27 || c == 'q')
            break;
        if (c == 'c') // TODO implement c key
        {
        }
        if (c == 'h')
        {
            print_task1_help();
        }
        if (c == 'r') // Restore original image
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);
            imshow("image", img);
            wshed = img0.clone();
            imshow("watershed transform", wshed);
            task1_next_step = GENERATE_SEEDS;
        }

        if (c == 'g' && task1_next_step == GENERATE_SEEDS) // generate seeds
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);

            seeds = generate_seeds(img0, marker_mask, k, temperature);
            task1_next_step = WATERSHED;
        }

        if (c == 'v' && task1_next_step == WATERSHED)
        {
            visualize_points("image", img, seeds, 200);
        }

        if (c == 'w' && task1_next_step == WATERSHED)
        {
            // Clear markers before watershed
            markers = Mat::zeros(marker_mask.size(), CV_32SC1);

            // Find contours in marker_mask and use them as regions
            vector<vector<Point>> contours;
            findContours(marker_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            printf("Found %zu contours for watershed\n", contours.size());

            // Draw each contour with a unique label
            for (int i = 0; i < contours.size(); i++)
            {
                drawContours(markers, contours, i, Scalar(i + 1), -1);
            }

            // Debug log
            markersDebugLog(markers);

            vector<Vec3b> color_tab;
            for (int i = 0; i < contours.size(); i++)
            {
                int b = rng.uniform(0, 255);
                int g = rng.uniform(0, 255);
                int r = rng.uniform(0, 255);
                color_tab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
            }

            double t = (double)getTickCount();
            watershed(img0, markers); // Perform watershed
            t = (double)getTickCount() - t;
            printf("watershed exec time = %gms\n", t / getTickFrequency() * 1000.);

            // Color regions using watershed result
            for (int i = 0; i < markers.rows; i++)
                for (int j = 0; j < markers.cols; j++)
                {
                    int idx = markers.at<int>(i, j);
                    Vec3b &dst = wshed.at<Vec3b>(i, j);
                    if (idx == -1)
                        dst = Vec3b(255, 255, 255); // Boundary
                    else if (idx <= 0 || idx > contours.size())
                        dst = Vec3b(0, 0, 0); // Background
                    else
                        dst = color_tab[idx - 1]; // Segmented region
                }

            addWeighted(wshed, 0.5, img_gray, 0.5, 0, wshed);
            imshow("watershed transform", wshed);
        }
    }

    return 0;
}
