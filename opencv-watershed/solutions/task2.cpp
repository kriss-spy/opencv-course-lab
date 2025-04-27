// 使用邻接表统计分水岭结果中各区域的邻接关系
// 并采用四原色法（合理选择初始着色区域，并基于图的广度优先遍历，采用队列对其他待着色区域进行着色顺利梳理，加速全图着色过程）
// 对分水岭结果重新着色（使用堆栈+回溯策略，优化回退率）

// solution
// choose the four colors, use enum 0~3, red, yellow, green, blue
// use original seed points/index to represent regions
// get adjacency list (matrix?)
// choose initial coloring region
// based on graph BFS, use queue
// search for the coloring solution
// color and display the final result in watershed transfrom window

// how to compile and run task2
// cd opencv-course-lab/opencv-watershed/solutions
// make
// cd build
// ./task2

// below is task1 code
// task2 not fully implemented
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "watershed_utils.h" // Include the utility functions

using namespace cv;
using namespace std;

// Global variables
Mat marker_mask, markers, img0, img, img_gray, wshed;
Point prev_pt(-1, -1);
NextStep task2_next_step;
FourColor four_colors;

const int k_min = 1;
const int k_max = 5000; // TODO choose proper values for k range
// theoretical max for 600x600 image, about 100000?

// TODO automatic stress test
int main(int argc, char **argv)
{
    int k = 0;
    double default_temperature = 0.01;
    double default_sigma = 1.02;
    vector<Point> seeds;

    task2_next_step = INPUT_IMAGE;
    string default_image = "../image/fruits.jpg";
    get_image(default_image);

    // Print image size for logging
    // std::cout << "Loaded image: " << filepath << std::endl;
    std::cout << "Image size: " << img0.cols << "x" << img0.rows << " pixels" << std::endl;
    // std::cout << "Total area: " << img0.cols * img0.rows << " pixels" << std::endl;

    RNG rng(getTickCount());

    task2_next_step = INPUT_K;
    k = get_k(k_min, k_max);

    task2_next_step = INPUT_TEMP;
    double temperature = get_temperature(0, 1, default_temperature);

    task2_next_step = INPUT_SIGMA;
    double sigma = get_sigma(1, 2, default_sigma);

    print_task1_help();

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
    task2_next_step = GENERATE_SEEDS;

    // Main loop
    for (;;)
    {
        int c = waitKey(0);

        if (c == 27 || c == 'q')
            break;

        if (c == 'h')
        {
            print_task2_help();
        }
        if (c == 'r') // Restore original image
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);
            imshow("image", img);
            wshed = img0.clone();
            imshow("watershed transform", wshed);
        }
        if (c == 'v' && task2_next_step == WATERSHED)
        {
            visualize_seeds(img, seeds);
        }
        if (c == 'g') // generate seeds
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);

            seeds = generate_seeds(k, temperature, sigma);
            // seeds = cyj_generateSeeds(k, img0.rows, img0.cols);
            task2_next_step = WATERSHED;
        }
        if (c == 'w' && task2_next_step == WATERSHED)
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

            task2_next_step = FOURCOLOR;
        }

        if (c == 'c' && task2_next_step == FOURCOLOR) // TODO implement four color
        {
            // get adjacency list
            // use BFS, queue
            // use backtracking
            // use stack
            // search for coloring solution
            // display

            // how to get adjacency list?
            // no way to do this with seeds points only
            // better make use of watershed results
            // adjacency is about regions, afterall
        }
    }

    return 0;
}
