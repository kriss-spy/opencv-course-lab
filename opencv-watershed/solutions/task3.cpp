// parent: task1.cpp

// task3
// 根据分水岭结果中各区域面积大小的“堆排序”结果，提示最大和最小面积，
// get the area of regions, heap sort, print max and min area in cli
// 用户输入查找范围（面积下界和上界），使用折半查找，程序对所有符合要求的分水岭结果（标记区域面积）进行突出显示
// input search range [lower_bound, upper_bound], use binary search, find all regions within range, color them and mark area in GUI
// 并以这些高亮区域的面积大小作为权值，进行哈夫曼编码（考虑深度+递归策略），绘制该哈夫曼树
// use these area values, huffman encode, draw huffman tree in a new window

// solution

// how to compile and run task3
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

using namespace cv;
using namespace std;

// Global variables
Mat marker_mask, markers, img0, img, img_gray, wshed;
Point prev_pt(-1, -1);
NextStep task3_next_step;

// Global variables for k, temperature, and sigma
int k;
double temperature;
double sigma;

const int k_min = 1;
const int k_max = 5000; // TODO choose proper values for k range
// theoretical max for 600x600 image, about 100000?

// New helper functions for task3
void initApp_task3(vector<Point> &seeds)
{
    double default_temperature = 0.01;
    double default_sigma = 1.02;

    task3_next_step = INPUT_IMAGE;
    string default_image = "../image/fruits.jpg";
    img0 = get_image(default_image);

    std::cout << "Image size: " << img0.cols << "x" << img0.rows << " pixels" << std::endl;

    task3_next_step = INPUT_K;
    k = get_k(k_min, k_max);

    task3_next_step = INPUT_TEMP;
    temperature = get_temperature(0, 1, default_temperature);

    task3_next_step = INPUT_SIGMA;
    sigma = get_sigma(1, 2, default_sigma);

    print_task1_help();

    // Initialize images
    img = img0.clone();
    img_gray = img0.clone();
    wshed = img0.clone();
    marker_mask = Mat::zeros(img.size(), CV_32SC1);
    markers = Mat::zeros(img.size(), CV_32SC1);
    cvtColor(img, marker_mask, COLOR_BGR2GRAY);
    cvtColor(marker_mask, img_gray, COLOR_GRAY2BGR);
}

void setupWindows_task3()
{
    // Create windows
    namedWindow("image", 1);
    namedWindow("watershed transform", 1);

    imshow("image", img);
    imshow("watershed transform", wshed);
    task3_next_step = GENERATE_SEEDS;
}

void runEventLoop_task3(vector<Point> &seeds)
{
    RNG rng(getTickCount());
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
            print_task3_help();
        }
        if (c == 'r') // Restore original image
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);
            imshow("image", img);
            wshed = img0.clone();
            imshow("watershed transform", wshed);
        }
        if (c == 'v' && task3_next_step == WATERSHED)
        {
            visualize_points("image", img, seeds, 200);
        }
        if (c == 'g') // generate seeds
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);

            seeds = generate_seeds(img0, marker_mask, k, temperature, sigma);
            // seeds = cyj_generateSeeds(k, img0.rows, img0.cols);
            task3_next_step = WATERSHED;
        }
        if (c == 'w' && task3_next_step == WATERSHED)
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
        if (c == 's' && task3_next_step == HEAP_SORT_AREA)
        {
        }
        if (c == 'h' && task3_next_step == HUFFMAN_TREE)
        {
        }
    }
}

int main(int argc, char **argv)
{
    vector<Point> seeds;

    initApp_task3(seeds);
    setupWindows_task3();
    runEventLoop_task3(seeds);

    return 0;
}
