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
// cd build
// ./task1

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
NextStep task1_next_step;

// Add this helper function to verify minimum distances
bool try_adjust_seeds(Point &seed, vector<Point> &violation_seeds, double min_dist)
{
    // TODO verity funcionality
    const int MAX_ATTEMPTS = 30;
    const int NUM_DIRECTIONS = 16;
    const double MAX_RADIUS_MULTIPLIER = 2.0;

    int img_height = marker_mask.rows;
    int img_width = marker_mask.cols;

    Point original_seed = seed;
    double current_min_dist = DBL_MAX;

    // Compute the current minimum distance to violating neighbors
    for (const Point &violator : violation_seeds)
    {
        double dist = norm(seed - violator);
        current_min_dist = min(current_min_dist, dist);
    }

    //--- 1) Force-directed approach
    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
    {
        Point2f force_vector(0, 0);
        bool all_satisfied = true;

        // Calculate net repulsion force from violating neighbors
        for (const Point &violator : violation_seeds)
        {
            double dist = norm(seed - violator);
            if (dist < min_dist)
            {
                all_satisfied = false;
                Point2f direction(seed.x - violator.x, seed.y - violator.y);
                double mag = norm(direction);
                if (mag > 0)
                {
                    direction.x /= mag;
                    direction.y /= mag;
                    double force_strength = (min_dist - dist) / min_dist;
                    force_vector.x += direction.x * force_strength;
                    force_vector.y += direction.y * force_strength;
                }
            }
        }

        if (all_satisfied)
        {
            // Already satisfied all constraints, done
            cout << "Seed adjusted successfully after " << attempt << " attempts" << endl;
            return true;
        }

        if (norm(force_vector) < 1e-6)
            break; // No meaningful force to move the seed

        // Normalize force
        double fmag = norm(force_vector);
        force_vector.x /= fmag;
        force_vector.y /= fmag;

        // Step size increases each attempt
        double step_size = min_dist * 0.2 * (1 + attempt * 0.1);

        // Try the new position
        Point new_seed(
            cvRound(seed.x + force_vector.x * step_size),
            cvRound(seed.y + force_vector.y * step_size));

        // Keep within image
        new_seed.x = max(0, min(img_width - 1, new_seed.x));
        new_seed.y = max(0, min(img_height - 1, new_seed.y));

        // Check if new position improves the minimum distance
        double new_min_dist = DBL_MAX;
        for (const Point &violator : violation_seeds)
        {
            double dist = norm(new_seed - violator);
            new_min_dist = min(new_min_dist, dist);
        }

        // If it improves (or at least doesn't worsen), update seed
        if (new_min_dist > current_min_dist)
        {
            seed = new_seed;
            current_min_dist = new_min_dist;
        }
        else
        {
            // Not an improvement, stop force attempts
            break;
        }
    }

    //--- 2) Systematic search in multiple directions if force-based failed
    Point best_seed = seed;
    double best_min_violation_dist = current_min_dist;

    for (int i = 0; i < NUM_DIRECTIONS; i++)
    {
        double angle = 2.0 * CV_PI * i / NUM_DIRECTIONS;
        for (double radius_mult = 0.5; radius_mult <= MAX_RADIUS_MULTIPLIER; radius_mult += 0.25)
        {
            double radius = min_dist * radius_mult;
            Point test_seed(
                cvRound(original_seed.x + cos(angle) * radius),
                cvRound(original_seed.y + sin(angle) * radius));

            // Stay in bounds
            test_seed.x = max(0, min(img_width - 1, test_seed.x));
            test_seed.y = max(0, min(img_height - 1, test_seed.y));

            // Check minimum distance to violating neighbors
            double local_min_dist = DBL_MAX;
            bool satisfies_all = true;
            for (const Point &violator : violation_seeds)
            {
                double dist = norm(test_seed - violator);
                local_min_dist = min(local_min_dist, dist);
                if (dist < min_dist)
                    satisfies_all = false;
            }

            // If fully satisfied, update and return
            if (satisfies_all)
            {
                seed = test_seed;
                return true;
            }

            // Otherwise track best improvement
            if (local_min_dist > best_min_violation_dist)
            {
                best_min_violation_dist = local_min_dist;
                best_seed = test_seed;
            }
        }
    }

    // If we found a slightly better position, use that
    if (best_min_violation_dist > current_min_dist)
    {
        seed = best_seed;
        cout << "Could not fully satisfy constraints, but improved min distance from "
             << current_min_dist << " to " << best_min_violation_dist
             << " (required: " << min_dist << ")" << endl;
        return true;
    }

    // Failed to improve enough
    cout << "Failed to adjust seed at (" << original_seed.x << ", " << original_seed.y
         << ") with min_dist=" << min_dist
         << ". Closest violation distance remained: " << current_min_dist << endl;
    return false;
}
// Jittered grid vs Poisson disc
// https://www.redblobgames.com/x/1830-jittered-grid/

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

    task1_next_step = INPUT_IMAGE;
    string default_image = "../image/fruits.jpg";
    img0 = get_image(default_image);

    // Print image size for logging
    // std::cout << "Loaded image: " << filepath << std::endl;
    std::cout << "Image size: " << img0.cols << "x" << img0.rows << " pixels" << std::endl;
    // std::cout << "Total area: " << img0.cols * img0.rows << " pixels" << std::endl;

    RNG rng(getTickCount());

    task1_next_step = INPUT_K;
    k = get_k(k_min, k_max);

    task1_next_step = INPUT_TEMP;
    double temperature = get_temperature(0, 1, default_temperature);

    task1_next_step = INPUT_SIGMA;
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
        }
        if (c == 'v' && task1_next_step == WATERSHED)
        {
            visualize_seeds(img, seeds);
        }
        if (c == 'g') // generate seeds
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);

            seeds = generate_seeds(k, temperature, sigma);
            // seeds = cyj_generateSeeds(k, img0.rows, img0.cols);
            task1_next_step = WATERSHED;
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
