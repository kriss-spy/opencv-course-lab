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

using namespace cv;
using namespace std;

// Global variables
Mat marker_mask, markers, img0, img, img_gray, wshed;
Point prev_pt(-1, -1);

enum next_step
{
    INPUT_IMAGE,
    INPUT_K,
    INPUT_TEMP,
    GENERATE_SEEDS,
    WATERSHED,
    EXIT
} task1_state;

// Debug logging function
void markersDebugLog(const Mat &markers)
{
#ifdef DEBUG
    // Print the number of contours found
    vector<vector<Point>> contours;
    findContours(marker_mask, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    cout << "Number of contours found: " << contours.size() << endl;

    // Find min/max values in markers before watershed
    double minVal, maxVal;
    minMaxLoc(markers, &minVal, &maxVal);
    cout << "Markers minVal: " << minVal << ", maxVal: " << maxVal << endl;
#endif
}

// Add this helper function to verify minimum distances
bool verifyMinimumDistance(const vector<Point> &seeds, double minDist)
{
    for (int i = 0; i < seeds.size(); i++)
    {
        for (int j = i + 1; j < seeds.size(); j++)
        {
            double dist = norm(seeds[i] - seeds[j]);
            if (dist < minDist)
            {
                printf("Distance violation between seeds %d and %d: %.2f < %.2f\n",
                       i + 1, j + 1, dist, minDist);
                return false;
            }
        }
    }
    return true;
}

void sampleDebugLog(const Mat &marker_mask, vector<Point> seeds, double minDist)
{
#ifdef DEBUG
    // Print the first 10 seeds for debugging
    printf("First 10 seed points:\n");
    for (int i = 0; i < min(10, (int)seeds.size()); i++)
    {
        printf("Seed %d: (%d, %d)\n", i + 1, seeds[i].x, seeds[i].y);

        // // Optionally, draw the seed number on the image for visualization
        // Point textPos(seeds[i].x + 5, seeds[i].y);
        // putText(marker_mask, to_string(i + 1), textPos, FONT_HERSHEY_SIMPLEX,
        //         0.4, Scalar(255), 1, LINE_AA);
    }

    bool distanceConstraintMet = verifyMinimumDistance(seeds, minDist);
    if (distanceConstraintMet)
    {
        printf("All seeds satisfy the minimum distance constraint\n");
    }
    else
    {
        printf("Warning: Some seeds do not satisfy the minimum distance constraint\n");
    }

#endif
}

void print_help()
{
    // TODO print introduction

    // Print instructions
    printf("Hot keys: \n"
           "\tESC or q - quit the program\n"
           "\tr - restore the original image\n"
           "\tg - generate seeds\n"
           "\tv - visualize generated seeds\n"
           "\tw - run watershed\n");
}

void get_image(string default_path)
{
    string filepath;
    cout << "please input image path" << endl;
    cout << "press enter to use default image fruits.jpg" << endl;
    while (true)
    {
        cout << "> ";
        // cin >> k;
        string input;
        getline(cin, input);
        if (!input.empty())
        {
            filepath = input;
            img0 = imread(filepath, 1);
            if (img0.empty())
            {
                cout << "please input a valid path!" << endl;
                continue;
            }
            else
            {
                return;
            }
        }
        else
        {
            img0 = imread(default_path, 1);
            return;
        }
    }
}

int get_k(int k_min, int k_max)
{
    int k = 0;
    cout << "please input k, desired number of random seed points" << endl;
    cout << "for example, 100, 500, 1000" << endl;
    while (true)
    {
        cout << "> ";
        // cin >> k;
        string input;
        getline(cin, input);
        if (!input.empty())
        {
            try
            {
                k = stoi(input);
                if (k < k_min || k > k_max)
                {
                    cout << "k out of range!" << endl;
                }
                else
                {
                    return k;
                }
            }
            catch (const std::invalid_argument &e)
            {
                cout << "Invalid input! Please enter a valid number." << endl;
            }
            catch (const std::out_of_range &e)
            {
                cout << "Input is too large!" << endl;
            }
        }
        else
        {
            cout << "please input k!" << endl;
            continue;
        }
    }
}

float get_temperature(float t_min, float t_max, float t_default)
{
    float t;
    cout << "please input temperature between 0 and 1, press enter to use default " << t_default << endl;
    while (true)
    {
        cout << "> ";
        string input;
        getline(cin, input);
        if (!input.empty())
        {
            try
            {
                t = stof(input);
                if (t < t_min || t > t_max)
                {
                    cout << "temperature out of range!" << endl;
                }
                else
                {
                    return t;
                }
            }
            catch (const std::invalid_argument &e)
            {
                cout << "Invalid input! Please enter a valid number." << endl;
            }
            catch (const std::out_of_range &e)
            {
                cout << "Input is too large!" << endl;
            }
        }
        else
        {
            return t_default;
        }
    }
}

// Jittered grid vs Poisson disc
// https://www.redblobgames.com/x/1830-jittered-grid/

void jittered_hex_grid_sample(int k, float temperature) // TODO better algorithm hex grid sample
{
}

vector<Point> jittered_grid_sample(int k, float temperature)
{
    // not fully using space at the end
    // BUG not meeting requirement for min distance

    int M = marker_mask.rows;                    // Image height
    int N = marker_mask.cols;                    // Image width
    double min_dist = sqrt((double)(M * N) / k); // Minimum distance required

    // Calculate grid dimensions to ensure we get approximately k cells
    // Adjust for image aspect ratio
    int grid_cols = ceil(sqrt((double)k * N / M));
    int grid_rows = ceil((double)k / grid_cols);

    // Ensure we have at least k cells
    while (grid_rows * grid_cols < k)
    {
        grid_cols++;
    }

    // Calculate cell dimensions
    double cell_width = (double)N / grid_cols;
    double cell_height = (double)M / grid_rows;

    // Random number generator
    RNG rng(getTickCount());

    // Vector to store generated seed points
    vector<Point> seeds;

    // For each cell, generate a random point with jitter
    for (int i = 0; i < grid_rows && seeds.size() < k; i++)
    {
        for (int j = 0; j < grid_cols && seeds.size() < k; j++)
        {
            // Calculate cell boundaries
            double cell_x_min = j * cell_width;
            double cell_y_min = i * cell_height;

            // Calculate cell center
            double center_x = cell_x_min + cell_width / 2;
            double center_y = cell_y_min + cell_height / 2;

            // Calculate maximum jitter that ensures minimum distance
            double jitter_amount_x = min(cell_width / 2, min_dist / 2);
            double jitter_amount_y = min(cell_height / 2, min_dist / 2);

            // Apply jitter from cell center
            int x = center_x + temperature * rng.uniform(-jitter_amount_x, jitter_amount_x);
            int y = center_y + temperature * rng.uniform(-jitter_amount_y, jitter_amount_y);

            // Ensure point is within image boundaries
            x = max(0, min(N - 1, x));
            y = max(0, min(M - 1, y));

            // Add point to seeds
            seeds.push_back(Point(x, y));
        }
    }
    if (seeds.size() > 0)
    {
        // Find min and max coordinates
        int min_x = N, min_y = M, max_x = 0, max_y = 0;
        for (const Point &p : seeds)
        {
            min_x = min(min_x, p.x);
            min_y = min(min_y, p.y);
            max_x = max(max_x, p.x);
            max_y = max(max_y, p.y);
        }

        // Calculate scale factors to expand points to full image
        double scale_x = (double)(N - 1) / max(1, max_x - min_x);
        double scale_y = (double)(M - 1) / max(1, max_y - min_y);

        // Use the smaller scale to maintain aspect ratio
        double scale = min(scale_x, scale_y);

        // Only apply scaling if it actually increases the spread (scale > 1.0)
        if (scale > 1.05)
        { // Add a small threshold to avoid unnecessary scaling
            // Calculate centroid for scaling from center
            double center_x = (min_x + max_x) / 2.0;
            double center_y = (min_y + max_y) / 2.0;

            // Apply scaling to all points
            for (Point &p : seeds)
            {
                // Scale from center
                double dx = p.x - center_x;
                double dy = p.y - center_y;

                p.x = round(center_x + dx * scale);
                p.y = round(center_y + dy * scale);

                // Ensure point remains within image boundaries
                p.x = max(0, min(N - 1, p.x));
                p.y = max(0, min(M - 1, p.y));
            }

            printf("Applied scaling factor of %.2f to better distribute points\n", scale);
        }
    }

    // Print information about generated points
    printf("Generated %zu seed points using jittered grid sampling\n", seeds.size());
    printf("Minimum distance threshold: %.2f pixels\n", min_dist);
    printf("Grid dimensions: %d rows x %d columns\n", grid_rows, grid_cols);

    sampleDebugLog(markers, seeds, min_dist);

    return seeds;
}

void visualize_seeds(vector<Point> seeds)
{
    // Draw circles and numbers on the original image
    for (int i = 0; i < seeds.size(); i++)
    {
        // Draw a visible circle at each seed point
        // Use a contrasting color that stands out on most images
        circle(img, seeds[i], 3, Scalar(0, 255, 255), FILLED);
        circle(img, seeds[i], 3, Scalar(0, 0, 0), 1); // Black outline for contrast

        // Place the seed number next to the point
        Point textPos(seeds[i].x + 5, seeds[i].y + 5);

        putText(img, to_string(i + 1), textPos, FONT_HERSHEY_SIMPLEX,
                0.4, Scalar(0, 0, 0), 2, LINE_AA); // Outlined text (thicker)
        putText(img, to_string(i + 1), textPos, FONT_HERSHEY_SIMPLEX,
                0.4, Scalar(255, 255, 255), 1, LINE_AA); // White text
    }

    // Update the displayed image
    imshow("image", img);

    // Print summary information
    printf("Visualized %zu seed points on the image\n", seeds.size());
}

vector<Point> generate_seeds(int k, float temperature)
{
    double t = (double)getTickCount();

    // Use 8-bit mask for visualization and 32-bit for watershed
    marker_mask = Mat::zeros(img.size(), CV_8UC1);

    // generate k random seed points in marker_mask
    vector<Point> seeds = jittered_grid_sample(k, temperature);

    // Draw smaller circles for markers to avoid overlapping
    // But use distinct values for each region
    for (int i = 0; i < seeds.size(); i++)
    {
        // Use modulo to handle values > 255
        circle(marker_mask, seeds[i], 5, Scalar((i % 254) + 1), FILLED);
    }

    t = (double)getTickCount() - t;
    printf("seeds generation time cost = %gms\n", t / getTickFrequency() * 1000.);

    return seeds;
}

const int k_min = 1;
const int k_max = 5000; // TODO choose proper values for k range
// theoretical max for 600x600 image, about 100000?

int main(int argc, char **argv)
{
    int k = 0;
    float default_temperature = 0.01;
    vector<Point> seeds;
    task1_state = INPUT_IMAGE;
    string default_image = "../image/fruits.jpg";
    get_image(default_image);

    // Print image size for logging
    // std::cout << "Loaded image: " << filepath << std::endl;
    std::cout << "Image size: " << img0.cols << "x" << img0.rows << " pixels" << std::endl;
    // std::cout << "Total area: " << img0.cols * img0.rows << " pixels" << std::endl;

    RNG rng(12345);

    task1_state = INPUT_K;
    k = get_k(k_min, k_max);
    float temperature = get_temperature(0, 1, default_temperature);

    print_help();

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
    task1_state = GENERATE_SEEDS;

    // Main loop
    for (;;)
    {
        int c = waitKey(0);

        if (c == 27 || c == 'q')
            break;
        if (c == 'h')
        {
            print_help();
        }
        if (c == 'r') // Restore original image
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);
            imshow("image", img);
            wshed = img0.clone();
            imshow("watershed transform", wshed);
        }
        if (c == 'v' && task1_state == WATERSHED)
        {
            visualize_seeds(seeds);
        }
        if (c == 'g') // generate seeds
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);

            seeds = generate_seeds(k, temperature);
            task1_state = WATERSHED;
        }
        if (c == 'w' && task1_state == WATERSHED)
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
