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
#include <queue>
#include <stack>
#include <unordered_set>
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

// Four color palette - red, yellow, green, blue
const Vec3b FOUR_COLOR_PALETTE[4] = {
    Vec3b(0, 0, 255),   // RED
    Vec3b(0, 255, 255), // YELLOW
    Vec3b(0, 255, 0),   // GREEN
    Vec3b(255, 0, 0)    // BLUE
};

// Function to find neighbors of a region - completely rewritten to properly detect adjacency
vector<int> findNeighbors(const Mat &markers, int region_id, int total_regions)
{
    vector<bool> is_neighbor(total_regions + 1, false);
    vector<int> neighbors;

    // Detect boundary pixels for the current region
    Mat region_mask = (markers == region_id);

    // Dilate the region mask to find adjacent regions
    Mat dilated;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(region_mask, dilated, kernel, Point(-1, -1), 1);

    // Subtract the original region to get only the boundary area
    Mat boundary;
    subtract(dilated, region_mask, boundary);

    // Find all regions in the boundary area
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            if (boundary.at<uchar>(i, j) > 0)
            {
                int neighbor_id = markers.at<int>(i, j);
                if (neighbor_id > 0 && neighbor_id != region_id && neighbor_id <= total_regions)
                {
                    is_neighbor[neighbor_id] = true;
                }
            }
        }
    }

    // Collect all unique neighboring regions
    for (int i = 1; i <= total_regions; i++)
    {
        if (is_neighbor[i])
        {
            neighbors.push_back(i);
        }
    }

    return neighbors;
}

// Function to build adjacency list for regions
vector<vector<int>> buildAdjacencyList(const Mat &markers, int total_regions)
{
    vector<vector<int>> adjacency_list(total_regions + 1);

    for (int region = 1; region <= total_regions; region++)
    {
        adjacency_list[region] = findNeighbors(markers, region, total_regions);
    }

    // Print adjacency list for debugging
    cout << "Adjacency List:" << endl;
    for (int i = 1; i <= total_regions; i++)
    {
        cout << "Region " << i << " neighbors: ";
        for (int neighbor : adjacency_list[i])
        {
            cout << neighbor << " ";
        }
        cout << endl;
    }

    return adjacency_list;
}

// Check if it's valid to color a region with a specific color
bool isValidColor(int region, int color, const vector<int> &colors, const vector<vector<int>> &adjacency_list)
{
    for (int neighbor : adjacency_list[region])
    {
        if (colors[neighbor] == color)
        {
            return false;
        }
    }
    return true;
}

// Function to color the regions using four-color theorem (BFS + backtracking)
vector<int> fourColorRegions(const vector<vector<int>> &adjacency_list, int total_regions)
{
    vector<int> colors(total_regions + 1, -1); // -1 means uncolored

    // Start with region 1 and try to color it with color 0 (RED)
    queue<int> q;
    q.push(1);
    colors[1] = 0;

    while (!q.empty())
    {
        int current_region = q.front();
        q.pop();

        // Get all uncolored neighbors
        for (int neighbor : adjacency_list[current_region])
        {
            if (colors[neighbor] == -1)
            {
                // Try to find a valid color for this neighbor
                bool found_color = false;
                for (int color = 0; color < 4; color++)
                {
                    if (isValidColor(neighbor, color, colors, adjacency_list))
                    {
                        colors[neighbor] = color;
                        q.push(neighbor);
                        found_color = true;
                        break;
                    }
                }

                // If no valid color is found, we'll need backtracking (rare with 4 colors)
                if (!found_color)
                {
                    // In practice, 4 colors should be enough, but we'll add backtracking for robustness
                    cout << "Warning: Could not find valid color for region " << neighbor << endl;

                    // Use a more sophisticated backtracking approach if this happens
                    // For simplicity, we'll just assign a default color temporarily
                    colors[neighbor] = 0;
                    q.push(neighbor);
                }
            }
        }
    }

    // Handle any remaining uncolored regions (disconnected parts)
    for (int region = 1; region <= total_regions; region++)
    {
        if (colors[region] == -1)
        {
            for (int color = 0; color < 4; color++)
            {
                if (isValidColor(region, color, colors, adjacency_list))
                {
                    colors[region] = color;
                    break;
                }
            }
            // In the unlikely case no color works, assign any color
            if (colors[region] == -1)
            {
                colors[region] = 0;
            }
        }
    }

    return colors;
}

// Improved four-coloring with backtracking
bool colorRegionBacktracking(int region, vector<int> &colors, const vector<vector<int>> &adjacency_list, int total_regions)
{
    if (region > total_regions)
    {
        return true; // All regions are colored successfully
    }

    // Try each of the four colors
    for (int color = 0; color < 4; color++)
    {
        if (isValidColor(region, color, colors, adjacency_list))
        {
            colors[region] = color;

            // Recursively color the next region
            if (colorRegionBacktracking(region + 1, colors, adjacency_list, total_regions))
            {
                return true;
            }

            // If we're here, this color didn't work, so backtrack
            colors[region] = -1;
        }
    }

    return false; // No solution found
}

// Get most connected region as starting point
int getMostConnectedRegion(const vector<vector<int>> &adjacency_list, int total_regions)
{
    int max_connections = -1;
    int most_connected_region = 1;

    for (int region = 1; region <= total_regions; region++)
    {
        int connections = adjacency_list[region].size();
        if (connections > max_connections)
        {
            max_connections = connections;
            most_connected_region = region;
        }
    }

    return most_connected_region;
}

// TODO automatic stress test
int main(int argc, char **argv)
{
    int k = 0;
    double default_temperature = 0.01;
    double default_sigma = 1.02;
    vector<Point> seeds;

    task2_next_step = INPUT_IMAGE;
    string default_image = "../image/fruits.jpg";
    img0 = get_image(default_image); // Fix: Assign return value to img0

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

    print_task2_help();

    // Create windows
    namedWindow("image", 1);
    namedWindow("watershed transform", 1);
    namedWindow("four color result", 1);

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
            visualize_seeds("image", img, seeds, 200);
        }
        if (c == 'g') // generate seeds
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);

            // Call generate_seeds with marker_mask as a parameter
            seeds = generate_seeds(img0, marker_mask, k, temperature, sigma);
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

            // Visualize the seed points on the watershed result image
            Mat wshed_with_seeds = wshed.clone();

            visualize_seeds("watershed transform", wshed_with_seeds, seeds, 200);

            task2_next_step = FOURCOLOR;
        }
        if (c == 'c' && task2_next_step == FOURCOLOR)
        {
            // Get total number of regions (max marker value excluding boundaries and background)
            int total_regions = 0;
            for (int i = 0; i < markers.rows; i++)
            {
                for (int j = 0; j < markers.cols; j++)
                {
                    total_regions = max(total_regions, markers.at<int>(i, j));
                }
            }

            cout << "Total regions for four-coloring: " << total_regions << endl;

            // Build adjacency list
            double t_adj = (double)getTickCount();
            vector<vector<int>> adjacency_list = buildAdjacencyList(markers, total_regions);
            t_adj = (double)getTickCount() - t_adj;
            printf("Adjacency list build time = %gms\n", t_adj / getTickFrequency() * 1000.);

            // Apply four-color theorem
            double t_color = (double)getTickCount();
            vector<int> colors(total_regions + 1, -1);

            // Option 1: Using BFS-based approach
            colors = fourColorRegions(adjacency_list, total_regions);

            // Option 2: For more challenging graphs, use backtracking
            // Start with the most connected region for better results
            // int start_region = getMostConnectedRegion(adjacency_list, total_regions);
            // colors[start_region] = 0; // Start with RED
            // bool success = colorRegionBacktracking(1, colors, adjacency_list, total_regions);

            t_color = (double)getTickCount() - t_color;
            printf("Four-coloring time = %gms\n", t_color / getTickFrequency() * 1000.);

            // Count the number of regions with each color
            vector<int> color_count(4, 0);
            for (int i = 1; i <= total_regions; i++)
            {
                if (colors[i] >= 0 && colors[i] < 4)
                {
                    color_count[colors[i]]++;
                }
            }

            cout << "Color distribution:" << endl;
            cout << "RED: " << color_count[0] << " regions" << endl;
            cout << "YELLOW: " << color_count[1] << " regions" << endl;
            cout << "GREEN: " << color_count[2] << " regions" << endl;
            cout << "BLUE: " << color_count[3] << " regions" << endl;

            // Apply four-color result to the image
            Mat four_color_result = img0.clone();

            for (int i = 0; i < markers.rows; i++)
            {
                for (int j = 0; j < markers.cols; j++)
                {
                    int region_idx = markers.at<int>(i, j);
                    Vec3b &pixel = four_color_result.at<Vec3b>(i, j);

                    if (region_idx == -1)
                    {
                        // Boundary
                        pixel = Vec3b(255, 255, 255);
                    }
                    else if (region_idx <= 0 || region_idx > total_regions)
                    {
                        // Background
                        pixel = Vec3b(0, 0, 0);
                    }
                    else
                    {
                        // Apply four-color
                        int color_idx = colors[region_idx];
                        if (color_idx >= 0 && color_idx < 4)
                        {
                            pixel = FOUR_COLOR_PALETTE[color_idx];
                        }
                        else
                        {
                            // Fallback (shouldn't happen)
                            pixel = Vec3b(128, 128, 128);
                        }
                    }
                }
            }

            // Create a version of the four-color result with seed points
            Mat four_color_with_seeds = four_color_result.clone();

            // Draw seed points with numbers on the four-color result
            if (seeds.size() > 200)
            {
                // For many seeds, just show points without numbers
                for (int i = 0; i < seeds.size(); i++)
                {
                    // Draw a visible circle at each seed point with a contrasting color
                    circle(four_color_with_seeds, seeds[i], 3, Scalar(0, 255, 255), FILLED);
                    circle(four_color_with_seeds, seeds[i], 3, Scalar(0, 0, 0), 1); // Black outline
                }

                // Add a label showing how many seeds are displayed
                String seedCountText = format("Seeds: %zu", seeds.size());
                putText(four_color_with_seeds, seedCountText, Point(20, 30), FONT_HERSHEY_SIMPLEX,
                        0.7, Scalar(0, 0, 0), 2);
                putText(four_color_with_seeds, seedCountText, Point(20, 30), FONT_HERSHEY_SIMPLEX,
                        0.7, Scalar(255, 255, 255), 1);
            }
            else
            {
                // For fewer seeds, include numbers
                for (int i = 0; i < seeds.size(); i++)
                {
                    // Draw a visible circle at each seed point
                    circle(four_color_with_seeds, seeds[i], 3, Scalar(0, 255, 255), FILLED);
                    circle(four_color_with_seeds, seeds[i], 3, Scalar(0, 0, 0), 1); // Black outline

                    // Place the seed number next to the point
                    Point textPos(seeds[i].x + 5, seeds[i].y + 5);

                    // Add outlined text for better visibility against various backgrounds
                    putText(four_color_with_seeds, to_string(i + 1), textPos, FONT_HERSHEY_SIMPLEX,
                            0.4, Scalar(0, 0, 0), 2, LINE_AA); // Outlined text
                    putText(four_color_with_seeds, to_string(i + 1), textPos, FONT_HERSHEY_SIMPLEX,
                            0.4, Scalar(255, 255, 255), 1, LINE_AA); // White text
                }
            }

            // Display the four-color result with seed points
            imshow("four color result", four_color_with_seeds);

            // Verify that the four-coloring is valid
            bool valid_coloring = true;
            for (int region = 1; region <= total_regions; region++)
            {
                for (int neighbor : adjacency_list[region])
                {
                    if (colors[region] == colors[neighbor])
                    {
                        cout << "ERROR: Region " << region << " and neighbor " << neighbor
                             << " have the same color: " << colors[region] << endl;
                        valid_coloring = false;
                    }
                }
            }

            if (valid_coloring)
            {
                cout << "Four-coloring verification: PASSED" << endl;
            }
            else
            {
                cout << "Four-coloring verification: FAILED" << endl;
            }
        }
    }

    return 0;
}
