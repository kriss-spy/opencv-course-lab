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
// build/task2

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
#include "helper.h"

using namespace cv;
using namespace std;

// Global variables
Mat marker_mask, markers, img0, img, img_gray, wshed;
// marker_mask: Binary mask where seed points/markers are stored as non-zero values; used to identify initial regions for watershed
// markers: Integer matrix where each region is labeled with a unique positive number; watershed algorithm fills this with region IDs (-1 for boundaries)
// img0: Original input image that remains unchanged
// img: Working copy of the image that can be modified
// img_gray: Grayscale version of the image for processing and visualization
// wshed: Output image where watershed results are visualized with colored regions
Point prev_pt(-1, -1);
NextStep task2_next_step;
FourColor four_colors;

const int k_min = 1;
const int k_max = 5000; // TODO choose proper values for k range
// theoretical max for 600x600 image, about 100000?

int k;
double temperature;
double sigma;

// Four color palette - red, yellow, green, blue
const Vec3b FOUR_COLOR_PALETTE[4] = {
    Vec3b(0, 0, 255),   // RED
    Vec3b(0, 255, 255), // YELLOW
    Vec3b(0, 255, 0),   // GREEN
    Vec3b(255, 0, 0)    // BLUE
};

// Add a global variable to store adjacency meeting points
vector<pair<Point, pair<int, int>>> adjacency_points; // Point and the two region IDs

// Function to find neighbors of a region - completely rewritten to properly detect adjacency
vector<int> findNeighbors(const Mat &markers, int region_id, int total_regions)
{
    vector<bool> is_neighbor(total_regions + 1, false);
    vector<int> neighbors;

    // // Debug matrix to visualize where adjacencies are found
    // Mat debug_vis;
    // if (region_id == 3 || region_id == 10)
    // { // Focus on the problematic regions
    //     cvtColor(markers == region_id, debug_vis, COLOR_GRAY2BGR);
    // }

    // Detect boundary pixels for the current region
    Mat region_mask = (markers == region_id);

    // Dilate the region mask to find adjacent regions
    Mat dilated;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(region_mask, dilated, kernel, Point(-1, -1), 1);

    // Subtract the original region to get only the boundary area
    Mat boundary;
    subtract(dilated, region_mask, boundary);

    cout << "Checking neighbors for region " << region_id << ":" << endl;

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
                    // Only print when we first find a neighbor
                    if (!is_neighbor[neighbor_id])
                    {
#ifdef DEBUG
                        cout << "  Found neighbor " << neighbor_id << " at position (" << j << "," << i << ")" << endl;
#endif
                        // Store the meeting point with both region IDs
                        // We'll store the point only once for each region pair (lower ID first)
                        if (region_id < neighbor_id)
                        {
                            adjacency_points.push_back({Point(j, i), {region_id, neighbor_id}});
                            // old region_id and neighbor_id are not in the same field
                            // region_id, 1~...
                            // neighbor_id, which is markers.at<int>(i, j), has nothing to do with region_id
                            // changed region_id to markers value to fix this
                        }

                        // // Mark this point in the debug visualization if it's one of our focus regions
                        // if ((region_id == 3 && neighbor_id == 10) || (region_id == 10 && neighbor_id == 3))
                        // {
                        //     if (!debug_vis.empty())
                        //     {
                        //         // Draw a bright yellow marker at the adjacency point
                        //         circle(debug_vis, Point(j, i), 3, Scalar(0, 255, 255), -1);

                        //         // Save the debug image for inspection
                        //         string filename = "debug_region_" + to_string(region_id) +
                        //                           "_neighbor_" + to_string(neighbor_id) + ".png";
                        //         imwrite(filename, debug_vis);
                        //         cout << "  Saved debug image to " << filename << endl;
                        //     }
                        // }
                    }
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
    // Clear the global adjacency points vector before building a new adjacency list
    adjacency_points.clear();

    vector<vector<int>> adjacency_list(total_regions + 1);
    vector<bool> visited_regions(total_regions, false);

    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int region = markers.at<int>(i, j);
            if (region > 0 && !visited_regions[region])
            {
                visited_regions[region] = true;
                adjacency_list[region] = findNeighbors(markers, region, total_regions);
            }
        }
    }
#ifdef DEBUG
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
#endif
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

// New helper functions
// Initialize image, parameters, and masks
void initApp(vector<Point> &seeds)
{
    print_welcome();
    print_task2_help();

    double default_temperature = 0.01;
    double default_sigma = 1.02;
    task2_next_step = INPUT_IMAGE;
    string default_image = "image/fruits.jpg";
    img0 = get_image(default_image);
    cout << "Image size: " << img0.cols << "x" << img0.rows << " pixels" << endl;

    task2_next_step = INPUT_K;
    k = get_k(k_min, k_max);

    task2_next_step = INPUT_TEMP;
    temperature = get_temperature(0, 1, default_temperature);

    task2_next_step = INPUT_SIGMA;
    sigma = get_sigma(1, 2, default_sigma);

    img = img0.clone();
    img_gray = img0.clone();
    wshed = img0.clone();
    marker_mask = Mat::zeros(img.size(), CV_32SC1);
    markers = Mat::zeros(img.size(), CV_32SC1);
    cvtColor(img, marker_mask, COLOR_BGR2GRAY);
    cvtColor(marker_mask, img_gray, COLOR_GRAY2BGR);
}

// Setup display windows
void setupWindows()
{
    namedWindow("image", 1);
    namedWindow("watershed transform", 1);
    namedWindow("four color result", 1);
    imshow("image", img);
    imshow("watershed transform", wshed);
    task2_next_step = GENERATE_SEEDS;
}

// Main event loop handling all key commands
void runEventLoop(vector<Point> &seeds)
{
    RNG rng(getTickCount());

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
            visualize_points("image", img, seeds, 200);
        }
        if (c == 'g') // generate seeds
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);

            // Call generate_seeds with marker_mask as a parameter
            seeds = generate_seeds(img0, marker_mask, k, temperature, sigma);
            // seeds: Vector of Points representing the locations of seed points for watershed segmentation
            // These points are the centers of initial regions that will grow during watershed
            task2_next_step = WATERSHED;
        }
        if (c == 'w' && task2_next_step == WATERSHED)
        {
            // Clear markers before watershed
            markers = Mat::zeros(marker_mask.size(), CV_32SC1);

            // Find contours in marker_mask and use them as regions
            vector<vector<Point>> contours;
            // contours: Vector of point vectors, each representing a continuous outline in the marker_mask
            // Each contour represents the boundary of a seed region and will become a watershed marker

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

            // pause();

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

            visualize_points("watershed transform", wshed_with_seeds, seeds, 200);

            task2_next_step = FOURCOLOR;
        }

        if (c == 'c' && task2_next_step == FOURCOLOR)
        {
            // Get total number of regions (max marker value excluding boundaries and background)
            int total_regions = seeds.size(); // 1~seeds.size()

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

#ifdef DEBUG
            // Create a version of the four-color result with seed points
            visualize_regions("four color result", four_color_result, seeds, markers);
            // note that region numbering is markers.at<seed>
#endif

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
}

// Replace existing main with simplified version
int main(int argc, char **argv)
{

    vector<Point> seeds;
    initApp(seeds);
    setupWindows();
    runEventLoop(seeds);
    return 0;
}
