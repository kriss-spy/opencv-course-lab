// parent: task1.cpp

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
// ./build/task2

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
#include <algorithm>         // Added for std::sort
#include "watershed_utils.h" // Include the utility functions
#include "sample.h"

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

// Global variables for displaying mouse marker info on GUI
Mat g_four_color_result_base_img;
string g_mouse_marker_text = "Mouse over to see Marker ID";
int g_actual_total_regions = 0; // Added global variable for actual region count

// Mouse callback function for the "four color result" window
void on_mouse_four_color_result(int event, int x, int y, int flags, void *userdata)
{
    if (event == EVENT_MOUSEMOVE)
    {
        if (g_four_color_result_base_img.empty() || markers.empty() || markers.type() != CV_32SC1)
        {
            // Base image or markers not ready, or markers have unexpected type
            g_mouse_marker_text = "Marker data not available";
        }
        else if (x >= 0 && x < markers.cols && y >= 0 && y < markers.rows)
        {
            int region_idx = markers.at<int>(y, x);
            g_mouse_marker_text = "Marker ID at (" + to_string(x) + "," + to_string(y) + "): " + to_string(region_idx);
        }
        else
        {
            g_mouse_marker_text = "Mouse out of bounds";
        }

        if (!g_four_color_result_base_img.empty())
        {
            Mat display_img_with_text = g_four_color_result_base_img.clone();
            Point text_origin(10, display_img_with_text.rows - 10);
            Scalar text_color_main(255, 255, 255); // White
            Scalar text_color_outline(0, 0, 0);    // Black
            double font_scale = 0.4;
            int thickness_main = 1;
            int thickness_outline = 2;

            // Draw outline text
            putText(display_img_with_text, g_mouse_marker_text, text_origin, FONT_HERSHEY_SIMPLEX, font_scale, text_color_outline, thickness_outline, LINE_AA);
            // Draw main text
            putText(display_img_with_text, g_mouse_marker_text, text_origin, FONT_HERSHEY_SIMPLEX, font_scale, text_color_main, thickness_main, LINE_AA);

            imshow("four color result", display_img_with_text);
        }
    }
}

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
vector<int> findNeighbors(const Mat &markers_mat, int region_id, int total_regions)
{
    vector<bool> is_neighbor(total_regions + 1, false);
    vector<int> neighbors;

    Mat region_mask = (markers_mat == region_id);
    Mat dilated_region;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(region_mask, dilated_region, kernel);

    // Iterate over the original image bounds
    for (int r = 0; r < markers_mat.rows; ++r)
    {
        for (int c = 0; c < markers_mat.cols; ++c)
        {
            // Check if this pixel is part of the dilated region but not the original region
            if (dilated_region.at<uchar>(r, c) > 0 && !region_mask.at<uchar>(r, c))
            {
                int neighbor_id = markers_mat.at<int>(r, c);
                // Ensure it's a valid, different region and not already added
                if (neighbor_id > 0 && neighbor_id <= total_regions && neighbor_id != region_id && !is_neighbor[neighbor_id])
                {
                    is_neighbor[neighbor_id] = true;
                    neighbors.push_back(neighbor_id);
#ifdef DEBUG
                    // cout << "  Region " << region_id << " found neighbor " << neighbor_id << " at (" << c << "," << r << ")" << endl;
#endif
                }
            }
        }
    }
    return neighbors;
}

// Function to build adjacency list for regions
vector<vector<int>> buildAdjacencyList(const Mat &markers_mat, int total_regions)
{
    adjacency_points.clear(); // Assuming adjacency_points is a global or accessible variable for debug
    vector<vector<int>> adjacency_list(total_regions + 1);

    // Iterate through each possible region ID from 1 to total_regions
    for (int region = 1; region <= total_regions; ++region)
    {
        adjacency_list[region] = findNeighbors(markers_mat, region, total_regions);
    }

#ifdef DEBUG
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

struct NodeInfo
{
    uint8_t forbid = 0;  // 4 bits: colours present in neighbourhood
    uint8_t tried = 0;   // 4 bits: colours already attempted here
    uint8_t satDeg = 0;  // # different colours in neighbourhood
    uint16_t degree = 0; // plain degree (tie-break for DSATUR)
    int colour = -1;     // current colour (-1 = uncoloured)
};

/* helper: first colour not in mask, -1 if none */
static inline int firstFree(uint8_t mask)
{
    for (int c = 0; c < 4; ++c)
        if (!(mask & (1u << c)))
            return c;
    return -1;
}

std::vector<int>
fourColorRegions(const std::vector<std::vector<int>> &g, int nRegions)
{
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    const int N = nRegions;
    std::vector<int> result(N + 1, -1);

    // For very small graphs, use simple greedy coloring (faster)
    if (N <= 10)
    {
        for (int v = 1; v <= N; ++v)
        {
            bool used[4] = {false};
            for (int nb : g[v])
                if (result[nb] >= 0)
                    used[result[nb]] = true;

            for (int c = 0; c < 4; ++c)
                if (!used[c])
                {
                    result[v] = c;
                    break;
                }

            if (result[v] == -1)
                result[v] = 0;
        }
        return result;
    }

    // For larger graphs, use hybrid DSatur/Kempe chains approach
    std::vector<NodeInfo> V(N + 1);

    // Initialize degrees for all vertices
    for (int v = 1; v <= N; ++v)
        V[v].degree = static_cast<uint16_t>(g[v].size());

    // Sort vertices by degree for better initial ordering
    std::vector<std::pair<int, int>> order_by_degree;
    for (int v = 1; v <= N; ++v)
        order_by_degree.push_back({g[v].size(), v});

    std::sort(order_by_degree.begin(), order_by_degree.end(),
              [](auto &a, auto &b)
              { return a.first > b.first; });

    // Start with highest degree vertex
    int first_v = order_by_degree[0].second;
    V[first_v].colour = 0;

    // Main coloring stack
    std::vector<int> stack = {first_v};
    stack.reserve(N);

    // Update constraints for first vertex's neighbors
    for (int nb : g[first_v])
    {
        V[nb].forbid |= 1u << 0;
        V[nb].satDeg++;
    }

    // Choose next vertex based on saturation degree (DSatur)
    auto chooseNext = [&]() -> int
    {
        int best = -1;
        uint8_t bestSat = 0;
        uint16_t bestDeg = 0;

        for (int v = 1; v <= N; ++v)
        {
            if (V[v].colour != -1)
                continue;

            uint8_t s = V[v].satDeg;
            if (s > bestSat || (s == bestSat && V[v].degree > bestDeg))
            {
                best = v;
                bestSat = s;
                bestDeg = V[v].degree;
            }
        }
        return best;
    };

    // DSatur algorithm with limited backtracking
    int next = chooseNext();
    int backtrack_count = 0;
    const int max_backtrack = std::min(N / 10, 50); // More aggressive limit for large graphs

    while (next != -1 && stack.size() < N)
    {
        NodeInfo &info = V[next];

        // Find first available color (not forbidden and not already tried)
        int c = firstFree(info.forbid | info.tried);

        if (c == -1)
        {
            // Need to backtrack
            backtrack_count++;

            if (backtrack_count > max_backtrack || stack.empty())
            {
                // Too much backtracking - exit early
                break;
            }

            // Reset for future visits
            info.tried = 0;

            // Backtrack to previous vertex
            next = stack.back();
            stack.pop_back();

            // Undo color assignment
            int old_color = V[next].colour;
            V[next].colour = -1;
            V[next].tried |= (1u << old_color); // Don't try this color again

            // Update neighbor constraints
            for (int nb : g[next])
            {
                // Check if we're actually removing a constraint
                bool removing = true;
                for (int other : g[nb])
                {
                    if (other != next && V[other].colour == old_color)
                    {
                        removing = false;
                        break;
                    }
                }

                if (removing)
                {
                    V[nb].forbid &= ~(1u << old_color);
                    if (V[nb].satDeg > 0)
                        V[nb].satDeg--;
                }
            }
            continue;
        }

        // Try color c for current vertex
        info.colour = c;
        info.tried |= (1u << c);
        stack.push_back(next);

        // Update neighbor constraints
        for (int nb : g[next])
        {
            if (V[nb].colour == -1)
            {
                uint8_t before = V[nb].forbid;
                V[nb].forbid |= (1u << c);
                if (before != V[nb].forbid)
                    V[nb].satDeg++;
            }
        }

        next = chooseNext();

        // Early timeout exit
        if (std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count() > 700)
            break;
    }

    // Copy partial results
    for (int v = 1; v <= N; ++v)
        if (V[v].colour != -1)
            result[v] = V[v].colour;

    // Complete any uncolored vertices with greedy approach
    std::vector<int> uncolored;
    for (int v = 1; v <= N; ++v)
        if (result[v] == -1)
            uncolored.push_back(v);

    // Sort uncolored vertices by degree (highest first)
    std::sort(uncolored.begin(), uncolored.end(),
              [&g](int a, int b)
              { return g[a].size() > g[b].size(); });

    // Color remaining vertices
    for (int v : uncolored)
    {
        bool used[4] = {false};

        // Check colors used by neighbors
        for (int nb : g[v])
            if (result[nb] >= 0 && result[nb] < 4)
                used[result[nb]] = true;

        // Find first available color
        int c = 0;
        while (c < 4 && used[c])
            c++;

        // Assign color (or fallback to 0 if needed)
        result[v] = (c < 4) ? c : 0;
    }

    // Final validation and conflict resolution
    bool conflicts_exist;
    int iterations = 0;
    const int MAX_ITERATIONS = 3; // Limit resolution attempts

    do
    {
        conflicts_exist = false;
        iterations++;

        // Process vertices in order of their degree
        for (auto &[deg, v] : order_by_degree)
        {
            // Check for conflicts with neighbors
            for (int nb : g[v])
            {
                if (result[v] == result[nb] && v < nb)
                {
                    conflicts_exist = true;

                    // Find alternative color for current vertex
                    bool used[4] = {false};
                    for (int other_nb : g[v])
                        if (other_nb != nb) // Skip the conflicting neighbor
                            used[result[other_nb]] = true;

                    // Try to find a color that's not used by other neighbors
                    int alt_color = 0;
                    while (alt_color < 4 && (used[alt_color] || alt_color == result[nb]))
                        alt_color++;

                    if (alt_color < 4)
                        result[v] = alt_color;
                    else
                    {
                        // If no alternative for v, try changing neighbor instead
                        bool nb_used[4] = {false};
                        for (int other_nb : g[nb])
                            if (other_nb != v)
                                nb_used[result[other_nb]] = true;

                        int nb_alt_color = 0;
                        while (nb_alt_color < 4 && (nb_used[nb_alt_color] || nb_alt_color == result[v]))
                            nb_alt_color++;

                        if (nb_alt_color < 4)
                            result[nb] = nb_alt_color;
                        else
                            result[v] = (result[v] + 1) % 4; // Cyclic shift as last resort
                    }
                    break; // Handle one conflict at a time
                }
            }
        }
    } while (conflicts_exist && iterations < MAX_ITERATIONS);

    // Final check - if any conflicts remain, use brute force coloring
    if (iterations >= MAX_ITERATIONS)
    {
        // This is a guaranteed valid coloring using a sweep approach
        for (int v = 1; v <= N; ++v)
        {
            std::vector<bool> used(4, false);

            for (int nb : g[v])
                if (nb < v) // Only consider already-processed neighbors
                    used[result[nb]] = true;

            for (int c = 0; c < 4; ++c)
            {
                if (!used[c])
                {
                    result[v] = c;
                    break;
                }
            }

            // In the unlikely case no color works, just use color 0
            // (This shouldn't happen as four colors are sufficient)
            if (result[v] == -1)
                result[v] = 0;
        }
    }

    return result;
}

// Structure to hold region ID and its degree for sorting
struct RegionInfo
{
    int id;
    int degree;

    bool operator<(const RegionInfo &other) const
    {
        // Primary sort: by degree descending (most constrained first)
        if (degree != other.degree)
        {
            return degree > other.degree;
        }
        // Secondary sort: by ID ascending (for stability/determinism)
        return id < other.id;
    }
};

// Modified backtracking function using a pre-sorted list of regions by degree
bool solveColoringRecursive(int region_idx_in_sorted_list,
                            const vector<RegionInfo> &sorted_regions_list,
                            vector<int> &colors,
                            const vector<vector<int>> &adjacency_list,
                            int total_regions_count)
{
    if (region_idx_in_sorted_list == total_regions_count)
    {
        return true; // All regions processed and colored
    }

    int current_region_id = sorted_regions_list[region_idx_in_sorted_list].id;

    // Try each of the four colors
    for (int color = 0; color < 4; ++color)
    {
        if (isValidColor(current_region_id, color, colors, adjacency_list))
        {
            colors[current_region_id] = color;

            if (solveColoringRecursive(region_idx_in_sorted_list + 1, sorted_regions_list, colors, adjacency_list, total_regions_count))
            {
                return true;
            }

            colors[current_region_id] = -1; // Backtrack
        }
    }

    return false; // No color worked for current_region_id
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
void initApp_task2(vector<Point> &seeds)
{
    print_welcome();
    print_task2_help();

    // double default_sigma = 1.02;
    task2_next_step = INPUT_IMAGE;
    // string default_image = "image/fruits.jpg";
    img0 = get_image();
    cout << "Image size: " << img0.cols << "x" << img0.rows << " pixels" << endl;

    task2_next_step = INPUT_K;
    k = get_k(k_min, k_max);

    task2_next_step = INPUT_TEMP;
    temperature = get_temperature(0, 1);

    // task2_next_step = INPUT_SIGMA;
    // sigma = get_sigma(1, 2, default_sigma);

    img = img0.clone();
    img_gray = img0.clone();
    wshed = img0.clone();
    marker_mask = Mat::zeros(img.size(), CV_32SC1);
    markers = Mat::zeros(img.size(), CV_32SC1);
    cvtColor(img, marker_mask, COLOR_BGR2GRAY);
    cvtColor(marker_mask, img_gray, COLOR_GRAY2BGR);
}

// Setup display windows
void setupWindows_task2()
{
    namedWindow("image", 1);
    namedWindow("watershed transform", 1);
    namedWindow("four color result", 1);
    imshow("image", img);
    imshow("watershed transform", wshed);
    setMouseCallback("four color result", on_mouse_four_color_result, 0);
    task2_next_step = GENERATE_SEEDS;
}

// Main event loop handling all key commands
void runEventLoop_task2(vector<Point> &seeds)
{
    RNG rng(getTickCount());
    int comp_count = 0; // Moved comp_count to a scope accessible by both 'w' and 'c' if needed, or use global.
                        // For this fix, g_actual_total_regions will be used.

    for (;;)
    {
        int c = waitKey(0);
        if (c == 27 || c == 'q')
        {
            task2_next_step = EXIT;
            break;
        }
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
            task2_next_step = GENERATE_SEEDS;
        }

        if (c == 'g' && task2_next_step == GENERATE_SEEDS) // generate seeds
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);

            // Call generate_seeds with marker_mask as a parameter
            seeds = generate_seeds(img0, marker_mask, k, temperature);
            // seeds: Vector of Points representing the locations of seed points for watershed segmentation
            // These points are the centers of initial regions that will grow during watershed
            task2_next_step = WATERSHED;
        }

        if (c == 'v' && task2_next_step == WATERSHED)
        {
            visualize_points("image", img, seeds, 200);
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

            comp_count = 0; // Reset comp_count before assigning region IDs
            for (int i = 0; i < contours.size(); i++)
            {
                drawContours(markers, contours, i, Scalar(i + 1), -1); // Region IDs are 1 to contours.size()
                comp_count++;
            }
            g_actual_total_regions = comp_count; // Store the actual number of regions

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
            if (g_actual_total_regions == 0)
            {
                cout << "Please run watershed segmentation first (press 'w')." << endl;
                continue;
            }
            int total_regions = g_actual_total_regions; // Use the actual region count

            cout << "Total regions for four-coloring: " << total_regions << endl;

            // Build adjacency list
            double t_adj = (double)getTickCount();
            vector<vector<int>> adjacency_list = buildAdjacencyList(markers, total_regions);
            t_adj = (double)getTickCount() - t_adj;
            printf("Adjacency list build time = %gms\n", t_adj / getTickFrequency() * 1000.);

            // Apply four-color theorem
            double t_color = (double)getTickCount();
            vector<int> colors(total_regions + 1, -1);

            // Prepare sorted list of regions by degree (descending)
            vector<RegionInfo> sorted_region_list(total_regions);
            for (int i = 0; i < total_regions; ++i)
            {
                sorted_region_list[i].id = i + 1; // Region IDs are 1-based
                sorted_region_list[i].degree = adjacency_list[i + 1].size();
            }
            std::sort(sorted_region_list.begin(), sorted_region_list.end());

            colors = fourColorRegions(adjacency_list, total_regions); // TODO optimize four color to meet requirement

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

            imshow("four color result", four_color_result);
        }
    }
}

// Replace existing main with simplified version
int main(int argc, char **argv)
{

    vector<Point> seeds;
    initApp_task2(seeds);
    setupWindows_task2();
    runEventLoop_task2(seeds);
    return 0;
}
