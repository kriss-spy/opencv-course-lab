// parent: task1.cpp

// task3
// 根据分水岭结果中各区域面积大小的“堆排序”结果，提示最大和最小面积，
// get the area of regions, heap sort, print max and min area in cli
// 用户输入查找范围（面积下界和上界），使用折半查找，程序对所有符合要求的分水岭结果（标记区域面积）进行突出显示
// input search range [lower_bound, upper_bound], use binary search, find all regions within range, color them and mark area in GUI
// 并以这些高亮区域的面积大小作为权值，进行哈夫曼编码（考虑深度+递归策略），绘制该哈夫曼树
// use these area values, huffman encode, draw huffman tree in a new window

// how to compile and run task3
// cd opencv-course-lab/opencv-watershed/solutions
// make
// ./build/task3

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>         // Required for std::lower_bound, std::upper_bound, std::sort
#include <vector>            // Required for std::vector
#include <unordered_set>     // Required for std::unordered_set
#include <queue>             // For Huffman priority_queue
#include <map>               // For Huffman codes map
#include <string>            // For Huffman codes
#include <fstream>           // For file output (DOT file)
#include <cstdlib>           // For system() call
#include "watershed_utils.h" // Include the utility functions
#include "sample.h"

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

vector<pair<int, int>> area_values;

// --- Huffman Coding Structures and Functions ---
struct HuffmanNode
{
    int label; // Region label (-1 for internal nodes)
    int area;  // Frequency (area)
    HuffmanNode *left, *right;

    HuffmanNode(int lbl, int ar) : label(lbl), area(ar), left(nullptr), right(nullptr) {}

    // Destructor to clean up child nodes if this node owns them
    // This is a simple destructor; for a robust solution, manage ownership carefully
    // or use smart pointers if the tree needs to persist beyond code generation.
    ~HuffmanNode()
    {
        delete left;
        delete right;
    }
};

struct CompareHuffmanNodes
{
    bool operator()(HuffmanNode *l, HuffmanNode *r)
    {
        return l->area > r->area; // Min-heap based on area
    }
};

#ifdef DEBUG
// Recursive function to print Huffman tree structure for debugging
void debugPrintHuffmanTreeRecursive(HuffmanNode *node, string current_code, int depth)
{
    if (!node)
    {
        return;
    }

    // Indentation for tree structure
    for (int i = 0; i < depth; ++i)
    {
        cout << "  ";
    }

    if (node->label != -1)
    { // Leaf node
        cout << "Leaf: Label=" << node->label << ", Area=" << node->area << ", Code=" << (current_code.empty() ? "0" : current_code) << endl;
    }
    else
    { // Internal node
        cout << "Internal: Area=" << node->area << " (Code Prefix: " << current_code << ")" << endl;
    }

    if (node->left)
    {
        for (int i = 0; i < depth; ++i)
            cout << "  ";
        cout << " L-> ";
        debugPrintHuffmanTreeRecursive(node->left, current_code + "0", depth + 1);
    }
    if (node->right)
    {
        for (int i = 0; i < depth; ++i)
            cout << "  ";
        cout << " R-> ";
        debugPrintHuffmanTreeRecursive(node->right, current_code + "1", depth + 1);
    }
}
#endif

// Recursively generates DOT representation for nodes and edges
// Returns the unique ID assigned to the current node
int generateDotRecursive(HuffmanNode *node, ofstream &out_file, const map<int, string> &codes, int &node_id_counter)
{
    if (!node)
    {
        // This case should ideally not be reached if called on a valid tree root.
        // However, if it can, returning a distinct value or handling it is important.
        return -1; // Indicates no node was processed, or an invalid node.
    }

    int current_node_id = node_id_counter++;
    string label_str;

#ifdef VERBOSE
    if (node->label != -1)
    { // Leaf node
        string code_str = "N/A";
        if (codes.count(node->label))
        {
            code_str = codes.at(node->label);
            // Huffman code for a single symbol tree is often "0"
            if (code_str.empty() && codes.size() == 1 && node->area > 0)
            {
                code_str = "0";
            }
        }
        // DOT language uses \\n for newlines inside labels with record shape
        label_str = "Label: " + to_string(node->label) + "\\nArea: " + to_string(node->area) + "\\nCode: " + code_str;
    }
    else
    { // Internal node
        label_str = "Internal\\nArea: " + to_string(node->area);
    }
#else // Not VERBOSE, make labels concise
    if (node->label != -1)
    { // Leaf node
        label_str = "L" + to_string(node->label) + ": " + to_string(node->area);
    }
    else
    { // Internal node
        label_str = to_string(node->area);
    }
#endif

    out_file << "  node" << current_node_id << " [label=\"" << label_str << "\"];\n";

    if (node->left)
    {
        int left_child_id = generateDotRecursive(node->left, out_file, codes, node_id_counter);
        if (left_child_id != -1)
        { // Ensure left child was valid before creating edge
            out_file << "  node" << current_node_id << " -> node" << left_child_id << " [label=\"0\"];\n";
        }
    }
    if (node->right)
    {
        int right_child_id = generateDotRecursive(node->right, out_file, codes, node_id_counter);
        if (right_child_id != -1)
        { // Ensure right child was valid
            out_file << "  node" << current_node_id << " -> node" << right_child_id << " [label=\"1\"];\n";
        }
    }
    return current_node_id;
}

// Generates the complete DOT file for the Huffman tree
void generateHuffmanDotFile(HuffmanNode *root, const map<int, string> &codes, const string &filename)
{
    ofstream out_file(filename);
    if (!out_file.is_open())
    {
        cerr << "Error: Could not open DOT file for writing: " << filename << endl;
        return;
    }

    out_file << "digraph HuffmanTree {\n";
    out_file << "  rankdir=TB; // Top-to-Bottom layout\n";
    out_file << "  graph [dpi=48]; // Set DPI for scaling (e.g., 48 for ~50% if default is ~96)\n";
    out_file << "  node [shape=record, style=rounded, fontname=\"Helvetica\"];\n";
    out_file << "  edge [fontname=\"Helvetica\"];\n";

    if (root)
    {
        int node_id_counter = 0;
        generateDotRecursive(root, out_file, codes, node_id_counter);
    }
    else
    {
        out_file << "  empty [label=\"Empty Tree\"];\n";
    }

    out_file << "}\n";
    out_file.close();
}
// --- End of Graphviz DOT file generation ---

// Recursive function to generate Huffman codes
void generateHuffmanCodesRecursive(HuffmanNode *root, string current_code, map<int, string> &huffman_codes_map)
{
    if (!root)
    {
        return;
    }

    // If it's a leaf node (has a label)
    if (root->label != -1)
    {                                                                               // Assuming -1 indicates an internal node
        huffman_codes_map[root->label] = current_code.empty() ? "0" : current_code; // Handle single node case
    }

    if (root->left)
    {
        generateHuffmanCodesRecursive(root->left, current_code + "0", huffman_codes_map);
    }
    if (root->right)
    {
        generateHuffmanCodesRecursive(root->right, current_code + "1", huffman_codes_map);
    }
}

// Builds Huffman Tree and generates codes
map<int, string> performHuffmanCoding(const vector<pair<int, int>> &regions_data)
{
    // regions_data is vector of {area, label}
    map<int, string> huffman_codes_map;
    if (regions_data.empty())
    {
        return huffman_codes_map;
    }

    priority_queue<HuffmanNode *, vector<HuffmanNode *>, CompareHuffmanNodes> pq;

    // Create a leaf node for each region and add it to the priority queue
    for (const auto &region_entry : regions_data)
    {
        // region_entry.first is area, region_entry.second is label
        if (region_entry.first > 0)
        { // Ensure area is positive for Huffman coding
            pq.push(new HuffmanNode(region_entry.second, region_entry.first));
        }
    }

    if (pq.empty())
    { // All regions had area 0 or less
        return huffman_codes_map;
    }

    // Special case: only one region/node
    if (pq.size() == 1)
    {
        HuffmanNode *root = pq.top();
        pq.pop();
        generateHuffmanCodesRecursive(root, "", huffman_codes_map); // Assign "0" or ""
        delete root;                                                // Clean up the single node
        return huffman_codes_map;
    }

    // Iterate while size of priority queue is more than 1
    while (pq.size() > 1)
    {
        HuffmanNode *left = pq.top();
        pq.pop();

        HuffmanNode *right = pq.top();
        pq.pop();

        // Create a new internal node with frequency equal to the sum of the two nodes' frequencies.
        // Label is -1 for internal nodes.
        HuffmanNode *top = new HuffmanNode(-1, left->area + right->area);
        top->left = left;
        top->right = right;
        pq.push(top);
    }

    // The remaining node is the root of the Huffman Tree
    HuffmanNode *root = pq.top();
    pq.pop(); // pq should be empty now

#ifdef DEBUG
    cout << "--- Huffman Tree Structure (Debug) ---" << endl;
    debugPrintHuffmanTreeRecursive(root, "", 0);
    cout << "--------------------------------------" << endl;
#endif

    generateHuffmanCodesRecursive(root, "", huffman_codes_map);

    // Call to generate DOT file before deleting the root
    if (root)
    { // Ensure root is not null before trying to generate DOT file
        generateHuffmanDotFile(root, huffman_codes_map, "huffman_tree.dot");
        // cout << "Generated huffman_tree.dot for Graphviz." << endl; // Moved to main loop for better user feedback
    }

    delete root; // This will recursively delete all nodes if destructor is set up correctly.

    return huffman_codes_map;
}

// --- Helper functions for Graphviz DOT file generation ---

// New helper functions for task3
void initApp_task3(vector<Point> &seeds)
{
    double default_temperature = 0.01;
    double default_sigma = 1.02;

    task3_next_step = INPUT_IMAGE;
    string default_image = "image/fruits.jpg"; // TODO tackle path problem when running from other locations
    img0 = get_image(default_image);

    std::cout << "Image size: " << img0.cols << "x" << img0.rows << " pixels" << std::endl;

    task3_next_step = INPUT_K;
    k = get_k(k_min, k_max);

    task3_next_step = INPUT_TEMP;
    temperature = get_temperature(0, 1, default_temperature);

    task3_next_step = INPUT_SIGMA;
    sigma = get_sigma(1, 2, default_sigma);

    print_task3_help();

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
    namedWindow("Search Area Value Range", WINDOW_AUTOSIZE); // Create the new window

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
        {
            task3_next_step = EXIT;
            break;
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
            task3_next_step = GENERATE_SEEDS;
        }

        if (c == 'g' && task3_next_step == GENERATE_SEEDS) // generate seeds
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);

            seeds = generate_seeds(img0, marker_mask, k, temperature, sigma);
            // seeds = cyj_generateSeeds(k, img0.rows, img0.cols);
            task3_next_step = WATERSHED;
        }

        if (c == 'v' && task3_next_step == WATERSHED)
        {
            visualize_points("image", img, seeds, 200);
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

            task3_next_step = HEAP_SORT_AREA;
        }
        if (c == 's' && task3_next_step == HEAP_SORT_AREA)
        {
            cout << "getting area values..." << endl;
            area_values = get_area_values(markers); // Now returns {label, area}

            if (area_values.empty())
            {
                cout << "No valid regions found to sort." << endl;
            }
            else
            {
                // Sort by area (first element of the pair)
                sort(area_values.begin(), area_values.end(), [](const pair<int, int> &a, const pair<int, int> &b)
                     { return a.second < b.second; });

                // min_element and max_element will now correctly use the first element (area)
                auto min_val_pair = area_values.front();
                auto max_val_pair = area_values.back();

                cout << "min area: " << min_val_pair.second << " (label " << min_val_pair.first << ")" << endl;
                cout << "max area: " << max_val_pair.second << " (label " << max_val_pair.first << ")" << endl;

                // Highlight min and max area regions in "watershed transform" window
                if (!wshed.empty() && !markers.empty())
                {
                    Mat wshed_with_highlights = wshed.clone(); // Start with a fresh clone of wshed

                    // --- Highlight Min Area Region ---
                    int min_area_label = min_val_pair.first;
                    int min_area_value = min_val_pair.second;
                    Vec3b highlight_color_min(255, 0, 255); // Bright Pink for min area
                    Point text_pos_min(-1, -1);

                    for (int r = 0; r < markers.rows; ++r)
                    {
                        for (int c = 0; c < markers.cols; ++c)
                        {
                            if (markers.at<int>(r, c) == min_area_label)
                            {
                                wshed_with_highlights.at<Vec3b>(r, c) = highlight_color_min;
                                if (text_pos_min.x == -1)
                                {
                                    text_pos_min = Point(c, r);
                                }
                            }
                        }
                    }
                    if (text_pos_min.x != -1)
                    {
                        string area_text_min = "Min Area: " + to_string(min_area_value);
                        Point display_pos_min(text_pos_min.x, text_pos_min.y - 5);
                        if (display_pos_min.x < 0)
                            display_pos_min.x = 0;
                        if (display_pos_min.y < 10)
                            display_pos_min.y = 10;
                        if (display_pos_min.x >= wshed_with_highlights.cols - 120)
                            display_pos_min.x = wshed_with_highlights.cols - 120;
                        if (display_pos_min.y >= wshed_with_highlights.rows)
                            display_pos_min.y = wshed_with_highlights.rows - 5;
                        putText(wshed_with_highlights, area_text_min, display_pos_min, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 2, LINE_AA);
                        putText(wshed_with_highlights, area_text_min, display_pos_min, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);
                    }
                    cout << "Highlighted min area region (Label: " << min_area_label << ") in 'watershed transform' window." << endl;

                    // --- Highlight Max Area Region ---
                    int max_area_label = max_val_pair.first;
                    int max_area_value = max_val_pair.second;
                    Vec3b highlight_color_max(0, 255, 255); // Cyan for max area
                    Point text_pos_max(-1, -1);

                    for (int r = 0; r < markers.rows; ++r)
                    {
                        for (int c = 0; c < markers.cols; ++c)
                        {
                            if (markers.at<int>(r, c) == max_area_label)
                            {
                                wshed_with_highlights.at<Vec3b>(r, c) = highlight_color_max;
                                if (text_pos_max.x == -1)
                                {
                                    text_pos_max = Point(c, r);
                                }
                            }
                        }
                    }

                    if (text_pos_max.x != -1)
                    {
                        string area_text_max = "Max Area: " + to_string(max_area_value);
                        // Try to position max area text differently if min area text is already there
                        // This is a simple offset, might need more sophisticated placement if regions overlap significantly
                        Point display_pos_max(text_pos_max.x, text_pos_max.y + 15);
                        if (text_pos_min.x != -1 && abs(text_pos_max.y - text_pos_min.y) < 20 && abs(text_pos_max.x - text_pos_min.x) < 100)
                        {
                            display_pos_max.y = text_pos_max.y + 20; // Shift down if too close to min text
                        }

                        if (display_pos_max.x < 0)
                            display_pos_max.x = 0;
                        if (display_pos_max.y < 10)
                            display_pos_max.y = 10;
                        if (display_pos_max.x >= wshed_with_highlights.cols - 120)
                            display_pos_max.x = wshed_with_highlights.cols - 120;
                        if (display_pos_max.y >= wshed_with_highlights.rows - 5)
                            display_pos_max.y = wshed_with_highlights.rows - 5;

                        putText(wshed_with_highlights, area_text_max, display_pos_max, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 2, LINE_AA);
                        putText(wshed_with_highlights, area_text_max, display_pos_max, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);
                    }
                    imshow("watershed transform", wshed_with_highlights);
                    cout << "Highlighted max area region (Label: " << max_area_label << ") in 'watershed transform' window." << endl;
                }
            }

            task3_next_step = MARK_AREA_WITHIN_RANGE;
        }
        if (c == 't' && task3_next_step == MARK_AREA_WITHIN_RANGE)
        {
            if (markers.empty() || img0.empty())
            {
                cout << "Please run watershed segmentation first (press 'w')." << endl;
                continue;
            }
            if (area_values.empty())
            {
                cout << "Please calculate area values first (press 's')." << endl;
                continue;
            }

            // area_values is already sorted by area by the 's' key logic.
            int min_possible_area = area_values.front().second;
            int max_possible_area = area_values.back().second;

            int lower_bound, upper_bound;
            get_area_range(lower_bound, upper_bound, min_possible_area, max_possible_area);
            cout << "Searching for regions with area between " << lower_bound << " and " << upper_bound << endl;

            // Use std::lower_bound and std::upper_bound to find the range of regions
            auto it_lower = std::lower_bound(area_values.begin(), area_values.end(), lower_bound,
                                             [](const pair<int, int> &elem, int val)
                                             {
                                                 return elem.second < val;
                                             });

            auto it_upper = std::upper_bound(it_lower, area_values.end(), upper_bound,
                                             [](int val, const pair<int, int> &elem)
                                             {
                                                 return val < elem.second;
                                             });

            vector<pair<int, int>> regions_in_range(it_lower, it_upper);

            if (regions_in_range.empty())
            {
                cout << "No regions found within the specified area range." << endl;
                // Display an empty or original image in the search window
                Mat empty_search_img = img0.clone();
                putText(empty_search_img, "No regions in range", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
                imshow("Search Area Value Range", empty_search_img);
            }
            else
            {
                cout << "Found " << regions_in_range.size() << " regions within the range." << endl;

                Mat search_result_img = img0.clone();                          // Start with the original image
                Mat regions_highlight_mask = Mat::zeros(img0.size(), CV_8UC3); // For highlighting regions

                unordered_set<int> labels_in_range;
                for (const auto &region_data : regions_in_range)
                {
                    labels_in_range.insert(region_data.first); // region_data.first is the label
                }

                // Color the regions in range on the highlight mask
                for (int i = 0; i < markers.rows; i++)
                {
                    for (int j = 0; j < markers.cols; j++)
                    {
                        int marker_id = markers.at<int>(i, j);
                        if (labels_in_range.count(marker_id))
                        {
                            regions_highlight_mask.at<Vec3b>(i, j) = Vec3b(0, 255, 255); // Highlight with yellow
                        }
                    }
                }

                addWeighted(search_result_img, 0.7, regions_highlight_mask, 0.3, 0, search_result_img);

                for (const auto &region_data : regions_in_range)
                {
                    int label = region_data.first;
                    int area = region_data.second;
                    Point text_pos = Point(-1, -1);
                    for (int r = 0; r < markers.rows; ++r)
                    {
                        for (int c = 0; c < markers.cols; ++c)
                        {
                            if (markers.at<int>(r, c) == label)
                            {
                                text_pos = Point(c, r);
                                goto found_point_for_text;
                            }
                        }
                    }
                found_point_for_text:
                    if (text_pos.x != -1)
                    {
                        Point display_pos(text_pos.x - 10, text_pos.y + 5);
                        if (display_pos.x < 0)
                            display_pos.x = 0;
                        if (display_pos.y < 0)
                            display_pos.y = 0;
                        if (display_pos.x >= search_result_img.cols - 50)
                            display_pos.x = search_result_img.cols - 50;
                        if (display_pos.y >= search_result_img.rows - 10)
                            display_pos.y = search_result_img.rows - 10;

                        string area_text = to_string(area);
                        putText(search_result_img, area_text, display_pos, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 2, LINE_AA);
                        putText(search_result_img, area_text, display_pos, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1, LINE_AA);
                    }
                }

                imshow("Search Area Value Range", search_result_img);
                cout << "Highlighted regions are shown in 'Search Area Value Range' window." << endl;
                // TODO actually implement huffman coding
                // --- Perform Huffman Coding ---
                cout << endl
                     << "Performing Huffman Coding for regions in range..." << endl;
                // regions_in_range is vector of {area, label}
                map<int, string> huffman_codes = performHuffmanCoding(regions_in_range);

                if (huffman_codes.empty() && !regions_in_range.empty())
                {
                    cout << "Could not generate Huffman codes (e.g. all regions in range have 0 area, or only one region with 0 area)." << endl;
                }
                else if (huffman_codes.empty() && regions_in_range.empty())
                {
                    // Already handled by "No regions found..."
                }
                else
                {
#ifdef DEBUG
                    cout << "Huffman Codes:" << endl;
                    // To print in a more readable order (e.g., by label or by area), sort if needed.
                    // For now, iterating through map (ordered by label by default for std::map)
                    for (const auto &code_pair : huffman_codes)
                    {
                        // code_pair.first is label, code_pair.second is code
                        cout << "  Region Label: " << code_pair.first;
                        // Find area for this label from regions_in_range for display
                        int area_for_label = 0;
                        bool area_found_for_label = false;
                        for (const auto &region_detail : regions_in_range)
                        {
                            if (region_detail.second == code_pair.first)
                            { // region_detail is {area, label}
                                area_for_label = region_detail.first;
                                area_found_for_label = true;
                                break;
                            }
                        }
                        if (area_found_for_label)
                        {
                            cout << " (Area: " << area_for_label << ")";
                        }
                        else
                        {
                            cout << " (Area: N/A)"; // Should not happen if label is from regions_in_range
                        }
                        cout << " -> Code: " << code_pair.second << endl;
                    }
#endif
                }
                // --- End of Huffman Coding ---
                // --- Visualize Huffman Tree using Graphviz ---
                cout << endl
                     << "Attempting to generate Huffman tree image using Graphviz..." << endl;
                // Check if huffman_tree.dot was generated (e.g. if regions_in_range was not empty and huffman coding produced a tree)
                // The dot file is generated within performHuffmanCoding if root is not null.
                // We assume if huffman_codes is not empty, a tree was attempted.
                if (!huffman_codes.empty() || (regions_in_range.size() == 1 && regions_in_range[0].first > 0))
                {
                    // The file "huffman_tree.dot" should exist if performHuffmanCoding created a tree.
                    int result = system("dot -Tpng huffman_tree.dot -o huffman_tree.png");
                    if (result == 0)
                    {
                        Mat tree_image = imread("huffman_tree.png");
                        if (!tree_image.empty())
                        {
                            namedWindow("Huffman Tree", WINDOW_AUTOSIZE);
                            imshow("Huffman Tree", tree_image);
                            cout << "Huffman tree image 'huffman_tree.png' generated and displayed." << endl;
                            cout << "Close the 'Huffman Tree' window to exit or continue." << endl;
                        }
                        else
                        {
                            cout << "Failed to load huffman_tree.png. 'dot' command might have succeeded but image is invalid or not found." << endl;
                        }
                    }
                    else
                    {
                        cout << "Failed to execute Graphviz 'dot' command (exit code: " << result << "). Ensure Graphviz is installed and 'dot' is in your system's PATH." << endl;
                        cout << "You can manually convert 'huffman_tree.dot' using: dot -Tpng huffman_tree.dot -o huffman_tree.png" << endl;
                    }
                }
                else if (regions_in_range.empty())
                {
                    // No regions, so no tree to visualize. Message already printed.
                }
                else
                {
                    cout << "No Huffman tree was generated (e.g., all regions had zero area), so no visualization." << endl;
                }
                // --- End of Huffman Tree Visualization ---
            }
            task3_next_step = EXIT; // Set next step to EXIT as requested
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
