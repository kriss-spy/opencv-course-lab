#ifndef WATERSHED_UTILS_H
#define WATERSHED_UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <random>
#include <vector>                        // For std::vector in format_string
#include <cstdarg>                       // For va_list in format_string
#include <cstdio>                        // For vsnprintf in format_string
#include <cstring>                       // For strerror
#include <cerrno>                        // For errno
#include <functional>                    // For std::function
#include <opencv2/core/utils/logger.hpp> // For setLogLevel
#ifdef _WIN32
#include <direct.h>
#define getcwd _getcwd
#else
#include <unistd.h>
#endif
#include <climits> // For PATH_MAX

using namespace cv;
using namespace std;

// ANSI escape codes for colors
const string RESET_COLOR = "\033[0m";
const string RED_COLOR = "\033[31m";
const string GREEN_COLOR = "\033[32m";
const string YELLOW_COLOR = "\033[33m";
const string BLUE_COLOR = "\033[34m";
const string MAGENTA_COLOR = "\033[35m";
const string CYAN_COLOR = "\033[36m";
const string WHITE_COLOR = "\033[37m";
const string BOLD_STYLE = "\033[1m";

// Emojis/Icons
const string ICON_INFO = "ℹ️ ";
const string ICON_SUCCESS = "✅ ";
const string ICON_WARNING = "⚠️  ";
const string ICON_ERROR = "❌ ";
const string ICON_DEBUG = "🐞 ";
const string ICON_PROMPT_ARROW = "➡️ "; // For prompts if desired, or keep plain
const string ICON_HEADER = "🚀 ";
const string ICON_HELP = "💡 ";

enum MessageType
{
    MSG_INFO,
    MSG_SUCCESS,
    MSG_WARNING,
    MSG_ERROR,
    MSG_DEBUG,
    MSG_PROMPT, // For user input lines like "> "
    MSG_HEADER,
    MSG_HELP,
    MSG_PLAIN // For direct output without icons/colors, like simple prompts
};

// Helper function to format strings like printf
std::string format_string(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);
    int size = vsnprintf(nullptr, 0, fmt, args_copy);
    va_end(args_copy);

    if (size < 0)
    {
        va_end(args);
        // Consider throwing an exception or returning a specific error string
        return "Error formatting string";
    }

    std::vector<char> buffer(size + 1);
    vsnprintf(buffer.data(), buffer.size(), fmt, args);
    va_end(args);
    return std::string(buffer.data());
}

void print_sth(MessageType type, const string &message, bool newline = true)
{
    string prefix_icon = "";
    string color_code = RESET_COLOR;
    string style_code = "";

    switch (type)
    {
    case MSG_INFO:
        prefix_icon = ICON_INFO;
        color_code = CYAN_COLOR;
        break;
    case MSG_SUCCESS:
        prefix_icon = ICON_SUCCESS;
        color_code = GREEN_COLOR;
        break;
    case MSG_WARNING:
        prefix_icon = ICON_WARNING;
        color_code = YELLOW_COLOR;
        break;
    case MSG_ERROR:
        prefix_icon = ICON_ERROR;
        color_code = RED_COLOR;
        break;
    case MSG_DEBUG:
        prefix_icon = ICON_DEBUG;
        color_code = MAGENTA_COLOR;
        break;
    case MSG_PROMPT: // Used for lines expecting user input
        // No icon, simple prompt
        std::cout << message;
        if (newline)
            std::cout << std::endl;
        return;
    case MSG_HEADER:
        prefix_icon = ICON_HEADER;
        color_code = BOLD_STYLE + MAGENTA_COLOR;
        break;
    case MSG_HELP:
        prefix_icon = ICON_HELP;
        color_code = BLUE_COLOR;
        break;
    case MSG_PLAIN: // For simple, unstyled output like "> "
        std::cout << message;
        if (newline)
            std::cout << std::endl;
        return;
    }
    std::cout << color_code << style_code << prefix_icon << RESET_COLOR << color_code << message << RESET_COLOR;
    if (newline)
    {
        std::cout << std::endl;
    }
}

// Enum for application state
enum NextStep
{
    INPUT_IMAGE,
    INPUT_K,
    INPUT_TEMP,
    INPUT_SIGMA,
    GENERATE_SEEDS,
    WATERSHED,
    FOURCOLOR,
    HEAP_SORT_AREA,
    MARK_AREA_WITHIN_RANGE,
    HUFFMAN_TREE,
    EXIT
};

enum FourColor
{
    RED,
    YELLOW,
    GREEN,
    BLUE
};

// Print help information
void print_task1_help()
{
    string help_text =
        "Task 1: Seed-based Watershed Segmentation\n"
        "\tUse the seed-based marker watershed algorithm (OpenCV's built-in `watershed`) to oversegment the input image.\n\n"
        "\tThe user provides an input image and an integer K. The program should automatically compute\n"
        "\tK random seed points, ensuring the distance between any two seed points is > (M*N/K)^0.5\n\n"
        "\tThen, the program should mark the positions and labels of all seed points on the original image\n"
        "\tand visualize the watershed algorithm results using semi-transparent random coloring.\n\n"
        "Hot keys: \n"
        "\tESC or q - quit the program\n"
        "\tr - restore the original image\n"
        "\tg - generate seeds\n"
        "\tv - visualize generated seeds\n"
        // "\tc - clear input and restart\n"
        "\tw - run watershed\n";
    print_sth(MSG_HELP, help_text);
}

void print_task2_help()
{
    string help_text =
        "Task 2: Four-Color Theorem for Watershed Segmentation\n"
        "\tUsing adjacency lists to analyze neighboring relationships between regions in watershed segmentation results\n"
        "\tand applying the four-color theorem to recolor watershed results.\n\n"
        "\tThis task analyzes region adjacency in watershed segmented images and demonstrates the four-color theorem\n"
        "\tby recoloring regions such that no adjacent regions share the same color, using only four distinct colors.\n\n"
        "Hot keys: \n"
        "\tESC or q - quit the program\n"
        "\tr - restore the original image\n"
        "\tg - generate seeds\n"
        "\tv - visualize generated seeds\n"
        //    "\tc - clear input and restart\n"
        "\tw - run watershed\n"
        "\tc - perform four color\n";
    print_sth(MSG_HELP, help_text);
}

void print_task3_help()
{
    string help_text =
        "Task 3: Region Area Analysis and Huffman Coding\n"
        "\tBased on the \"heap sort\" results of the area sizes, print the maximum and minimum areas.\n"
        "\tUsers input a search range (lower and upper bounds for area). Using binary search, the program\n"
        "\thighlights all watershed results (marked regions) that meet the criteria.\n\n"
        "\tThese highlighted regions' area sizes are then used as weights to perform Huffman coding,\n"
        "\tand the corresponding Huffman tree is visualized.\n\n"
        "Hot keys: \n"
        "\tESC or q - quit the program\n"
        "\tr - restore the original image\n"
        "\tg - generate seeds\n"
        "\tv - visualize generated seeds\n"
        //    "\tc - clear input and restart\n"
        "\tw - run watershed\n"
        "\ts - sort area values and print min and max\n"
        "\tt - input search range, mark areas, draw huffman tree";
    print_sth(MSG_HELP, help_text);
}

// Print current directory for debugging
void print_current_dir()
{
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
    {
        print_sth(MSG_DEBUG, "Current working directory: " + string(cwd));
    }
    else
    {
        print_sth(MSG_ERROR, string("getcwd() error: ") + strerror(errno));
    }
}

// Get image from user input or use default
Mat get_image()
{
    string default_path = "image/conbini-trolley.jpg";
    print_current_dir();

    Mat img0;
    print_sth(MSG_PROMPT, "Please input image name from 'image/' folder (e.g., fruits.jpg)");
    print_sth(MSG_PROMPT, "Press enter to use default image: " + default_path.substr(default_path.find_last_of("/\\") + 1));
    while (true)
    {
        string filepath = "image/";
        print_sth(MSG_PLAIN, "> ", false);
        string input;
        getline(cin, input);
        if (!input.empty())
        {
            filepath += input;
            // Temporarily disable OpenCV's own error logging for imread
            cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
            img0 = imread(filepath, 1);
            // Restore a more verbose log level (e.g., WARNING or INFO)
            cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING); // Or your preferred default

            if (img0.empty())
            {
                print_sth(MSG_WARNING, "Could not open or find the image. Please input a valid path!");
                continue;
            }
            else
            {
                print_sth(MSG_SUCCESS, "Image loaded: " + filepath);
                return img0;
            }
        }
        else
        {
            // Temporarily disable OpenCV's own error logging for imread
            cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
            img0 = imread(default_path, 1);
            // Restore a more verbose log level
            cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING); // Or your preferred default

            if (img0.empty())
            {
                print_sth(MSG_ERROR, "Default image could not be loaded: " + default_path);
                // Potentially exit or throw an error if default image is critical
                // For now, we'll let the loop continue or caller handle empty Mat
                return Mat();
            }
            print_sth(MSG_SUCCESS, "Default image loaded: " + default_path);
            return img0;
        }
    }
}

// Get k (number of seeds) from user input
int get_k(int k_min, int k_max)
{
    int k = 0;
    print_sth(MSG_PROMPT, format_string("Please input k (number of random seed points, e.g., 100, 500, 1000). Range: [%d, %d]", k_min, k_max));
    while (true)
    {
        print_sth(MSG_PLAIN, "> ", false);
        string input;
        getline(cin, input);
        if (!input.empty())
        {
            // Check for non-digit characters
            bool has_non_digit = false;
            for (char c : input)
            {
                if (!isdigit(c))
                {
                    // Allow a leading minus sign if k_min can be negative, though current context implies positive k
                    // For this specific function, k is expected to be positive.
                    has_non_digit = true;
                    break;
                }
            }

            if (has_non_digit)
            {
                print_sth(MSG_ERROR, "Invalid input! Please enter a whole number without non-digit characters (e.g., 'e', '.').");
                continue;
            }

            try
            {
                k = stoi(input);
                if (k < k_min || k > k_max)
                {
                    print_sth(MSG_WARNING, format_string("k is out of range! Please enter a value between %d and %d.", k_min, k_max));
                }
                else
                {
                    return k;
                }
            }
            catch (const std::invalid_argument &e)
            {
                // This case might be less likely now with the pre-check, but kept for robustness
                print_sth(MSG_ERROR, "Invalid input! Please enter a valid number.");
            }
            catch (const std::out_of_range &e)
            {
                print_sth(MSG_ERROR, "Input is too large! Please enter a valid number within the integer range.");
            }
        }
        else
        {
            print_sth(MSG_PROMPT, "No input received. Please input k!");
            continue;
        }
    }
}

// Get temperature parameter from user input
double get_temperature(double t_min, double t_max)
{
    double t;
    double t_default = 0.01;
    print_sth(MSG_PROMPT, format_string("Please input temperature (e.g., 0.0 to 1.0). Press enter to use default: %.2f", t_default));
    while (true)
    {
        print_sth(MSG_PLAIN, "> ", false);
        string input;
        getline(cin, input);
        if (!input.empty())
        {
            try
            {
                t = stof(input);
                if (t < t_min || t > t_max)
                {
                    print_sth(MSG_WARNING, format_string("Temperature is out of range! Please enter a value between %.2f and %.2f.", t_min, t_max));
                }
                else
                {
                    return t;
                }
            }
            catch (const std::invalid_argument &e)
            {
                print_sth(MSG_ERROR, "Invalid input! Please enter a valid number.");
            }
            catch (const std::out_of_range &e)
            {
                print_sth(MSG_ERROR, "Input is too large! Please enter a valid number.");
            }
        }
        else
        {
            print_sth(MSG_INFO, format_string("Using default temperature: %.2f", t_default));
            return t_default;
        }
    }
}

// Get sigma parameter from user input
double get_sigma(double sigma_min, double sigma_max, double sigma_default)
{
    double sigma;
    print_sth(MSG_PROMPT, format_string("Please input sigma. Press enter to use default: %.2f", sigma_default));
    while (true)
    {
        print_sth(MSG_PLAIN, "> ", false);
        string input;
        getline(cin, input);
        if (!input.empty())
        {
            try
            {
                sigma = stof(input);
                if (sigma < sigma_min || sigma > sigma_max)
                {
                    print_sth(MSG_WARNING, format_string("Sigma is out of range! Please enter a value between %.2f and %.2f.", sigma_min, sigma_max));
                }
                else
                {
                    return sigma;
                }
            }
            catch (const std::invalid_argument &e)
            {
                print_sth(MSG_ERROR, "Invalid input! Please enter a valid number.");
            }
            catch (const std::out_of_range &e)
            {
                print_sth(MSG_ERROR, "Input is too large! Please enter a valid number.");
            }
        }
        else
        {
            print_sth(MSG_INFO, format_string("Using default sigma: %.2f", sigma_default));
            return sigma_default;
        }
    }
}

void print_welcome()
{
    print_sth(MSG_HEADER, "OpenCV Watershed Lab Program");
    print_sth(MSG_INFO, "===================================");
}

vector<pair<int, int>> get_area_values(Mat &markers)
{
    // Returns vector of pairs: {area_value, marker_id}
    unordered_map<int, int> region_pixel_counts; // map<marker_id, pixel_count>
    for (int r = 0; r < markers.rows; r++)
    {
        for (int c = 0; c < markers.cols; c++)
        {
            int marker_id = markers.at<int>(r, c);
            // Valid marker_ids for regions are positive integers.
            // -1 is boundary, 0 might be background or unassigned.
            if (marker_id > 0)
            {
                region_pixel_counts[marker_id]++;
            }
        }
    }

    vector<pair<int, int>> result; // vector of {area_value, marker_id}
    result.reserve(region_pixel_counts.size());
    for (const auto &entry : region_pixel_counts)
    {
        int marker_id = entry.first;
        int area_value = entry.second;
        if (area_value > 0) // Ensure area is greater than 0
        {
            result.push_back({marker_id, area_value});
        }
    }
    return result;
}

void heap_sort(vector<pair<int, int>> &area_values)
{
    // sort(area_values.begin(), area_values.end(), [](pair<int, int> a, pair<int, int> b)
    //      { return a.second < b.second; });

    // Implement actual heap sort algorithm
    int n_total = area_values.size();
    if (n_total <= 1) // No need to sort if 0 or 1 elements
    {
        return;
    }

    std::function<void(int, int)> heapify_func; // Declare std::function for heapify

    // Define the heapify lambda
    heapify_func = [&](int current_heap_size, int root_idx)
    {
        int largest = root_idx;       // Initialize largest as root
        int left = 2 * root_idx + 1;  // Left child
        int right = 2 * root_idx + 2; // Right child

        // If left child is larger than current largest
        if (left < current_heap_size && area_values[left].second > area_values[largest].second)
            largest = left;

        // If right child is larger than current largest
        if (right < current_heap_size && area_values[right].second > area_values[largest].second)
            largest = right;

        // If largest is not root
        if (largest != root_idx)
        {
            swap(area_values[root_idx], area_values[largest]);

            // Recursively heapify the affected sub-tree
            heapify_func(current_heap_size, largest);
        }
    };

    // Build max heap (rearrange array)
    // Iterate from the last non-leaf node up to the root
    for (int i = n_total / 2 - 1; i >= 0; i--)
        heapify_func(n_total, i);

    // Extract elements from heap one by one
    for (int i = n_total - 1; i > 0; i--)
    {
        // Move current root (max element) to end of the unsorted portion
        swap(area_values[0], area_values[i]);

        // Call heapify on the reduced heap (size is now 'i', root is 0)
        heapify_func(i, 0);
    }
}

int binary_search(vector<pair<int, int>> &area_values, int val, bool is_lower_bound)
{
    int l = 0;
    int r = area_values.size();
    int m = 0;
    if (is_lower_bound)
    {
        // 1 3 5
        // 4
        // l r m
        // 0 3 1
        // 2 3 2
        // 2 2 2
        while (l < r)
        {
            m = (l + r) / 2;
            if (area_values[m].second == val)
            {
                return m;
            }
            else if (area_values[m].second > val)
            {
                r = m;
            }
            else
            {
                l = m + 1;
            }
        }
        // Check if l is a valid index before accessing area_values[l]
        if (l < area_values.size() && area_values[l].second >= val) // Changed > to >= for lower_bound logic
        {
            return l;
        }
        else
        {
            // It's possible no value is >= val, especially if val is > max element.
            // In a typical lower_bound, this would return area_values.size().
            // For this specific use case, returning -1 for "not found" or "out of bounds"
            print_sth(MSG_ERROR, format_string("Can't find area value greater than or equal to %d. Max value is %d.", val, area_values.back().second));
            return -1; // Or area_values.size() if following std::lower_bound convention
        }
    }
    else // is_upper_bound (finding index of first element > val, then subtract 1 for element <= val)
    {
        // This logic seems to be for finding an element strictly less than val,
        // or the largest element <= val. Let's clarify the intent for upper_bound.
        // A common use of upper_bound is to find the first element *greater* than val.
        // If we want the largest element *less than or equal to* val:
        l = 0;
        r = area_values.size() - 1; // search within valid indices
        int ans = -1;
        while (l <= r)
        {
            m = l + (r - l) / 2;
            if (area_values[m].second <= val)
            {
                ans = m;   // potential answer
                l = m + 1; // try to find a larger one
            }
            else
            {
                r = m - 1;
            }
        }
        if (ans != -1)
        {
            return ans;
        }
        else
        {
            print_sth(MSG_ERROR, format_string("Can't find area value less than or equal to %d. Min value is %d.", val, area_values.front().second));
            return -1;
        }
    }
}

// Get area range from user input
void get_area_range(int &lower_bound, int &upper_bound, int min_possible_area, int max_possible_area)
{
    print_sth(MSG_PROMPT, "Please input the lower bound for area search.");
    if (min_possible_area <= max_possible_area && min_possible_area >= 0)
    {
        print_sth(MSG_INFO, format_string("Min recorded area is %d. Example: %d", min_possible_area, min_possible_area));
    }

    while (true)
    {
        print_sth(MSG_PLAIN, "Lower bound > ", false);
        std::string input;
        std::getline(std::cin, input);
        if (!input.empty())
        {
            try
            {
                lower_bound = std::stoi(input);
                if (lower_bound < 0)
                {
                    print_sth(MSG_WARNING, "Area cannot be negative. Please enter a valid non-negative number.");
                }
                else
                {
                    break;
                }
            }
            catch (const std::invalid_argument &e)
            {
                print_sth(MSG_ERROR, "Invalid input! Please enter a valid number.");
            }
            catch (const std::out_of_range &e)
            {
                print_sth(MSG_ERROR, "Input is too large! Please enter a valid number.");
            }
        }
        else
        {
            print_sth(MSG_PROMPT, "Please input a lower bound!");
        }
    }

    print_sth(MSG_PROMPT, "Please input the upper bound for area search.");
    if (min_possible_area <= max_possible_area && max_possible_area >= 0)
    {
        print_sth(MSG_INFO, format_string("Max recorded area is %d. Example: %d", max_possible_area, max_possible_area));
    }
    while (true)
    {
        print_sth(MSG_PLAIN, "Upper bound > ", false);
        std::string input;
        std::getline(std::cin, input);
        if (!input.empty())
        {
            try
            {
                upper_bound = std::stoi(input);
                if (upper_bound < lower_bound)
                {
                    print_sth(MSG_WARNING, format_string("Upper bound (%d) cannot be less than lower bound (%d).", upper_bound, lower_bound));
                }
                else
                {
                    break;
                }
            }
            catch (const std::invalid_argument &e)
            {
                print_sth(MSG_ERROR, "Invalid input! Please enter a valid number.");
            }
            catch (const std::out_of_range &e)
            {
                print_sth(MSG_ERROR, "Input is too large! Please enter a valid number.");
            }
        }
        else
        {
            print_sth(MSG_PROMPT, "Please input an upper bound!");
        }
    }
}

#endif // WATERSHED_UTILS_H