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
Mat get_image(string default_path)
{
    print_current_dir();

    Mat img0;
    print_sth(MSG_PROMPT, "Please input image name from 'image/' folder (e.g., fruits.jpg)");
    print_sth(MSG_PROMPT, "Press enter to use default image: " + default_path.substr(default_path.find_last_of("/\\") + 1));
    while (true)
    {
        string filepath = "../image/";
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
double get_temperature(double t_min, double t_max, double t_default)
{
    double t;
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

// Verify minimum distance between seed points
bool verifyMinimumDistance(const vector<Point> &seeds, double minDist)
{
    bool flag = true;
    int max_violation_print = 20;
    int violation_cnt = 0;
    double max_violation_diff = 0; // Renamed from max_violation to avoid conflict
    for (size_t i = 0; i < seeds.size(); i++)
    {
        for (size_t j = i + 1; j < seeds.size(); j++)
        {
            double dist = norm(seeds[i] - seeds[j]);
            if (dist < minDist)
            {
                violation_cnt++;
                max_violation_diff = max(max_violation_diff, minDist - dist);
#ifdef DEBUG
                if (violation_cnt < max_violation_print)
                {
                    print_sth(MSG_DEBUG, format_string("Distance violation between seeds %zu and %zu: %.2f < %.2f",
                                                       i + 1, j + 1, dist, minDist));
                }
#endif
                flag = false;
            }
        }
    }
    print_sth(MSG_INFO, format_string("Distance violation ratio: %d/%zu seeds.", violation_cnt, seeds.size()));
    print_sth(MSG_INFO, format_string("Max distance difference for violations: %.2f pixels.", max_violation_diff));

    return flag;
}

// Debug log for marker analysis
void markersDebugLog(const Mat &markers)
{
#ifdef DEBUG
    // Print the number of contours found
    vector<vector<Point>> contours;
    Mat marker_mask_copy;
    markers.convertTo(marker_mask_copy, CV_8UC1);
    findContours(marker_mask_copy, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    print_sth(MSG_DEBUG, format_string("Number of contours found in markers: %zu", contours.size()));

    // Find min/max values in markers before watershed
    double minVal, maxVal;
    minMaxLoc(markers, &minVal, &maxVal);
    print_sth(MSG_DEBUG, format_string("Markers minVal: %.0f, maxVal: %.0f", minVal, maxVal));
#endif
}

// Debug log for seed sample analysis
void sampleDebugLog(const Mat &marker_mask, vector<Point> seeds, double minDist)
{
#ifdef DEBUG
    print_sth(MSG_DEBUG, "First 10 seed points:");
    for (int i = 0; i < min(10, (int)seeds.size()); i++)
    {
        print_sth(MSG_DEBUG, format_string("Seed %d: (%d, %d)", i + 1, seeds[i].x, seeds[i].y));
    }
#endif
}

// Calculate distance between two points
double calculateDistance(const Point &p1, const Point &p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p2.y - p2.y, 2));
}

// Visualize seeds on the image
void visualize_points(string window_title, const Mat &img, const vector<Point> &points,
                      int max_numbered_points = 50,
                      const Scalar &point_color = Scalar(0, 255, 255),
                      int radius = 3,
                      bool show_numbers = true)
{
    Mat display = img.clone();

    // Draw all points
    for (int i = 0; i < points.size(); i++)
    {
        // Draw a visible circle at each point
        circle(display, points[i], radius, point_color, FILLED);
        circle(display, points[i], radius, Scalar(0, 0, 0), 1); // Black outline for contrast

        // Only show numbers if requested and if there aren't too many points
        if (show_numbers && points.size() <= max_numbered_points)
        {
            // Place the point number next to the point
            Point textPos(points[i].x + 5, points[i].y + 5);

            putText(display, to_string(i + 1), textPos, FONT_HERSHEY_SIMPLEX,
                    0.4, Scalar(0, 0, 0), 2, LINE_AA); // Outlined text (thicker)
            putText(display, to_string(i + 1), textPos, FONT_HERSHEY_SIMPLEX,
                    0.4, Scalar(255, 255, 255), 1, LINE_AA); // White text
        }
    }

    // Update the displayed image
    imshow(window_title, display);

    // Print summary information
    print_sth(MSG_INFO, format_string("Visualized %zu seed points in window '%s'", points.size(), window_title.c_str()));
}

void visualize_regions(string window_title, const Mat &img, const vector<Point> &points, cv::Mat markers,
                       int max_numbered_points = 100,
                       const Scalar &point_color = Scalar(0, 255, 255),
                       int radius = 3,
                       bool show_numbers = true)
{
    Mat display = img.clone();

    // Draw all points
    for (int i = 0; i < points.size(); i++)
    {
        // Draw a visible circle at each point
        circle(display, points[i], radius, point_color, FILLED);
        circle(display, points[i], radius, Scalar(0, 0, 0), 1); // Black outline for contrast

        // Only show numbers if requested and if there aren't too many points
        if (show_numbers && points.size() <= max_numbered_points)
        {
            Point seed_location = points[i];
            Point textPos(seed_location.x + 5, seed_location.y + 5);

            int region_id_at_seed = -99; // Default value if lookup fails or seed is out of bounds

            // Ensure seed coordinates are within the bounds of the 'markers' matrix
            // and that markers matrix is valid before attempting to access its elements.
            if (!markers.empty() && markers.type() == CV_32SC1 &&
                seed_location.y >= 0 && seed_location.y < markers.rows &&
                seed_location.x >= 0 && seed_location.x < markers.cols)
            {
                // Correct access: markers.at<int>(row, col) which is markers.at<int>(y, x)
                region_id_at_seed = markers.at<int>(seed_location.y, seed_location.x);
            }

            putText(display, to_string(region_id_at_seed), textPos, FONT_HERSHEY_SIMPLEX,
                    0.4, Scalar(0, 0, 0), 2, LINE_AA); // Outlined text (thicker)
            putText(display, to_string(region_id_at_seed), textPos, FONT_HERSHEY_SIMPLEX,
                    0.4, Scalar(255, 255, 255), 1, LINE_AA); // White text
        }
    }

    // Update the displayed image
    imshow(window_title, display);

    // Print summary information
    print_sth(MSG_INFO, format_string("Visualized %zu regions with seed points in window '%s'", points.size(), window_title.c_str()));
}

// bool try_adjust_seeds(Point &seed, vector<Point> &violation_seeds, double min_dist, const Mat &marker_mask)
// {
//     // TODO verity funcionality
//     const int MAX_ATTEMPTS = 30;
//     const int NUM_DIRECTIONS = 16;
//     const double MAX_RADIUS_MULTIPLIER = 2.0;

//     int img_height = marker_mask.rows;
//     int img_width = marker_mask.cols;

//     Point original_seed = seed;
//     double current_min_dist = DBL_MAX;

//     // Compute the current minimum distance to violating neighbors
//     for (const Point &violator : violation_seeds)
//     {
//         double dist = norm(seed - violator);
//         current_min_dist = min(current_min_dist, dist);
//     }

//     //--- 1) Force-directed approach
//     for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
//     {
//         Point2f force_vector(0, 0);
//         bool all_satisfied = true;

//         // Calculate net repulsion force from violating neighbors
//         for (const Point &violator : violation_seeds)
//         {
//             double dist = norm(seed - violator);
//             if (dist < min_dist)
//             {
//                 all_satisfied = false;
//                 Point2f direction(seed.x - violator.x, seed.y - violator.y);
//                 double mag = norm(direction);
//                 if (mag > 0)
//                 {
//                     direction.x /= mag;
//                     direction.y /= mag;
//                     double force_strength = (min_dist - dist) / min_dist;
//                     force_vector.x += direction.x * force_strength;
//                     force_vector.y += direction.y * force_strength;
//                 }
//             }
//         }

//         if (all_satisfied)
//         {
//             // Already satisfied all constraints, done
//             print_sth(MSG_DEBUG, format_string("Seed adjusted successfully via force-directed approach after %d attempts.", attempt));
//             return true;
//         }

//         if (norm(force_vector) < 1e-6)
//             break; // No meaningful force to move the seed

//         // Normalize force
//         double fmag = norm(force_vector);
//         force_vector.x /= fmag;
//         force_vector.y /= fmag;

//         // Step size increases each attempt
//         double step_size = min_dist * 0.2 * (1 + attempt * 0.1);

//         // Try the new position
//         Point new_seed(
//             cvRound(seed.x + force_vector.x * step_size),
//             cvRound(seed.y + force_vector.y * step_size));

//         // Keep within image
//         new_seed.x = max(0, min(img_width - 1, new_seed.x));
//         new_seed.y = max(0, min(img_height - 1, new_seed.y));

//         // Check if new position improves the minimum distance
//         double new_min_dist = DBL_MAX;
//         for (const Point &violator : violation_seeds)
//         {
//             double dist = norm(new_seed - violator);
//             new_min_dist = min(new_min_dist, dist);
//         }

//         // If it improves (or at least doesn't worsen), update seed
//         if (new_min_dist > current_min_dist)
//         {
//             seed = new_seed;
//             current_min_dist = new_min_dist;
//         }
//         else
//         {
//             print_sth(MSG_DEBUG, "Force-directed adjustment did not improve minimum distance, stopping force attempts.");
//             // Not an improvement, stop force attempts
//             break;
//         }
//     }

//     //--- 2) Systematic search in multiple directions if force-based failed
//     Point best_seed = seed;
//     double best_min_violation_dist = current_min_dist;

//     for (int i = 0; i < NUM_DIRECTIONS; i++)
//     {
//         double angle = 2.0 * CV_PI * i / NUM_DIRECTIONS;
//         for (double radius_mult = 0.5; radius_mult <= MAX_RADIUS_MULTIPLIER; radius_mult += 0.25)
//         {
//             double radius = min_dist * radius_mult;
//             Point test_seed(
//                 cvRound(original_seed.x + cos(angle) * radius),
//                 cvRound(original_seed.y + sin(angle) * radius));

//             // Stay in bounds
//             test_seed.x = max(0, min(img_width - 1, test_seed.x));
//             test_seed.y = max(0, min(img_height - 1, test_seed.y));

//             // Check minimum distance to violating neighbors
//             double local_min_dist = DBL_MAX;
//             bool satisfies_all = true;
//             for (const Point &violator : violation_seeds)
//             {
//                 double dist = norm(test_seed - violator);
//                 local_min_dist = min(local_min_dist, dist);
//                 if (dist < min_dist)
//                     satisfies_all = false;
//             }

//             // If fully satisfied, update and return
//             if (satisfies_all)
//             {
//                 seed = test_seed;
//                 print_sth(MSG_DEBUG, "Seed adjusted successfully via systematic search.");
//                 return true;
//             }

//             // Otherwise track best improvement
//             if (local_min_dist > best_min_violation_dist)
//             {
//                 best_min_violation_dist = local_min_dist;
//                 best_seed = test_seed;
//             }
//         }
//     }

//     // If we found a slightly better position, use that
//     if (best_min_violation_dist > current_min_dist)
//     {
//         seed = best_seed;
//         print_sth(MSG_WARNING, format_string("Could not fully satisfy constraints, but improved min distance from %.2f to %.2f (required: %.2f)",
//                                              current_min_dist, best_min_violation_dist, min_dist));
//         return true;
//     }

//     // Failed to improve enough
//     print_sth(MSG_WARNING, format_string("Failed to adjust seed at (%d, %d) with min_dist=%.2f. Closest violation distance remained: %.2f",
//                                          original_seed.x, original_seed.y, min_dist, current_min_dist));
//     return false;
// }

// helper for logs -------------------------------------------------------------
void log_msg(int level, const std::string &s); // <-- your print_sth wrapper

// -----------------------------------------------------------------------------
// Returns k points, each (x,y) an *integer* pixel, pair-wise distance >
//            d_req = sqrt(M*N / k)
//
// Throws std::runtime_error only if k is impossible with that bound
// (e.g. k > M*N or the image is only a few pixels wide).
// -----------------------------------------------------------------------------
std::vector<cv::Point>
jittered_hex_grid_sample(const cv::Mat &img,
                         int k,
                         double temperature = 1.0,
                         double sigma = 1.0 /*sigma  unused*/,
                         bool /*zoomToEdge*/ = true)
{
    using cv::Point2d; // sub-pixel helper
    const int M = img.rows, N = img.cols;
    if (k <= 0)
        return {};

    const double area = static_cast<double>(M) * N;
    const double d_req = std::sqrt(area / k); // target min distance
    const double SAFETY = std::sqrt(2.0);     // loss when rounding → int

    // equal-area hex: side s0  ⇒  centre spacing d0 = 1.07392 * d_req
    auto side_from_area = [](double a)
    { return std::sqrt(2 * a / (3 * std::sqrt(3.0))); };

    double s0 = side_from_area(area / k); // initial hex side
    double shrink = 1.0;                  // we may tighten lattice
    std::vector<Point2d> candidates;
    cv::RNG rng(static_cast<uint64_t>(std::random_device{}())); // Fixed BUG: Use random_device for seed

    // -------------- adaptive lattice until we can fit >= k --------------------
    while (true)
    {
        const double s = s0 * shrink;
        const double d_cc = std::sqrt(3.0) * s; // centre-to-centre
        if (d_cc <= d_req + SAFETY + 1e-6)      // cannot shrink more
            break;

        // jitter radius keeps safety margin intact
        double r_jit = 0.5 * (d_cc - (d_req + SAFETY));
        const double r_in = 0.5 * std::sqrt(3.0) * s; // in-radius
        r_jit = std::min(r_jit, 0.95 * r_in);

        const double pad = r_jit;             // grid overscan
        const double dx = std::sqrt(3.0) * s; // lattice step x
        const double dy = 1.5 * s;            // lattice step y

        candidates.clear();
        int int_row_idx = 0;                                       // Integer row index for staggering
        for (double y = -pad; y < M + pad; y += dy, ++int_row_idx) // y is the y-center
        {
            // Fixed BUG: Use integer row index for robust staggering logic
            const double x0 = (int_row_idx & 1 ? 0.5 * dx : 0.0) - pad;
            for (double x = x0; x < N + pad; x += dx) // x is the x-center
            {
                const double rho = rng.uniform(0.0, r_jit);
                const double theta = rng.uniform(0.0, 2 * CV_PI);
                const double px = x + rho * std::cos(theta);
                const double py = y + rho * std::sin(theta);
                if (0 <= px && px < N && 0 <= py && py < M)
                    candidates.emplace_back(px, py);
            }
        }
        if (static_cast<int>(candidates.size()) >= k)
            break;
        shrink *= 0.98; // 2 % denser and try again
    }

    if (static_cast<int>(candidates.size()) < k)
    {
        print_sth(MSG_WARNING, format_string("Could only place %zu points with requested spacing (requested %d)", candidates.size(), k));
        print_sth(MSG_PROMPT, "Proceed with fewer points? (y/n)");

        while (true)
        {
            print_sth(MSG_PLAIN, "> ", false);
            std::string user_choice;
            std::getline(std::cin, user_choice);

            if (user_choice == "y" || user_choice == "Y")
            {
                print_sth(MSG_INFO, format_string("Proceeding with %zu points", candidates.size()));
                k = static_cast<int>(candidates.size());
                break;
            }
            else if (user_choice == "n" || user_choice == "N")
            {
                throw std::runtime_error("User aborted: Could not place requested number of points");
            }
            else
            {
                print_sth(MSG_WARNING, "Please enter 'y' to proceed or 'n' to abort");
            }
        }
    }

    // -------------- choose exactly k of them ----------------------------------
    std::shuffle(candidates.begin(), candidates.end(),
                 std::mt19937_64{std::random_device{}()}); // Fixed BUG: Use random_device for seed
    candidates.resize(k);

    // -------------- convert to int pixels (duplicates cannot happen) ----------
    std::vector<cv::Point> out;
    out.reserve(k);
    for (const auto &p : candidates)
        out.emplace_back(static_cast<int>(std::round(p.x)),
                         static_cast<int>(std::round(p.y)));

#ifdef _DEBUG
    // verify guarantee in float domain
    for (int i = 0; i < k; ++i)
        for (int j = i + 1; j < k; ++j)
            if (cv::norm(candidates[i] - candidates[j]) <= d_req - 1e-6)
                throw std::logic_error("distance guarantee violated – should never happen");
#endif

    print_sth(MSG_INFO, format_string("Hex sampler: generated %d points, min-distance %.3f px", k, d_req));
    return out;
}

vector<Point> jittered_grid_sample(const Mat &marker_mask, int k, double temperature, bool zoomToEdge = true)
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

    if (zoomToEdge)
    {
        // zoom
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

                print_sth(MSG_INFO, format_string("Applied scaling factor of %.2f to better distribute points.", scale));
            }
        }
    }
    // Print information about generated points
    print_sth(MSG_INFO, format_string("Generated %zu seed points using jittered grid sampling.", seeds.size()));
    print_sth(MSG_INFO, format_string("Target minimum distance threshold: %.2f pixels.", min_dist));
    print_sth(MSG_INFO, format_string("Grid dimensions: %d rows x %d columns.", grid_rows, grid_cols));

    bool distanceConstraintMet = verifyMinimumDistance(seeds, min_dist);
    if (distanceConstraintMet)
    {
        print_sth(MSG_SUCCESS, "All seeds satisfy the minimum distance constraint.");
    }
    else
    {
        print_sth(MSG_WARNING, "Some seeds do not satisfy the minimum distance constraint.");
    }

    sampleDebugLog(marker_mask, seeds, min_dist);

    return seeds;
}

vector<Point> cyj_generateSeeds(int K, int rows, int cols)
{
    vector<Point> seeds;
    double minDistance = sqrt((rows * cols) / K);
    int cellSize = static_cast<int>(minDistance / sqrt(2));

    // 初始化网格
    int gridCols = (cols + cellSize - 1) / cellSize;
    int gridRows = (rows + cellSize - 1) / cellSize;
    vector<vector<Point>> grid(gridRows, vector<Point>(gridCols, Point(-1, -1)));

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> disX(0, cols - 1);
    uniform_int_distribution<> disY(0, rows - 1);

    // 从边缘开始生成初始点
    vector<Point> edgePoints;
    for (int x = 0; x < cols; ++x)
    {
        edgePoints.push_back(Point(x, 0));
        edgePoints.push_back(Point(x, rows - 1));
    }
    for (int y = 1; y < rows - 1; ++y)
    {
        edgePoints.push_back(Point(0, y));
        edgePoints.push_back(Point(cols - 1, y));
    }

    shuffle(edgePoints.begin(), edgePoints.end(), gen);

    // 从边缘点中选择初始点
    for (const Point &edgePoint : edgePoints)
    {
        if (seeds.size() >= K)
            break;

        bool valid = true;
        int gridX = edgePoint.x / cellSize;
        int gridY = edgePoint.y / cellSize;

        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                int x = gridX + dx;
                int y = gridY + dy;
                if (x >= 0 && x < gridCols && y >= 0 && y < gridRows)
                {
                    Point neighbor = grid[y][x];
                    if (neighbor.x != -1 && calculateDistance(neighbor, edgePoint) < minDistance)
                    {
                        valid = false;
                        break;
                    }
                }
            }
            if (!valid)
                break;
        }

        if (valid)
        {
            seeds.push_back(edgePoint);
            grid[edgePoint.y / cellSize][edgePoint.x / cellSize] = edgePoint;
        }
    }

    // 使用队列进行采样
    queue<Point> activeList;
    for (const Point &seed : seeds)
    {
        activeList.push(seed);
    }

    // 记录最近生成的10个点
    vector<Point> recentPoints;
    const int recentCount = 5;

    while (!activeList.empty() && seeds.size() < K)
    {
        Point current = activeList.front();
        activeList.pop();

        // 进行20次尝试，每次生成100个候选点
        for (int attempt = 0; attempt < 20 && seeds.size() < K; ++attempt)
        {
            vector<Point> candidates;
            vector<double> minDistances;

            // 生成100个候选点
            for (int i = 0; i < 100; ++i)
            {
                double angle = 2 * M_PI * (gen() / (double)gen.max());
                double radius = minDistance * (1 + 0.0 * (gen() / (double)gen.max()));
                int newX = current.x + radius * cos(angle);
                int newY = current.y + radius * sin(angle);

                if (newX < 0 || newX >= cols || newY < 0 || newY >= rows)
                    continue;

                Point candidate(newX, newY);

                // 检查候选点是否满足距离条件
                bool valid = true;
                int gridX = newX / cellSize;
                int gridY = newY / cellSize;

                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        int x = gridX + dx;
                        int y = gridY + dy;
                        if (x >= 0 && x < gridCols && y >= 0 && y < gridRows)
                        {
                            Point neighbor = grid[y][x];
                            if (neighbor.x != -1 && calculateDistance(neighbor, candidate) < minDistance)
                            {
                                valid = false;
                                break;
                            }
                        }
                    }
                    if (!valid)
                        break;
                }

                if (valid)
                {
                    // 计算候选点到最近recentPoints的最小距离
                    double minDist = numeric_limits<double>::max();
                    for (const Point &p : recentPoints)
                    {
                        double dist = calculateDistance(candidate, p);
                        if (dist < minDist)
                            minDist = dist;
                    }
                    candidates.push_back(candidate);
                    minDistances.push_back(minDist);
                }
            }

            // 如果有候选点，选择距离最近recentPoints最小的点
            if (!candidates.empty())
            {
                auto minIt = min_element(minDistances.begin(), minDistances.end());
                int bestIndex = distance(minDistances.begin(), minIt);
                Point bestCandidate = candidates[bestIndex];

                // 添加最佳候选点到种子列表
                seeds.push_back(bestCandidate);
                grid[bestCandidate.y / cellSize][bestCandidate.x / cellSize] = bestCandidate;
                activeList.push(bestCandidate);

                // 更新最近点列表
                recentPoints.push_back(bestCandidate);
                if (recentPoints.size() > recentCount)
                {
                    recentPoints.erase(recentPoints.begin());
                }
            }
        }
    }

    if (seeds.size() < K)
    {
        // cerr << "Warning: Could not generate " << K << " seeds. Generated " << seeds.size() << " seeds instead." << endl;
        print_sth(MSG_WARNING, format_string("Could not generate %d seeds. Generated %zu seeds instead.", K, seeds.size()));
    }

    return seeds;
}

vector<Point> generate_seeds(const Mat &img, Mat &marker_mask, int k, double temperature, double sigma)
{
    double t = (double)getTickCount();
    print_sth(MSG_INFO, "Generating seed points...");

    // Use 8-bit mask for visualization and 32-bit for watershed
    marker_mask = Mat::zeros(img.size(), CV_8UC1);

    // generate k random seed points in marker_mask
    vector<Point> seeds = jittered_hex_grid_sample(marker_mask, k, temperature, sigma, true);

    // Draw smaller circles for markers to avoid overlapping
    // But use distinct values for each region
    for (int i = 0; i < seeds.size(); i++)
    {
        // Use modulo to handle values > 255
        circle(marker_mask, seeds[i], 5, Scalar((i % 254) + 1), FILLED);
    }

    t = (double)getTickCount() - t;
    print_sth(MSG_SUCCESS, format_string("Seed generation time cost = %.2f ms", t / getTickFrequency() * 1000.));

    return seeds;
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