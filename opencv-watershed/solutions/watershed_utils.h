#ifndef WATERSHED_UTILS_H
#define WATERSHED_UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <random>
#ifdef _WIN32
#include <direct.h>
#define getcwd _getcwd
#else
#include <unistd.h>
#endif
#include <climits> // For PATH_MAX

using namespace cv;
using namespace std;

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
    EXIT
};

enum FourColor
{
    RED,
    YELLOW,
    GREEN,
    BLUE
};

// enum task2_next_step
// {
//     INPUT_IMAGE,
//     INPUT_K,
//     INPUT_TEMP,
//     INPUT_SIGMA,
//     GENERATE_SEEDS,
//     WATERSHED,
//     FOURCOLOR,
//     EXIT
// };

// Print help information
void print_task1_help()
{
    // Print instructions
    printf("Hot keys: \n"
           "\tESC or q - quit the program\n"
           "\tr - restore the original image\n"
           "\tg - generate seeds\n"
           "\tv - visualize generated seeds\n"
           "\tc - clear input and restart\n"
           "\tw - run watershed\n");
}

void print_task2_help()
{
    // Print instructions
    printf("Hot keys: \n"
           "\tESC or q - quit the program\n"
           "\tr - restore the original image\n"
           "\tg - generate seeds\n"
           "\tv - visualize generated seeds\n"
           //    "\tc - clear input and restart\n"
           "\tw - run watershed\n"
           "\tc - perform four color\n");
}

// Print current directory for debugging
void print_current_dir()
{
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
    {
        cout << "Current working directory: " << cwd << endl;
    }
    else
    {
        perror("getcwd() error");
    }
}

// Get image from user input or use default
Mat get_image(string default_path)
{
    print_current_dir();

    Mat img0;
    cout << "please input image in image/ folder" << endl;
    cout << "press enter to use default image fruits.jpg" << endl;
    while (true)
    {
        string filepath = "../image/";
        cout << "> ";
        string input;
        getline(cin, input);
        if (!input.empty())
        {
            filepath += input;
            img0 = imread(filepath, 1);
            if (img0.empty())
            {
                cout << "please input a valid path!" << endl;
                continue;
            }
            else
            {
                return img0;
            }
        }
        else
        {
            img0 = imread(default_path, 1);
            return img0;
        }
    }
}

// Get k (number of seeds) from user input
int get_k(int k_min, int k_max)
{
    int k = 0;
    cout << "please input k, desired number of random seed points" << endl;
    cout << "for example, 100, 500, 1000" << endl;
    while (true)
    {
        cout << "> ";
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

// Get temperature parameter from user input
double get_temperature(double t_min, double t_max, double t_default)
{
    double t;
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

// Get sigma parameter from user input
double get_sigma(double sigma_min, double sigma_max, double sigma_default)
{
    double sigma;
    cout << "please input sigma, press enter to use default " << sigma_default << endl;
    while (true)
    {
        cout << "> ";
        string input;
        getline(cin, input);
        if (!input.empty())
        {
            try
            {
                sigma = stof(input);
                if (sigma < sigma_min || sigma > sigma_max)
                {
                    cout << "sigma out of range!" << endl;
                }
                else
                {
                    return sigma;
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
    double max_violation = 0;
    for (int i = 0; i < seeds.size(); i++)
    {
        for (int j = i + 1; j < seeds.size(); j++)
        {
            double dist = norm(seeds[i] - seeds[j]);
            if (dist < minDist)
            {
                violation_cnt++;
                max_violation = max(max_violation, minDist - dist);
#ifdef DEBUG
                if (violation_cnt < max_violation_print)
                {
                    printf("Distance violation between seeds %d and %d: %.2f < %.2f\n",
                           i + 1, j + 1, dist, minDist);
                }
#endif
                flag = false;
            }
        }
    }
    cout << "violation ratio: " << violation_cnt << "/" << seeds.size() << endl;
    cout << "max diff: " << max_violation << endl;

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
    cout << "Number of contours found: " << contours.size() << endl;

    // Find min/max values in markers before watershed
    double minVal, maxVal;
    minMaxLoc(markers, &minVal, &maxVal);
    cout << "Markers minVal: " << minVal << ", maxVal: " << maxVal << endl;
#endif
}

// Debug log for seed sample analysis
void sampleDebugLog(const Mat &marker_mask, vector<Point> seeds, double minDist)
{
#ifdef DEBUG
    // Print the first 10 seeds for debugging
    printf("First 10 seed points:\n");
    for (int i = 0; i < min(10, (int)seeds.size()); i++)
    {
        printf("Seed %d: (%d, %d)\n", i + 1, seeds[i].x, seeds[i].y);
    }
#endif
}

// Calculate distance between two points
double calculateDistance(const Point &p1, const Point &p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
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
    printf("Visualized %zu seed points in %s\n", points.size(), window_title.c_str());
}
bool try_adjust_seeds(Point &seed, vector<Point> &violation_seeds, double min_dist, const Mat &marker_mask)
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

vector<Point> jittered_hex_grid_sample(const Mat &marker_mask, int k, double temperature, double sigma, bool zoomToEdge = true)
{
    int M = marker_mask.rows;                    // Image height
    int N = marker_mask.cols;                    // Image width
    double min_dist = sqrt((double)(M * N) / k); // Minimum distance required
    // double scaled_min_dist = sigma * min_dist;   // Scaled minimum distance
    double scaled_min_dist = ceil(min_dist); // Scaled minimum distance

    double spacing = scaled_min_dist;
    double h_spacing = spacing;
    double v_spacing = spacing * sqrt(3) / 2;

    RNG rng(getTickCount());

    vector<Point> seeds;
    vector<Point> neighbor_seeds;
    for (double y = 0; y < M && seeds.size() < k; y += v_spacing)
    {
        bool odd_row = (int)(y / v_spacing) % 2 == 1;
        double row_offset = odd_row ? h_spacing / 2 : 0;

        for (double x = row_offset; x < N && seeds.size() < k; x += h_spacing)
        {
            double final_x = x;
            double final_y = y;

            if (temperature > 0)
            {
                double max_jitter = spacing * 0.29 * temperature;
                final_x += rng.uniform(-max_jitter, max_jitter);
                final_y += rng.uniform(-max_jitter, max_jitter);
            }

            // Ensure point is within image boundaries
            if (final_x >= 0 && final_x < N && final_y >= 0 && final_y < M)
            {
                // Convert to int for storage
                seeds.push_back(Point(round(final_x), round(final_y)));
            }
        }
    }

    if (zoomToEdge)
    {
        if (seeds.size() > 0)
        {
            int min_x = N, min_y = M, max_x = 0, max_y = 0;
            for (const Point &p : seeds)
            {
                min_x = min(min_x, p.x);
                min_y = min(min_y, p.y);
                max_x = max(max_x, p.x);
                max_y = max(max_y, p.y);
            }

            for (Point &p : seeds)
            {
                double mapped_x = ((double)(p.x - min_x) / (max_x - min_x)) * (N - 1);
                double mapped_y = ((double)(p.y - min_y) / (max_y - min_y)) * (M - 1);

                p.x = round(mapped_x);
                p.y = round(mapped_y);
            }
        }
    }

    // Print information about generated points
    printf("Generated %zu seed points using jittered hex grid sampling\n", seeds.size());
    printf("Minimum distance threshold: %.2f pixels\n", min_dist);
    printf("Scaled distance threshold: %.2f pixels\n", scaled_min_dist);

    bool distanceConstraintMet = verifyMinimumDistance(seeds, min_dist);
    if (distanceConstraintMet)
    {
        printf("All seeds satisfy the minimum distance constraint\n");
    }
    else
    {
        printf("Warning: Some seeds do not satisfy the minimum distance constraint\n");
    }
    sampleDebugLog(marker_mask, seeds, min_dist);

#ifdef DEBUG
    cout << "Do you want to try adjusting seeds to satisfy min distance requirement? [y/n]" << endl;
    cout << "press enter to skip" << endl;
    cout << ">";
    string input;
    getline(cin, input);
    if (!input.empty())
    {
        if (input == "y")
        {
            // Adjust seeds based on hex grid structure
            bool adjusted = false;
            int iteration_count = 0;
            int max_iterations = 5;

            cout << "Adjusting seeds to satisfy minimum distance requirements..." << endl;

            // Map to track which seeds have been processed
            vector<bool> processed(seeds.size(), false);

            // Adjust until all seeds meet the constraint or max iterations reached
            while (iteration_count < max_iterations)
            {
                adjusted = false;

                // Create a data structure to quickly find neighboring seeds
                vector<vector<int>> neighbors(seeds.size());

                // Find potential neighbors for each seed
                for (int i = 0; i < seeds.size(); i++)
                {
                    for (int j = i + 1; j < seeds.size(); j++) // TODO replace enumeration of neighboring seeds with better approach
                    {
                        double dist = norm(seeds[i] - seeds[j]);
                        // Use a slightly larger threshold to include potential neighbors
                        if (dist < min_dist * 1.5)
                        {
                            neighbors[i].push_back(j);
                            neighbors[j].push_back(i);
                        }
                    }
                }

                // Process each seed and its neighbors
                for (int i = 0; i < seeds.size(); i++)
                {
                    if (processed[i])
                        continue;

                    vector<Point> violating_neighbors;
                    for (int j : neighbors[i])
                    {
                        double dist = norm(seeds[i] - seeds[j]);
                        if (dist < min_dist)
                        {
                            violating_neighbors.push_back(seeds[j]);
                        }
                    }

                    // If there are violations, try to adjust the seed
                    if (!violating_neighbors.empty())
                    {
                        bool success = try_adjust_seeds(seeds[i], violating_neighbors, min_dist, marker_mask);
                        if (success)
                        {
                            adjusted = true;
                            processed[i] = true;
                        }
                    }
                    else
                    {
                        processed[i] = true; // No violations, mark as processed
                    }
                }

                // Check if we've made any adjustments in this iteration
                if (!adjusted)
                {
                    cout << "No further adjustments possible after " << iteration_count + 1 << " iterations." << endl;
                    break;
                }

                iteration_count++;

                // Check if we now satisfy the distance constraint
                bool currentDistanceConstraintMet = verifyMinimumDistance(seeds, min_dist);
                if (currentDistanceConstraintMet)
                {
                    cout << "All distance constraints satisfied after " << iteration_count << " iterations!" << endl;
                    break;
                }
            }

            // Final check
            bool distanceConstraintMet = verifyMinimumDistance(seeds, min_dist);
            if (distanceConstraintMet)
            {
                printf("Now all seeds satisfy the minimum distance constraint\n");
            }
            else
            {
                printf("Warning: Some seeds still do not satisfy the minimum distance constraint\n");
            }
            sampleDebugLog(marker_mask, seeds, min_dist);
        }
        else if (input == "n")
        {
            ;
        }
        else
        {
            cout << "invalid input, skip adjusting" << endl;
        }
    }
#endif

    return seeds;
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

                printf("Applied scaling factor of %.2f to better distribute points\n", scale);
            }
        }
    }
    // Print information about generated points
    printf("Generated %zu seed points using jittered grid sampling\n", seeds.size());
    printf("Minimum distance threshold: %.2f pixels\n", min_dist);
    printf("Grid dimensions: %d rows x %d columns\n", grid_rows, grid_cols);

    bool distanceConstraintMet = verifyMinimumDistance(seeds, min_dist);
    if (distanceConstraintMet)
    {
        printf("Now all seeds satisfy the minimum distance constraint\n");
    }
    else
    {
        printf("Warning: Some seeds still do not satisfy the minimum distance constraint\n");
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
        cerr << "Warning: Could not generate " << K << " seeds. Generated " << seeds.size() << " seeds instead." << endl;
    }

    return seeds;
}

vector<Point> generate_seeds(const Mat &img, Mat &marker_mask, int k, double temperature, double sigma)
{
    double t = (double)getTickCount();

    // Use 8-bit mask for visualization and 32-bit for watershed
    marker_mask = Mat::zeros(img.size(), CV_8UC1);

    // generate k random seed points in marker_mask
    vector<Point> seeds = jittered_grid_sample(marker_mask, k, temperature);
    // vector<Point> seeds = jittered_hex_grid_sample(marker_mask, k, temperature, sigma, true);

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

#endif // WATERSHED_UTILS_H