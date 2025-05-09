#ifndef SAMPLE_H
#define SAMPLE_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "watershed_utils.h"

using namespace cv;
using namespace std;

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
    print_sth(MSG_INFO, format_string("Distance violation cnt: %d", violation_cnt));
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

std::vector<cv::Point>
jittered_hex_grid_sample(const cv::Mat &marker_mask,
                         int k,
                         double temperature = 1.0)
{
    using cv::Point2d; // sub-pixel helper
    const int M = marker_mask.rows, N = marker_mask.cols;
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

    sampleDebugLog(marker_mask, out, d_req);

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

vector<Point> backup_generateSeeds(int K, int rows, int cols)
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

vector<Point> generate_seeds(const Mat &img, Mat &marker_mask, int k, double temperature)
{
    // TODO ensure at least k seeds
    double t = (double)getTickCount();
    print_sth(MSG_INFO, "Generating seed points...");

    // Use 8-bit mask for visualization and 32-bit for watershed
    marker_mask = Mat::zeros(img.size(), CV_8UC1);

    // generate k random seed points in marker_mask
    // vector<Point> seeds = jittered_grid_sample(marker_mask, k, temperature, true);
    vector<Point> seeds = jittered_hex_grid_sample(marker_mask, k, temperature);
    // vector<Point> seeds = backup_generateSeeds(k, img.rows, img.cols);

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

#endif // SAMPLE_H