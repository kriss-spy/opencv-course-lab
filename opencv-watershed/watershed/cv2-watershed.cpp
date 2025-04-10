// use make
// or compile manually
// g++ cv2-watershed.cpp -o cv2-watershed -I/usr/include/opencv4 -L/usr/lib \
-lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

// original watershed.cpp doesn't use cv2, but cv.h
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

// Mouse callback function to draw markers
void on_mouse(int event, int x, int y, int flags, void *param)
{
    if (img.empty())
        return;

    if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
        prev_pt = Point(-1, -1);
    else if (event == EVENT_LBUTTONDOWN)
        prev_pt = Point(x, y);
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
    {
        Point pt(x, y);
        if (prev_pt.x < 0)
            prev_pt = pt;
        line(marker_mask, prev_pt, pt, Scalar::all(255), 5, 8, 0);
        line(img, prev_pt, pt, Scalar::all(255), 5, 8, 0);
        prev_pt = pt;
        imshow("image", img);
    }
}

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

int main(int argc, char **argv)
{
    // Load image
    const char *filename = argc >= 2 ? argv[1] : "../fruits.jpg";
    RNG rng(12345);

    img0 = imread(filename, 1);
    if (img0.empty())
        return 0;

    // Print instructions
    printf("Hot keys: \n"
           "\tESC - quit the program\n"
           "\tr - restore the original image\n"
           "\tw or ENTER - run watershed algorithm\n"
           "\t\t(before running it, roughly mark the areas on the image)\n");

    // Create windows
    namedWindow("image", 1);
    namedWindow("watershed transform", 1);

    // Initialize images
    img = img0.clone();
    img_gray = img0.clone();
    wshed = img0.clone();
    marker_mask = Mat::zeros(img.size(), CV_8UC1);
    markers = Mat::zeros(img.size(), CV_32SC1);
    cvtColor(img, marker_mask, COLOR_BGR2GRAY);
    cvtColor(marker_mask, img_gray, COLOR_GRAY2BGR);

    imshow("image", img);
    imshow("watershed transform", wshed);
    setMouseCallback("image", on_mouse, 0);

    // Main loop
    for (;;)
    {
        int c = waitKey(0);

        if (c == 27)
            break;

        if (c == 'r') // Restore original image
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);
            imshow("image", img);
        }

        if (c == 'w' || c == '\r') // Run watershed
        {
            vector<vector<Point>> contours;
            findContours(marker_mask, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

            markers = Scalar::all(0);
            int comp_count = 0;
            for (size_t i = 0; i < contours.size(); i++)
            {
                drawContours(markers, contours, (int)i, Scalar::all(++comp_count), -1);
            }

            // Debug log
            markersDebugLog(markers);

            vector<Vec3b> color_tab;
            for (int i = 0; i < comp_count; i++)
            {
                int b = rng.uniform(0, 255);
                int g = rng.uniform(0, 255);
                int r = rng.uniform(0, 255);
                color_tab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
            }

            double t = (double)getTickCount();
            watershed(img0, markers); // Perform watershed
            t = (double)getTickCount() - t;
            printf("exec time = %gms\n", t / getTickFrequency() * 1000.);

            for (int i = 0; i < markers.rows; i++)
                for (int j = 0; j < markers.cols; j++)
                {
                    int idx = markers.at<int>(i, j);
                    Vec3b &dst = wshed.at<Vec3b>(i, j);
                    if (idx == -1)
                        dst = Vec3b(255, 255, 255); // Boundary
                    else if (idx <= 0 || idx > comp_count)
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

// cli demo
// % ./cv2-watershed
// Hot keys:
//         ESC - quit the program
//         r - restore the original image
//         w or ENTER - run watershed algorithm
//                 (before running it, roughly mark the areas on the image)
// QSettings::value: Empty key passed
// QSettings::value: Empty key passed
// exec time = 0.219244ms
// exec time = 0.470228ms
// exec time = 1.02048ms