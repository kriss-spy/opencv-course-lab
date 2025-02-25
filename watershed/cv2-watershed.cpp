#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat marker_mask, markers, img0, img, img_gray, wshed;
Point prev_pt(-1, -1);

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

int main(int argc, char **argv)
{
    const char *filename = argc >= 2 ? argv[1] : "fruits.jpg";
    RNG rng(12345);

    img0 = imread(filename, 1);
    if (img0.empty())
        return 0;

    printf("Hot keys: \n"
           "\tESC - quit the program\n"
           "\tr - restore the original image\n"
           "\tw or ENTER - run watershed algorithm\n"
           "\t\t(before running it, roughly mark the areas on the image)\n");

    namedWindow("image", 1);
    namedWindow("watershed transform", 1);

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

    for (;;)
    {
        int c = waitKey(0);

        if (c == 27)
            break;

        if (c == 'r')
        {
            marker_mask = Scalar::all(0);
            img0.copyTo(img);
            imshow("image", img);
        }

        if (c == 'w' || c == '\r')
        {
            vector<vector<Point>> contours;
            findContours(marker_mask, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            markers = Scalar::all(0);
            int comp_count = 0;
            for (size_t i = 0; i < contours.size(); i++)
                drawContours(markers, contours, (int)i, Scalar::all(++comp_count), -1);

            vector<Vec3b> color_tab;
            for (int i = 0; i < comp_count; i++)
            {
                int b = rng.uniform(0, 255);
                int g = rng.uniform(0, 255);
                int r = rng.uniform(0, 255);
                color_tab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
            }

            double t = (double)getTickCount();
            watershed(img0, markers);
            t = (double)getTickCount() - t;
            printf("exec time = %gms\n", t / getTickFrequency() * 1000.);

            for (int i = 0; i < markers.rows; i++)
                for (int j = 0; j < markers.cols; j++)
                {
                    int idx = markers.at<int>(i, j);
                    Vec3b &dst = wshed.at<Vec3b>(i, j);
                    if (idx == -1)
                        dst = Vec3b(255, 255, 255);
                    else if (idx <= 0 || idx > comp_count)
                        dst = Vec3b(0, 0, 0);
                    else
                        dst = color_tab[idx - 1];
                }

            addWeighted(wshed, 0.5, img_gray, 0.5, 0, wshed);
            imshow("watershed transform", wshed);
        }
    }

    return 0;
}