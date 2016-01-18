////
////  main.cpp
////  Opencv-Test
////
////  Created by Ornicha Choungaramvong on 1/8/2559 BE.
////  Copyright © 2559 Ornicha Choungaramvong. All rights reserved.
////
//
//
//
//#include <iostream>
//#include <opencv2/opencv.hpp>
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//
//
//using namespace cv;
//using namespace std;
//
//int main(int argc, const char * argv[]) {
//    
//    
//    cv::Mat image,dst;
//    image = cv::imread("/Users/Ornicha/Desktop/test/Opencv-Test/Opencv-Test/fruits.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
//    
//    if(! image.data )                              // Check for invalid input
//    {
//        std::cout <<  "Could not open or find the image" << std::endl ;
//        return -1;
//    }
//    
//    namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
//    imshow( "Display window", image );                   // Show our image inside it.
//    
//    dst = image.clone();
//    GaussianBlur( image, dst, Size( 15, 15 ), 0, 0 );
//    
//    namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
//    imshow( "Display", dst );                   // Show our image inside it.
//    
//    cv::waitKey(0);                                          // Wait for a keystroke in the window
//    return 0;
//}


//---------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------
//
////Find contours
//
//
//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "iostream"
//using namespace cv;
//using namespace std;
//int main( )
//{
//    Mat image,dst;
//    image = imread("/Users/Ornicha/Desktop/image/pill2.jpg", 1);
//    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
//    imshow( "Display window", image );
//    
//    
//    Mat gray;
//    cvtColor(image, gray, CV_BGR2GRAY);
//    cv::namedWindow("11", cv::WINDOW_AUTOSIZE);
//    cv::imshow("11", gray);
//    
//    
//    dst = gray.clone();
// // GaussianBlur( image, gray, Size( 15, 15 ), 0, 0 );
//    
//    //Binary image
//    cv::Mat binaryMat(gray.size(), gray.type());
//    
//    //Apply thresholding
//    cv::threshold(gray, binaryMat, 0, 255, cv::THRESH_BINARY);
//    
//    cv::namedWindow("22", cv::WINDOW_AUTOSIZE);
//    cv::imshow("22", binaryMat);
//    
//    
//    cv::Mat const structure_elem = cv::getStructuringElement(
//                                                             cv::MORPH_RECT, cv::Size(5, 5));
//    cv::Mat open_result;
//    cv::morphologyEx(binaryMat, open_result,
//                     cv::MORPH_OPEN, structure_elem);
//    cv::namedWindow("33", cv::WINDOW_AUTOSIZE);
//    cv::imshow("33", open_result);
//    
//    
//    Canny(open_result, gray, 100, 200, 3);
//    /// Find contours
//    vector<vector<Point> > contours;
//    vector<Vec4i> hierarchy;
//    RNG rng(12345);
//    findContours( gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
//    
//    
//    /// Draw contours
//    Mat drawing = Mat::zeros( gray.size(), CV_8UC3 );
//    
//    for(unsigned int i=0;i<contours.size();i++)
//    {
//        cout << "# of contour points: " << contours[i].size() << endl ;
//        
//        for(unsigned int j=0;j<contours[i].size();j++)
//        {
//           // cout << "Point(x,y)=" << contours[i][j] << endl;
//        }
//        
//        cout << " Area: " << contourArea(contours[i]) << endl;
//    }
//    
//    for( int i = 0; i< contours.size(); i++ )
//    {
//        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//                 drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
//    }
//    
//    imshow( "Result window", drawing );
//    waitKey(0);
//    return 0;
//
//
//}

//---------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------
//
////Binary image//
//
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <iostream>
//
//int main()
//{
//    cv::Mat src = cv::imread("/Users/Ornicha/Desktop/image/pill2.jpg");
//    if (!src.data)
//        return -1;
//    
//    cv::imshow("src", src);
//    
//    // Create binary image from source image
//    cv::Mat bw;
//    cv::cvtColor(src, bw, CV_BGR2GRAY);
//    cv::imshow("grsy", bw);
//    cv::threshold(bw, bw, 40, 255, CV_THRESH_BINARY);
//    cv::imshow("bw", bw);
//    
//    // Perform the distance transform algorithm
//    cv::Mat dist;
//    cv::distanceTransform(bw, dist, CV_DIST_L2, 3);
//    
//    // Normalize the distance image for range = {0.0, 1.0}
//    // so we can visualize and threshold it
//    cv::normalize(dist, dist, 0, 1., cv::NORM_MINMAX);
//    cv::imshow("dist", dist);
//    
//    // Threshold to obtain the peaks
//    // This will be the markers for the foreground objects
//    cv::threshold(dist, dist, .5, 1., CV_THRESH_BINARY);
//    cv::imshow("dist2", dist);
//    
//    // Create the CV_8U version of the distance image
//    // It is needed for cv::findContours()
//    cv::Mat dist_8u;
//    dist.convertTo(dist_8u, CV_8U);
//    
//    // Find total markers
//    std::vector<std::vector<cv::Point> > contours;
//    cv::findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//    int ncomp = contours.size();
//    
//    // Create the marker image for the watershed algorithm
//    cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32SC1);
//    
//    // Draw the foreground markers
//    for (int i = 0; i < ncomp; i++)
//        cv::drawContours(markers, contours, i, cv::Scalar::all(i+1), -1);
//    
//    // Draw the background marker
//    cv::circle(markers, cv::Point(5,5), 3, CV_RGB(255,255,255), -1);
//    cv::imshow("markers", markers*10000);
//    
//    // Perform the watershed algorithm
//    cv::watershed(src, markers);
//    
//    // Generate random colors
//    std::vector<cv::Vec3b> colors;
//    for (int i = 0; i < ncomp; i++)
//    {
//        int b = cv::theRNG().uniform(0, 255);
//        int g = cv::theRNG().uniform(0, 255);
//        int r = cv::theRNG().uniform(0, 255);
//        
//        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
//    }
//    
//    // Create the result image
//    cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
//    
//    // Fill labeled objects with random colors
//    for (int i = 0; i < markers.rows; i++)
//    {
//        for (int j = 0; j < markers.cols; j++)
//        {
//            int index = markers.at<int>(i,j);
//            if (index > 0 && index <= ncomp)
//                dst.at<cv::Vec3b>(i,j) = colors[index-1];
//            else
//                dst.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
//        }
//    }
//    
//    cv::imshow("dst", dst);
//    
//    cv::waitKey(0);
//    return 0;
//}


//---------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------


//crop image//


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
using namespace cv;
using namespace std;

int main() {
    
    // read in the apple (change path to the file)
    Mat img0 = imread("/Users/MagicMagic/Desktop/OpenCV/image/yellow.jpg", 1);
    
    Mat img1;
    cvtColor(img0, img1, CV_RGB2GRAY);
    
    // apply your filter
    Canny(img1, img1, 100, 200);
    
    // find the contours
    vector< vector<Point> > contours;
    findContours(img1, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
//    cout << " Area: " << contours.size() << endl;
//
//    cout << " Area: " << contourArea(contours[0]) << endl;
    //cout << " Area: " << contourArea(contours[1]) << endl;
    
    // you could also reuse img1 here
    Mat mask = Mat::zeros(img1.rows, img1.cols, CV_8UC1);
    
    // CV_FILLED fills the connected components found
    drawContours(mask, contours, -1, Scalar(255), CV_FILLED);
    
    
    /*
     Before drawing all contours you could also decide
     to only draw the contour of the largest connected component
     found. Here's some commented out code how to do that:
     */
    
    //    vector<double> areas(contours.size());
    //    for(int i = 0; i < contours.size(); i++)
    //        areas[i] = contourArea(Mat(contours[i]));
    //    double max;
    //    Point maxPosition;
    //    minMaxLoc(Mat(areas),0,&max,0,&maxPosition);
    //    drawContours(mask, contours, maxPosition.y, Scalar(1), CV_FILLED);
    
    // let's create a new image now
    Mat crop(img0.rows, img0.cols, CV_8UC3);
    
    // set background to green
    crop.setTo(Scalar(255,255,255));
    
    // and copy the magic apple
    img0.copyTo(crop, mask);
    
    // normalize so imwrite(...)/imshow(...) shows the mask correctly!
    normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);
    
     cout << " Mask: " << mask.size() << endl;
    
//        for (int i = 0; i < mask.rows; i++)
//        {
//            for (int j = 0; j < mask.cols; j++)
//            {
//
//    
//    Vec3b rgb=img0.at<Vec3b>(i,j);
//    int B=rgb.val[0];
//    int G=rgb.val[1];
//    int R=rgb.val[2];
//    
    Mat HSV;
//
    char name[30];
    
    int i = (img0.rows/2);
    int j = (img0.cols/2);
    
    cout << " i: " << i << endl;
    cout << " j: " << j << endl;
 
    Vec3b rgb=img0.at<Vec3b>(i,j);
    int B=rgb.val[0];
    int G=rgb.val[1];
    int R=rgb.val[2];
    
    cout << " R: " << R << endl;
    cout << " G: " << G << endl;
    cout << " B: " << B << endl;
    
    Mat RGB=img0(Rect(i,j,1,1));
    cvtColor(img0, HSV,CV_BGR2HSV);
    
    Vec3b hsv=HSV.at<Vec3b>(i,j);
    int H=hsv.val[0];
    int S=hsv.val[1];
    int V=hsv.val[2];
                
    cout << " H: " << H << endl;
    cout << " S: " << S << endl;
    cout << " V: " << V << endl;
    
    if(V == 0)
    {
    cout << " Color: White" << endl;
    }
    else  if( S == 0)
    {
        cout << " Color: Black" << endl;
    }
    
   else if (H >= 0 && H <= 3)
    {
        cout << " Color: Red" << endl;
    }
    else if (H >= 4 && H <= 10)
    {
        cout << " Color: Red-Orange" << endl;
    }
    else if (H >= 11 && H <=12 )
    {
        cout << " Color: Orange" << endl;
    }
    else if (H >= 13 && H <= 20)
    {
        cout << " Color: Orange-Brown" << endl;
    }
    else if (H >= 21 && H <= 25)
    {
        cout << " Color: Orange-Yellow" << endl;
    }
    else if (H >= 26 && H <= 30)
    {
        cout << " Color: Yellow" << endl;
    }
    else if (H >= 31 && H <= 40)
    {
        cout << " Color: Yellow-Green" << endl;
    }
    else if (H >= 41 && H <= 70)
    {
        cout << " Color: Green" << endl;
    }
    else if (H >= 71 && H <= 85)
    {
        cout << " Color: Green-Cyan" << endl;
    }
    else if (H >= 86 && H <= 100)
    {
        cout << " Color: Cyan" << endl;
    }
    else if (H >= 101 && H <= 110)
    {
        cout << " Color: Cyan-Blue" << endl;
    }
    else if (H >= 111 && H <= 130)
    {
        cout << " Color: Blue" << endl;
    }
    else if (H >= 131 && H <= 140)
    {
        cout << " Color: Violet" << endl;
    }
    else if (H >= 141 && H <= 160)
    {
        cout << " Color: Magenta" << endl;
    }
    else if (H >= 161 && H <= 167)
    {
        cout << " Color: Magenta-Pink" << endl;
    }
    else if (H >= 168 && H <= 175)
    {
        cout << " Color: Pink" << endl;
    }
    else if (H >= 176 && H <= 177)
    {
        cout << " Color: Pink-Red" << endl;
    }
    else if (H >= 176 && H <= 177)
    {
        cout << " Color: Pink-Red" << endl;
    }
    else if (H >= 178 && H <= 180)
    {
        cout << " Color: Red" << endl;
    }
    sprintf(name,"H=%d",H);
    putText(img0,".", Point(i,j) , FONT_HERSHEY_SIMPLEX, .7, Scalar(255,0,0), 2,8,false );
     putText(HSV,".", Point(i,j) , FONT_HERSHEY_SIMPLEX, .7, Scalar(255,0,0), 2,8,false );
//
//                
//            }//
//        }
    
    
    // show the images
    imshow("original", img0);
    imshow("mask", mask);
    imshow("canny", img1);
    imshow("cropped", crop);
     imshow("HSV", HSV);
    

    
//    imwrite("/home/philipp/img/apple_canny.jpg", img1);
//    imwrite("/home/philipp/img/apple_mask.jpg", mask);
//    imwrite("/home/philipp/img/apple_cropped.jpg", crop);
    
    waitKey();
    return 0;
}


//---------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------
