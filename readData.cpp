#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"



#include <boost/thread/thread.hpp>
#include <boost/shared_ptr.hpp>


#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace cv;

class readData{

public:
    void readRGBDFromFile(string& rgb_name1, string& depth_name1, string& rgb_name2, string& depth_name2)
    {
         img_1 = imread(rgb_name1, CV_LOAD_IMAGE_GRAYSCALE);
         depth_1 = imread(depth_name1, CV_LOAD_IMAGE_ANYDEPTH); // CV_LOAD_IMAGE_ANYDEPTH

         img_2 = imread(rgb_name2, CV_LOAD_IMAGE_GRAYSCALE);
         depth_2 = imread(depth_name2, CV_LOAD_IMAGE_ANYDEPTH);
         assert(img_1.type()==CV_8U);
         assert(img_2.type()==CV_8U);
         assert(depth_1.type()==CV_16U);
         assert(depth_2.type()==CV_16U);


     //    cv::imshow("depth 1", depth_1);
     //    cv::waitKey(0);
    }


    void featureMatching()
    {
        //-- Step 1: Detect the keypoints using SURF Detector
        int minHessian = 400;

        SurfFeatureDetector detector( minHessian );


        detector.detect( img_1, keypoints_1 );
        detector.detect( img_2, keypoints_2 );
        //imshow( "Good Matches", img_matches );

        //-- Step 2: Calculate descriptors (feature vectors)
        SurfDescriptorExtractor extractor;


        Mat descriptors_1, descriptors_2;

        extractor.compute(img_1, keypoints_1, descriptors_1 );
        extractor.compute(img_2, keypoints_2, descriptors_2 );



        //-- Step 3: Matching descriptor vectors using FLANN matcher
        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;
        matcher.match( descriptors_1, descriptors_2, matches );

        double max_dist = 0; double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_1.rows; i++ )
        {
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );

        //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
        //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
        //-- small)
        //-- PS.- radiusMatch can also be used here.
        std::vector< DMatch > good_matches;

        for( int i = 0; i < descriptors_1.rows; i++ )
        {
          if( matches[i].distance <= max(2*min_dist, 0.02) )
            { good_matches.push_back( matches[i]); }
        }

        //-- Draw only "good" matches
        Mat img_matches;
        drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        //-- Show detected matches
        imshow( "Good Matches", img_matches);

        vector<int> matchedKeypointsIndex1, matchedKeypointsIndex2;

        for( int i = 0; i < (int)good_matches.size(); i++ )
        {
            printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
            matchedKeypointsIndex1.push_back(good_matches[i].queryIdx);
            matchedKeypointsIndex2.push_back(good_matches[i].trainIdx);
        }

        for(int i=0; i<(int)matchedKeypointsIndex1.size(); i++)
        {
            int x1=keypoints_1[matchedKeypointsIndex1[i]].pt.x;
            int y1=keypoints_1[matchedKeypointsIndex1[i]].pt.y;
            auto depthValue1 = depth_1.at<unsigned short>(keypoints_1[matchedKeypointsIndex1[i]].pt.y, keypoints_1[matchedKeypointsIndex1[i]].pt.x);
            double worldZ1=0;
            if(depthValue1 > min_dis && depthValue1 < max_dis )
            {
               worldZ1=depthValue1/factor;
            }
           // cout<<"matchedKeypointsIndex1[i] "<<matchedKeypointsIndex1[i]<<"  x1 "<<x1<<" y1  "<<y1<<" worldZ1  "<<worldZ1<<"  depthValue1  "<<depthValue1<<endl;

            double worldX1=(keypoints_1[matchedKeypointsIndex1[i]].pt.x-cx)*worldZ1/fx;
            double worldY1=(keypoints_1[matchedKeypointsIndex1[i]].pt.y-cy)*worldZ1/fy;

            cout<<i<<"th matchedKeypointsIndex1  "<<matchedKeypointsIndex1[i]<<"   worldX1  "<<worldX1<<"  worldY1   "<<worldY1<<"  worldZ1   "<<worldZ1<<endl;
        }

        for(int i=0; i<(int)matchedKeypointsIndex2.size(); i++)
        {
            int x2=keypoints_2[matchedKeypointsIndex2[i]].pt.x;
            int y2=keypoints_2[matchedKeypointsIndex2[i]].pt.y;
            auto depthValue2 = depth_2.at<unsigned short>(keypoints_2[matchedKeypointsIndex2[i]].pt.y, keypoints_2[matchedKeypointsIndex2[i]].pt.x);
            double worldZ2=0;
            if(depthValue2> min_dis && depthValue2 < max_dis )
            {
               worldZ2=depthValue2/factor;
            }
           // cout<<"matchedKeypointsIndex2[i] "<<matchedKeypointsIndex2[i]<<"  x2 "<<x2<<" y2 "<<y2<<" z2 "<<z2<<"  depthValue2  "<<depthValue2<<endl;

            double worldX2=(keypoints_2[matchedKeypointsIndex2[i]].pt.x-cx)*worldZ2/fx;
            double worldY2=(keypoints_2[matchedKeypointsIndex2[i]].pt.y-cy)*worldZ2/fy;

            cout<<i<<"th matchedKeypointsIndex2  "<<matchedKeypointsIndex2[i]<<"   worldX2  "<<worldX2<<"  worldY2   "<<worldY2<<"  worldZ2  "<<worldZ2<<endl;
        }

   }

    void testing(string& rgb_name1, string& depth_name1, string& rgb_name2, string& depth_name2)
    {
        readRGBDFromFile(rgb_name1, depth_name1, rgb_name2, depth_name2);
        featureMatching();

    }

    Mat img_1, img_2;

    Mat depth_1, depth_2;


    vector<Point2f> points2d1, points2d2;
    vector<KeyPoint> keypoints_1, keypoints_2;

    vector<Point3d>  feature_point3D_1, feature_point3D_2;

    vector< DMatch > matches;

        //camera parameters
    double fx = 525.0; //focal length x
    double fy = 525.0; //focal le

    double cx = 319.5; //optical centre x
    double cy = 239.5; //optical centre y

    double min_dis = 800;
    double max_dis = 35000;

    double X1, Y1, Z1, X2, Y2, Z2;
    double factor = 5000;






    /* factor = 5000 for the 16-bit PNG files
    or factor =1 for the 32-bit float images in the ROS bag files

    for v in range (depth_image.height):
    for u in range (depth_image.width):

    Z = depth_image[v,u]/factor;
    X = (u-cx) * Z / fx;
    Y = (v-cy) * Z / fy;
    */

};

int main()
{
    readData r;

    string rgb1="/home/lili/workspace/rgbd_dataset_freiburg2_large_with_loop/MotionEstimation/FeatureMatching/rgb3.png";
    string depth1="/home/lili/workspace/rgbd_dataset_freiburg2_large_with_loop/MotionEstimation/FeatureMatching/depth3.png";
    string rgb2="/home/lili/workspace/rgbd_dataset_freiburg2_large_with_loop/MotionEstimation/FeatureMatching/rgb4.png";
    string depth2="/home/lili/workspace/rgbd_dataset_freiburg2_large_with_loop/MotionEstimation/FeatureMatching/depth4.png";

    r.testing(rgb1,depth1,rgb2,depth2);


    return 0;
}
