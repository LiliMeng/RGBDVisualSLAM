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



//#include <pcl/visualization/cloud_viewer.h>

//#include <pcl/console/parse.h>

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

        for( int i = 0; i < (int)good_matches.size(); i++ )
        {
            printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

            //waitKey(0);
        }


    void generate3DFeatures()
    {
        for (int i=0; i<keypoints_1.size(); i++)
        {
            //assert(depth_1.type() == CV_8U);
            printf("depth type is %d\n", depth_1.type());
            auto depthValue1 = depth_1.at<unsigned short>(keypoints_1[i].pt.y, keypoints_1[i].pt.x);
            cout<<"depthValue1 "<<depthValue1<<endl;
            if(depthValue1 > min_dis && depthValue1 < max_dis )
            {
                Z1 = depthValue1/factor;
                X1 = (keypoints_1[i].pt.x-cx)*Z1/fx;
                Y1 = (keypoints_1[i].pt.y-cy)*Z1/fy;

                cout<<"X1"<<X1<<endl;
                cout<<"Y1"<<Y1<<endl;
                cout<<"Z1"<<Z1<<endl;
                feature_point3D_1.push_back(Point3d(X1,Y1,Z1));
            }
        }

        for (int i=0; i<keypoints_2.size(); i++)
        {
            auto depthValue2 = depth_2.at<unsigned short>(keypoints_2[i].pt.y, keypoints_2[i].pt.x);

            if(depthValue2 > min_dis && depthValue2 < max_dis )
            {
                Z2 = depthValue2/factor;
                X2 = (keypoints_2[i].pt.x-cx)*Z2/fx;
                Y2 = (keypoints_2[i].pt.y-cy)*Z2/fy;
                feature_point3D_2.push_back(Point3d(X2,Y2,Z2));
                cout<<"X2"<<X2<<endl;
                cout<<"Y2"<<Y2<<endl;
                cout<<"Z2"<<Z2<<endl;
            }
        }
     }

     /*
     void draw3DKeypoints()
     {

        pcl::visualization::CloudViewer viewer("Cloud Viewer");
      //Draw 3D keypoitns over the point cloud
        pointCloudKeypoints3D_2.reset(new pcl::PointCloud<pcl::PointXYZRGBA>());
        for(int i=0; i<keypoints_2.size(); i++)
        {
            int y=keypoints_2[i].pt.x;
            int x=keypoints_2[i].pt.y;
            int z=depth_2.at<unsigned short>(keypoints_2[i].pt.y, keypoints_2[i].pt.x);

            for(int rIndex=-2;rIndex<=2;rIndex++)
            {
                for(int cIndex=-2;cIndex<=2;cIndex++)
                {
                    int pointIndex=(x+cIndex)*640+(y+rIndex);
                    if(pointIndex>=0 && pointIndex<640*480)
                    {
                        pointCloudKeypoints3D_2->points[pointIndex].r=0;
                        pointCloudKeypoints3D_2->points[pointIndex].g=255;
                        pointCloudKeypoints3D_2->points[pointIndex].b=0;
                    }
                }
            }
        }
        viewer.showCloud(pointCloudKeypoints3D_2);
     }
     */

    void testing(string& rgb_name1, string& depth_name1, string& rgb_name2, string& depth_name2)
    {
        readRGBDFromFile(rgb_name1, depth_name1, rgb_name2, depth_name2);
        featureMatching();
        generate3DFeatures();
       // draw3DKeypoints();
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


    //pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pointCloudKeypoints3D_1, pointCloudKeypoints3D_2;



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

    string rgb1="/home/lili/workspace/rgbd_dataset_freiburg2_large_with_loop/MotionEstimation/FeatureMatching/rgb1.png";
    string depth1="/home/lili/workspace/rgbd_dataset_freiburg2_large_with_loop/MotionEstimation/FeatureMatching/depth1.png";
    string rgb2="/home/lili/workspace/rgbd_dataset_freiburg2_large_with_loop/MotionEstimation/FeatureMatching/rgb2.png";
    string depth2="/home/lili/workspace/rgbd_dataset_freiburg2_large_with_loop/MotionEstimation/FeatureMatching/depth2.png";

    r.testing(rgb1,depth1,rgb2,depth2);


    return 0;
}
