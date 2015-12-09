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

#include <Eigen/Geometry>

//Lourakis LM Optimisation includes
#include "levmar.h"


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
          if( matches[i].distance <= max(3*min_dist, 0.03) )
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

        vector<Point2d> imgpts1, imgpts2;

        for( int i = 0; i < (int)good_matches.size(); i++ )
        {
            printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
            matchedKeypointsIndex1.push_back(good_matches[i].queryIdx);
            matchedKeypointsIndex2.push_back(good_matches[i].trainIdx);
            imgpts1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
            imgpts2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
        }

        //Find the fundamental matrix

        Mat F = findFundamentalMat(imgpts1, imgpts2, FM_RANSAC, 3, 0.99);

        cout<<"Testing F" <<endl<<Mat(F)<<endl;
        //Essential matrix: compute then extract cameras [R|t]
        Mat K=cv::Mat(3,3,CV_64F);
        K.at<double>(0,0)=fx;
        K.at<double>(1,1)=fy;
        K.at<double>(2,2)=1;
        K.at<double>(0,2)=cx;
        K.at<double>(1,2)=cy;
        K.at<double>(0,1)=0;
        K.at<double>(1,0)=0;
        K.at<double>(2,0)=0;
        K.at<double>(2,1)=0;

        cout<<"Testing K" <<endl<<Mat(K)<<endl;

        Mat_<double> E=K.t()*F*K;

        if(fabsf(determinant(E)) > 1e-07)
        {
			cout << "det(E) != 0 : " << determinant(E) << "\n";

		}
		else
		{
            cout<<"Essential Matrix is correct"<<endl;
		}

        cout<<"Testing E" <<endl<<Mat(E)<<endl;

        //decompose E to P' Hz(9.19)
        SVD svd(E,SVD::MODIFY_A);
        Mat svd_u = svd.u;
        Mat svd_vt = svd.vt;
        Mat svd_w = svd.w;

        cout<<"Testing svd" <<endl<<Mat(svd_u)<<endl;


        Matx33d W(0,-1,0,
        1,0,0,
        0,0,1);  //HZ 9.13

        Mat_<double> R = svd_u*Mat(W)*svd_vt;  //HZ 9.19
        Mat_<double> t = svd_u.col(2);
        Matx34d P1=Matx34d( R(0,0), R(0,1), R(0,2), t(0),
                    R(1,0), R(1,1), R(1,2), t(1),
                    R(2,0), R(2,1), R(2,2), t(2));

       cout << "Testing P1 " << endl << Mat(P1) << endl;

        vector<Point3d>  feature_world3D_1;

       // project the matched 2D keypoint in image1 to the 3D points in world coordinate(the camera coordinate equals to the world coordinate in this case) using back-projection
        for(int i=0; i<(int)matchedKeypointsIndex1.size(); i++)
         {
            auto depthValue1 = depth_1.at<unsigned short>(keypoints_1[matchedKeypointsIndex1[i]].pt.y, keypoints_1[matchedKeypointsIndex1[i]].pt.x);
            double worldZ1=0;
            if(depthValue1 > min_dis && depthValue1 < max_dis )
            {
               worldZ1=depthValue1/factor;
            }

            double worldX1=(keypoints_1[matchedKeypointsIndex1[i]].pt.x-cx)*worldZ1/fx;
            double worldY1=(keypoints_1[matchedKeypointsIndex1[i]].pt.y-cy)*worldZ1/fy;

            cout<<i<<"th matchedKeypointsIndex1  "<<matchedKeypointsIndex1[i]<<"   worldX1  "<<worldX1<<"  worldY1   "<<worldY1<<"  worldZ1   "<<worldZ1<<endl;

            //store point cloud
            feature_world3D_1.push_back(Point3d(worldX1,worldY1,worldZ1));

        }

       cv::Mat rvec(3,1,cv::DataType<double>::type);
       cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);

       rotationMatrix.at<double>(0,0)=R(0,0);
       rotationMatrix.at<double>(0,1)=R(0,1);
       rotationMatrix.at<double>(0,2)=R(0,2);
       rotationMatrix.at<double>(1,0)=R(1,0);
       rotationMatrix.at<double>(1,1)=R(1,1);
       rotationMatrix.at<double>(1,2)=R(1,2);
       rotationMatrix.at<double>(2,0)=R(2,0);
       rotationMatrix.at<double>(2,1)=R(2,1);
       rotationMatrix.at<double>(2,2)=R(2,2);

       Rodrigues(rotationMatrix, rvec);

       cout<<"Testing Rodrigues "<<Mat(rvec)<<endl;

        cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
        distCoeffs.at<double>(0) = 0;
        distCoeffs.at<double>(1) = 0;
        distCoeffs.at<double>(2) = 0;
        distCoeffs.at<double>(3) = 0;

       std::vector<cv::Point2d> estimatedProjectedPoints2;
       cv::projectPoints(feature_world3D_1, rvec, t, K, distCoeffs, estimatedProjectedPoints2);

        //calculate reprojection error
        double totalErr = 0;
        double err = 0;

        for(int i = 0; i < (int)good_matches.size(); ++i)
        {
            std::cout << "feature points from image1 in world coordinate " <<feature_world3D_1[i]<< " re-projected to image2 in image coordinate" << estimatedProjectedPoints2[i] << std::endl;
            //for image1, camera coordinate is the same with world coordinate
            err = norm(Mat(imgpts2[i]), Mat(estimatedProjectedPoints2[i]), CV_L2);              // reprojection error for measured points and estimatedReprojected points in image2

            cout<<"err  "<<err<<endl;

            totalErr  += err*err;                // sum it up

        }

        int n = (int)good_matches.size();
        double meanPerPointError=std::sqrt(totalErr/n);    //calculate the arithmmatical mean
        cout<<"meanPerPointError: "<<meanPerPointError<<endl;

       cv::Mat rotAndtransVec(6,1,cv::DataType<double>::type);

       rotAndtransVec.at<double>(0)=rvec.at<double>(0);
       rotAndtransVec.at<double>(1)=rvec.at<double>(1);
       rotAndtransVec.at<double>(2)=rvec.at<double>(2);
       rotAndtransVec.at<double>(3)=t.at<double>(0);
       rotAndtransVec.at<double>(4)=t.at<double>(1);
       rotAndtransVec.at<double>(5)=t.at<double>(2);

       cout<<"Testing rotAndtransVec "<<Mat(rotAndtransVec)<<endl;



       /*
       float opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	   opts[0]=LM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-20;
	   opts[4]= LM_DIFF_DELTA;*/

       // Allocate working memory outside levmar for efficiency
     //  float * work = new float[(LM_DIF_WORKSZ(6, max_inlier_count*3))];

    // par_vec stores a minimal representation of the transformation (i.e. translation + rotation) to be used in the levmar rountine.
    // This allows us to avoid the ||q||^2 = 1 constraint in the optimization and hence deal with it as an unconstrained optimization
    // problem.
     double par_vec[6];
     par_vec[0]=rvec.at<double>(0);
     par_vec[1]=rvec.at<double>(1);
     par_vec[2]=rvec.at<double>(2);
     par_vec[3]=t.at<double>(0);
     par_vec[4]=t.at<double>(1);
     par_vec[5]=t.at<double>(2);



  // int slevmar_return = slevmar_dif(lm_projectPoints,   /*double *p,   I/O: initial parameter estimates. On output contains the estimated solution */
 //                                    par_vec,  /*double *x,     I: measurement vector. NULL implies a zero vector */
  //                                   hx,   /*double *x,          I: measurement vector. NULL implies a zero vector */
//                                     6,    /* I: parameter vector dimension (i.e. #unknowns), here is rotation angles and translation*/
 //                                    max_inlier_count * 3,   /* int n,  I: measurement vector dimension */
 //                                    100,   /*int itmax,       I: maximum number of iterations */
   //                                  opts,    /* double opts[5],  I: opts[0-4] = minim. options [\tau, \epsilon1, \epsilon2, \epsilon3, \delta]. Respectively the
                                                /* scale factor for initial \mu, stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2 and the
                                  -0.946237410033647              * step used in difference approximation to the Jacobian. If \delta<0, the Jacobian is approximated
                                                * with central differences which are more accurate (but slower!) compared to the forward differences
                                                * employed by default. Set to NULL for defaults to be used. */

                               //info,    /* double info[LM_INFO_SZ], O: information regarding the minimization. Set to NULL if don't care
                                               /* info[0]= ||e||_2 at initial p.
                                               * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, \mu/max[J^T J]_ii ], all computed at estimated p.
                                               * info[5]= # iterations,
                                               * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
                                                *                                 2 - stopped by small Dp
                                                *                                 3 - stopped by itmax
                                                *                                 4 - singular matrix. Restart from current p with increased \mu
                     *                                 5 - no further error reduction is possible. Restart with increased mu
                     *                                 6 - stopped by small ||e||_2
                     *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values; a user error
                     * info[7]= # function evaluations
                     * info[8]= # Jacobian evaluations
                     * info[9]= # linear systems solved, i.e. # attempts for reducing error
                     */
                                    // work,  /*double *work,      I: working memory, allocated internally if NULL. If !=NULL, it is assumed to point to
                                             /* a memory chunk at least LM_DIF_WORKSZ(m, n)*sizeof(double) bytes long*/
                                     //covariance,  /*double *covar,     O: Covariance matrix corresponding to LS solution; Assumed to point to a mxm matrix.
                                                   /* Set to NULL if not needed.*/

                                    // reinterpret_cast<void *>(&projectPoints_t));  /*void *adata   I: pointer to possibly needed additional data, passed uninterpreted to func.
                                                                                    /* Set to NULL if not needed
                                                                                    */

     /*
       ofstream fout1("feature_points.csv");

        for(int i=0; i<(int)matchedKeypointsIndex1.size(); i++)
        {
            int imageX1=keypoints_1[matchedKeypointsIndex1[i]].pt.x;
            int imageY1=keypoints_1[matchedKeypointsIndex1[i]].pt.y;
            auto depthValue1 = depth_1.at<unsigned short>(keypoints_1[matchedKeypointsIndex1[i]].pt.y, keypoints_1[matchedKeypointsIndex1[i]].pt.x);

           cout<<"matchedKeypointsIndex1[i] "<<matchedKeypointsIndex1[i]<<"  imageX1 "<<imageX1<<" imageY1  "<<imageY1<<endl;
           fout1<<i<<" "<<matchedKeypointsIndex1[i]<<" "<<imageX1<<" "<<imageY1<<endl;

        }

        for(int i=0; i<(int)matchedKeypointsIndex2.size(); i++)
        {
            int imageX2=keypoints_2[matchedKeypointsIndex2[i]].pt.x;
            int imageY2=keypoints_2[matchedKeypointsIndex2[i]].pt.y;

           cout<<"matchedKeypointsIndex2[i] "<<matchedKeypointsIndex2[i]<<"  imageX2 "<<imageX2<<" imageY2 "<<imageY2<<endl;
           fout1<<i<<" "<<matchedKeypointsIndex2[i]<<" "<<imageX2<<" "<<imageY2<<endl;
          }

         for(int i=0; i<(int)matchedKeypointsIndex1.size(); i++)
         {
            auto depthValue1 = depth_1.at<unsigned short>(keypoints_1[matchedKeypointsIndex1[i]].pt.y, keypoints_1[matchedKeypointsIndex1[i]].pt.x);
            double worldZ1=0;
            if(depthValue1 > min_dis && depthValue1 < max_dis )
            {
               worldZ1=depthValue1/factor;
            }

            double worldX1=(keypoints_1[matchedKeypointsIndex1[i]].pt.x-cx)*worldZ1/fx;
            double worldY1=(keypoints_1[matchedKeypointsIndex1[i]].pt.y-cy)*worldZ1/fy;

            cout<<i<<"th matchedKeypointsIndex1  "<<matchedKeypointsIndex1[i]<<"   worldX1  "<<worldX1<<"  worldY1   "<<worldY1<<"  worldZ1   "<<worldZ1<<endl;
            fout1<<i<<" "<<matchedKeypointsIndex1[i]<<" "<<worldX1<<" "<<worldY1<<" "<<worldZ1<<endl;
        }

        for(int i=0; i<(int)matchedKeypointsIndex2.size(); i++)
        {
            auto depthValue2 = depth_2.at<unsigned short>(keypoints_2[matchedKeypointsIndex2[i]].pt.y, keypoints_2[matchedKeypointsIndex2[i]].pt.x);
            double cameraZ2=0;
            if(depthValue2> min_dis && depthValue2 < max_dis )
            {
               cameraZ2=depthValue2/factor;
            }
            double cameraX2=(keypoints_2[matchedKeypointsIndex2[i]].pt.x-cx)*cameraZ2/fx;
            double cameraY2=(keypoints_2[matchedKeypointsIndex2[i]].pt.y-cy)*cameraZ2/fy;

            cout<<i<<"th matchedKeypointsIndex2  "<<matchedKeypointsIndex2[i]<<"   cameraX2  "<<cameraX2<<"  cameraY2   "<<cameraY2<<"  cameraZ2  "<<cameraZ2<<endl;
            fout1<<i<<" "<<matchedKeypointsIndex2[i]<<" "<<cameraX2<<" "<<cameraY2<<" "<<cameraZ2<<endl;
        }
      */
   }



    void testing(string& rgb_name1, string& depth_name1, string& rgb_name2, string& depth_name2)
    {
        readRGBDFromFile(rgb_name1, depth_name1, rgb_name2, depth_name2);
        featureMatching();
    }

    Mat img_1, img_2;

    Mat depth_1, depth_2;



    vector<KeyPoint> keypoints_1, keypoints_2;


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
