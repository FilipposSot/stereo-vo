/*

Author: Filippos Sotiropoulos (filippos.sotiropoulos@gmail.com)

*/

#include "vo_methods.hpp"

using namespace cv;
using namespace std;

#define MAX_FRAME 1000
#define MIN_NUM_POINTS 2000

int main( int argc, char** argv )	{

    ofstream myfile;
    myfile.open ("results1_1.txt");

    double scale = 1.00;

    char text[100];
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;  
    cv::Point textOrg(10, 50);

    // feature detection, tracking
    vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
    vector<uint> pointIndex;
    vector<uchar> status;

    //TODO: add a fucntion to load these values directly from KITTI's calib files
    // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
    double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157);

    Mat points3D_t0, points4D_t0;
    Mat projMat_l = (Mat_<float>(3,4) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 4.538225000000e+01, 0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, -1.130887000000e-01, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 3.779761000000e-03);
    Mat projMat_r = (Mat_<float>(3,4) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.372877000000e+02, 0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 2.369057000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.915215000000e-03);

    vector<Point2f> prevFeatures = points2;
    vector<Point2f> currFeatures;

    clock_t begin = clock();

    namedWindow( "Estimated Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

    Mat traj = Mat::zeros(1000, 1000, CV_8UC3);
    // traj.setTo(cv::Scalar(255,255,255));
    
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat pose = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat Rpose = cv::Mat::eye(3, 3, CV_64F);
    
    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat frame_pose32 = cv::Mat::eye(4, 4, CV_32F);

    Mat currImage_l, currImage_r;
    Mat prevImage_l, prevImage_r;

    // Load in the initial images from Kitti
    char filename_l[100];
    char filename_r[100];

    sprintf(filename_l, "/home/filippos/data/KITTI_odometry/data_odometry_color/dataset/sequences/00/image_2/%06d.png", 0);
    sprintf(filename_r, "/home/filippos/data/KITTI_odometry/data_odometry_color/dataset/sequences/00/image_3/%06d.png", 0);
    Mat prevImage_c_l = imread(filename_l);
    Mat prevImage_c_r = imread(filename_r);

    if ( !prevImage_c_l.data || !prevImage_c_r.data ) { 
      std::cout<< " --(!) Error reading images " << std::endl; return -1;
    }

    cvtColor(prevImage_c_l, prevImage_l, COLOR_BGR2GRAY);
    cvtColor(prevImage_c_r, prevImage_r, COLOR_BGR2GRAY);

    // declare keypoints for detection
    std::vector<KeyPoint> keypoints_l, keypoints_rd;
    vector<Point2f> prevPoints_l;
    cv::Mat inliers; 

    // Capture a video of the
    
    VideoWriter video_trajectory("trajectory.avi", cv::VideoWriter::fourcc('M','J','P','G'), 25, Size(1000,1000));
    VideoWriter video_tracking("tracking.avi"  , cv::VideoWriter::fourcc('M','J','P','G'), 25, prevImage_c_l.size());

    for(int numFrame=1; numFrame < MAX_FRAME; numFrame++)	{
    	
        // Load the left and right images and convert to colour
        sprintf(filename_l, "/home/filippos/data/KITTI_odometry/data_odometry_color/dataset/sequences/00/image_2/%06d.png", numFrame);
        sprintf(filename_r, "/home/filippos/data/KITTI_odometry/data_odometry_color/dataset/sequences/00/image_3/%06d.png", numFrame);
        Mat currImage_c_l = imread(filename_l);
        Mat currImage_c_r = imread(filename_r);
        cvtColor(currImage_c_l, currImage_l, COLOR_BGR2GRAY);
        cvtColor(currImage_c_r, currImage_r, COLOR_BGR2GRAY);


        // check the number of points and if its below a threshold add some more
        if (prevPoints_l.size() < MIN_NUM_POINTS) {
          appendNewPoints(prevImage_c_l, prevPoints_l);
        }

        vector<Point2f> prevPoints_r, currPoints_l, currPoints_r;
        vector<uchar> status_l, status_r;

        // match the points in the left and right image and then track to the next time
        Mat image_tracking;
        matchAndTrack( prevImage_l, prevImage_r,
                       currImage_l, currImage_r, 
                       prevPoints_l, 
                       prevPoints_r, 
                       currPoints_l, 
                       currPoints_r,
                       image_tracking);

        // triangulate the points based on a opencv function and 
        cv::triangulatePoints( projMat_l,  projMat_r,  currPoints_l, currPoints_r,  points4D_t0);
        cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

        // Calculate the relative rotation and translation
        odometryCalculation(projMat_l, projMat_r, 
                            prevPoints_l, currPoints_l,
                            points3D_t0, rotation, translation, inliers);


        // clone images and points for the next iteration
        prevImage_l = currImage_l.clone();
        prevImage_r = currImage_r.clone();
        prevImage_c_l = currImage_c_l.clone();
        prevImage_c_r = currImage_c_r.clone();
        prevPoints_l  = currPoints_l;

        cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);
        cv::Mat rigid_body_transformation;

        // if the rotation is reasonable integrate the visual odometry to the current pose
        if(abs(rotation_euler[1])<0.1 && abs(rotation_euler[0])<0.1 && abs(rotation_euler[2])<0.1)
        {
            integrateOdometry(0 , rigid_body_transformation, frame_pose, rotation, translation);

        } else {

            std::cout << "Too large rotation"  << std::endl;
        }

        visualizeTrajectory(traj, frame_pose, points3D_t0, inliers );
        
        video_trajectory.write(traj);
        video_tracking.write(image_tracking);

        waitKey(1);

    }

  return 0;
}