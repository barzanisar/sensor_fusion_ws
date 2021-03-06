/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2; // no. of images which are held in memory (ring buffer) at the same time
    bool bVis = false; // visualize results

    std::ofstream logfile;
    logfile.open("logs.csv");
    logfile << "Detector,Descriptor,Frame,Keypoints in ROI,Keypts in ROI matched,Time for detection [ms],Time for descriptor [ms],Total time [ms],\n";

    std::vector<string> detTypes = {"HARRIS", "SHITOMASI", "SIFT", "FAST", "ORB", "BRISK", "AKAZE"};
    std::vector<string> descTypes = {"SIFT", "BRIEF", "ORB", "BRISK", "FREAK", "AKAZE"};

    for (string detectorType : detTypes)
    {
        for (string descriptorType : descTypes)
        {
            if (detectorType == "AKAZE" && descriptorType != "AKAZE") // AKAZE keypoints only work with AKAZE descriptors and vice versa
            {
                continue;
            }

            if (detectorType != "AKAZE" && descriptorType == "AKAZE")
            {
                continue;
            }

            if (detectorType == "SIFT" && descriptorType == "ORB") // Computing ORB descriptors from SIFT keypoints causes memory insufficient error 
            {
                continue;
            }

            deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

            /* MAIN LOOP OVER ALL IMAGES */

            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                // Data frame is ring buffer so pop oldest image if buffer is full
                if (dataBuffer.size() == dataBufferSize)
                {
                    dataBuffer.pop_front();
                }

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;
                dataBuffer.push_back(frame);

                cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

                /* DETECT IMAGE KEYPOINTS */

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image
                //string detectorType = "HARRIS"; // HARRIS, FAST, BRISK, ORB, AKAZE, SIFT,
                float keyptsTimeMs = 0;

                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, keyptsTimeMs);
                }
                else if (detectorType.compare("HARRIS") == 0)
                {
                    detKeypointsHarris(keypoints, imgGray, keyptsTimeMs);
                }
                else
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, keyptsTimeMs);
                }

                // only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = true;
                cv::Rect vehicleRect(535, 180, 180, 150);
                vector<cv::KeyPoint> keypointsOnVehicle;
                if (bFocusOnVehicle)
                {
                    for (auto keypoint : keypoints)
                    {
                        if (vehicleRect.contains(keypoint.pt))
                            keypointsOnVehicle.push_back(keypoint);
                    }

                    keypoints = keypointsOnVehicle;
                    cout << "Key points on the vehicle: " << keypointsOnVehicle.size() << endl;
                }

                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;
                cout << "#2 : DETECT KEYPOINTS done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                cv::Mat descriptors;
                //string descriptorType = "BRISK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
                float descTimeMs = 0;
                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, descTimeMs);
                //// EOF STUDENT ASSIGNMENT

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                if (dataBuffer.size() == 1)
                {
                    logfile << detectorType << "," << descriptorType << "," << imgIndex
                            << "," << keypoints.size() << ","
                            << "NaN"
                            << ","
                            << keyptsTimeMs << "," << descTimeMs << "," << keyptsTimeMs + descTimeMs << endl;
                }

                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {
                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";                                                                                                   // MAT_BF, MAT_FLANN
                    string descriptorKind = (descriptorType.compare("SIFT") == 0 || descriptorType.compare("SURF") == 0) ? "DES_HOG" : "DES_BINARY"; // DES_BINARY, DES_HOG
                    cout << descriptorKind << " " << descriptorType << endl;
                    string selectorType = "SEL_KNN"; // SEL_NN, SEL_KNN

                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                     (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                     matches, descriptorKind, matcherType, selectorType);

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    logfile << detectorType << "," << descriptorType << "," << imgIndex
                            << "," << keypoints.size() << "," << matches.size() << ","
                            << keyptsTimeMs << "," << descTimeMs << "," << keyptsTimeMs + descTimeMs << endl;

                    // visualize matches between current and previous image
                    bVis = false;
                    if (bVis)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                    bVis = false;
                }

            } // eof loop over all images
        }     // eof descTypes
    }         // eof detTypes

    logfile.close();

    return 0;
}
