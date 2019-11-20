
#include <iostream>
#include <algorithm>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "camFusion.hpp"
#include "lidarData.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type); // X,Y,Z,1 points in Lidar frame
    cv::Mat Y(3, 1, cv::DataType<double>::type); // x,y,z points in camera image

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left, bottom + 20), cv::FONT_ITALIC, 0.5, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left, bottom + 50), cv::FONT_ITALIC, 0.5, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> potentialMatchesInBB;
    std::vector<float> distanceMoved;
    float meanDist = 0;
    for (auto match : kptMatches)
    {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
        {
            float dist = cv::norm(kptsCurr[match.trainIdx].pt - kptsPrev[match.queryIdx].pt);
            distanceMoved.push_back(dist);
            meanDist += dist;

            potentialMatchesInBB.push_back(match);
        }
    }

    //float mean = std::accumulate(distanceMoved.begin(), distanceMoved.end(), 0.0) / distanceMoved.size();
    meanDist = meanDist / distanceMoved.size();
    cout << "mean dist moved by keypts " << meanDist << endl;

    float distanceThreshold = 1.5 * meanDist; //pixels

    for (uint i = 0; i < potentialMatchesInBB.size(); ++i)
    {
        if (distanceMoved[i] < distanceThreshold)
        {
            boundingBox.kptMatches.push_back(potentialMatchesInBB[i]);
        }
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    float dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}

float findRobustMinX(std::vector<LidarPoint> &lidarPoints)
{
    sort(lidarPoints.begin(), lidarPoints.end(), [](const LidarPoint &lhs, const LidarPoint &rhs) {
        return lhs.x < rhs.x;
    });

    //return lidarPoints[0].x;

    unsigned int firstNpts = lidarPoints.size() / 2; // 100

    //return lidarPoints[firstNpts/2].x; //median of first n pts

    float mean = std::accumulate(lidarPoints.begin(), lidarPoints.begin() + firstNpts, 0.0, [](double sum, const LidarPoint &lpt) { return sum + lpt.x; }) / firstNpts;

    //return mean;
    
    std::vector<float> diff(firstNpts);
    std::transform(lidarPoints.begin(), lidarPoints.begin() + firstNpts, diff.begin(), [mean](const LidarPoint &lpt) { return lpt.x - mean; });

    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    float stdev = std::sqrt(sq_sum / firstNpts);

    float lowerBound = mean - 2 * stdev;

    cout << "mean " << mean << endl;
    cout << "lowerBound " << lowerBound << endl;

    for (auto lpt : lidarPoints)
    {
        if (lpt.x > lowerBound) // check if lpt.x is inlier
        {
            return lpt.x;
        }
        else
        {
            cout << "outlier " << lpt.x << endl;
        }
    }
}
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    float minXPrev = findRobustMinX(lidarPointsPrev);
    cout << endl << "minXPrev " << minXPrev << endl;
    showLidarTopview(lidarPointsPrev, cv::Size(5.5, 10.0), cv::Size(400, 800), minXPrev, true);
    float minXCurr = findRobustMinX(lidarPointsCurr);
    cout << "minXCurr " << minXCurr << endl;
    showLidarTopview(lidarPointsCurr, cv::Size(5.5, 10.0), cv::Size(400, 800) ,minXCurr, true);

    float velocity = (minXPrev - minXCurr) * frameRate;

    TTC = minXCurr / velocity;
}

void matchBoundingBoxes(const std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, const DataFrame &prevFrame, const DataFrame &currFrame)
{
    if (prevFrame.boundingBoxes.size() == 0 || currFrame.boundingBoxes.size() == 0)
    {
        std::cout << "ERROR: No bounding boxes found! " << std::endl;
        return;
    }

    std::cout << "prevFrame.boundingBoxes.size() " << prevFrame.boundingBoxes.size() << "currFrame.boundingBoxes.size() " << currFrame.boundingBoxes.size() << std::endl;
    cv::Mat hist = cv::Mat::zeros(prevFrame.boundingBoxes.size(), currFrame.boundingBoxes.size(), CV_16UC1); //cv::DataType<uint>::type

    std::vector<cv::DMatch> selMatches;
    for (auto match : matches)
    {
        std::vector<int> prevFrameBBoxIds, currFrameBBoxIds;
        //int prevFrameBBoxId, currFrameBBoxId;
        cv::Point2f prevFramePt = prevFrame.keypoints[match.queryIdx].pt;
        cv::Point2f currFramePt = currFrame.keypoints[match.trainIdx].pt;

        for (auto bBox : prevFrame.boundingBoxes)
        {
            if (bBox.roi.contains(prevFramePt))
            {
                prevFrameBBoxIds.push_back(bBox.boxID);
            }
        }

        for (auto bBox : currFrame.boundingBoxes)
        {
            if (bBox.roi.contains(currFramePt))
            {
                currFrameBBoxIds.push_back(bBox.boxID);
            }
        }

        if (prevFrameBBoxIds.size() == 1 && currFrameBBoxIds.size() == 1)
        {
            hist.at<uint16_t>(prevFrameBBoxIds[0], currFrameBBoxIds[0]) += 1;
            // if (prevFrameBBoxIds[0] == 5 || currFrameBBoxIds[0] == 5)
            // {
            //     selMatches.push_back(match);
            //     std::cout <<  "prevFrameBBoxIds[0] " << prevFrameBBoxIds[0] << " " <<  currFrameBBoxIds[0] << std::endl;
            //     std::cout << "hist(5,5) " << hist.at<uint16_t>(prevFrameBBoxIds[0], currFrameBBoxIds[0]) << std::endl;
            // }
        }
    }

    // //print hist for debugging
    // for (uint row = 0; row < hist.rows; ++row){
    //     for (uint col = 0; col < hist.cols; ++col)
    //     {
    //         cout << hist.at<uint16_t>(row,col) << ", ";
    //     }
    //     cout << endl;
    // }

    for (uint i = 0; i < hist.rows; i++)
    {
        double max = 0;
        cv::Point maxLoc;
        cv::minMaxLoc(hist.row(i), NULL, &max, NULL, &maxLoc);

        uint minNumberOfMatches = 10;
        std::cout << "row " << i << " col " << maxLoc.x << " val " << hist.at<uint16_t>(i, maxLoc.x) << std::endl;

        if (hist.at<uint16_t>(i, maxLoc.x) > minNumberOfMatches)
        {
            std::cout << "prevFrame bbox " << i << " matches with currFrame bbox " << maxLoc.x << std::endl;
            bbBestMatches.emplace(i, maxLoc.x);
        }
    }

    // // visualize matches between current and previous image
    // bool bVis = true;
    // if (bVis)
    // {
    //     cv::Mat matchImg = (currFrame.cameraImg).clone();
    //     cv::drawMatches(prevFrame.cameraImg, prevFrame.keypoints,
    //                     currFrame.cameraImg, currFrame.keypoints,
    //                     selMatches, matchImg,
    //                     cv::Scalar::all(-1), cv::Scalar::all(-1),
    //                     vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //     string windowName = "Matching keypoints between two camera images";
    //     cv::namedWindow(windowName, 7);
    //     cv::imshow(windowName, matchImg);
    //     cout << "Press key to continue to next image" << endl;
    //     cv::waitKey(0); // wait for key to be pressed
    // }
    // bVis = false;
}
