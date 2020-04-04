/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <list>
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

/* MAIN PROGRAM 
usage:
./prog [det tyype] [desc type]
*/
int main(int argc, const char *argv[]) {
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "./";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // Vehicle settings
    cv::Rect vehicleRect(560, 190, 110, 135);

    // experiment settings
    FeatureType detectorType = GetFeatureType(argv[1]);   //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    FeatureType descriptorType = GetFeatureType(argv[2]); //// -> BRIEF, ORB, FREAK, AKAZE, SIFT
    constexpr bool bFocusOnVehicle = true;

    string matcherType = "MAT_BF"; // MAT_BF, MAT_FLANN
    string descOpt;                // DES_BINARY, DES_HOG
    switch (descriptorType) {
        case SIFT:
        case AKAZE:
            descOpt = "DES_HOG";
            break;

        case BRIEF:
        case FREAK:
        case ORB:
        default:
            descOpt = "DES_BINARY";
    }
    string selectorType = "SEL_KNN"; // SEL_NN, SEL_KNN

    // data log
    std::string expResultsFile =
            string("./results/exp_detector_") + GetFeatureTypeName(detectorType) + string("_descriptor_") +
            GetFeatureTypeName(descriptorType) + string(".csv");
    std::vector<FeaturePerformance2D> perfResults;
    std::ofstream expResults(expResultsFile, std::ofstream::out);

    // misc
    int dataBufferSize = 2;     // no. of images which are held in memory (ring buffer) at the same time
    list<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;          // visualize results
    bool bVisualizeKeypoints = false;

    /* MAIN LOOP OVER ALL IMAGES */
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++) {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
        FeaturePerformance2D perf;

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);
        while (dataBuffer.size() > dataBufferSize) {
            dataBuffer.pop_front();
        }

        //// EOF STUDENT ASSIGNMENT
        std::cout << "#1 : LOAD IMAGE INTO BUFFER done, img buffer size: " << dataBuffer.size() << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        switch (detectorType) {
            case SHITMOASI:
                detKeypointsShiTomasi(keypoints, imgGray, perf, bVisualizeKeypoints);
                break;

            case HARRIS:
                detKeypointsHarris(keypoints, imgGray, perf, bVisualizeKeypoints);
                break;

            case FAST:
            case BRISK:
            case AKAZE:
            case SIFT:
            case ORB:
                detKeypointsModern(keypoints, imgGray, detectorType, perf, bVisualizeKeypoints);
                break;

            default:
                throw std::runtime_error(string("unknown feature type ") + to_string(detectorType));
        }

        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        if (bFocusOnVehicle) {
            FilterKeypointsInRect(keypoints, vehicleRect);
        }
        perf.numKeyPoints = keypoints.size();

        // visualize here
        if (bVisualizeKeypoints)
            VisualizeKeypoints(dataBuffer.back().cameraImg, keypoints, GetFeatureTypeName(detectorType));

        //// EOF STUDENT ASSIGNMENT
        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.rbegin())->keypoints = keypoints;

        std::cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descripType

        cv::Mat descriptors;
        descKeypoints(dataBuffer.rbegin()->keypoints, dataBuffer.rbegin()->cameraImg, descriptors, descriptorType,
                      perf);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.rbegin())->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */
            vector<cv::DMatch> matches;

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
            matchDescriptors(dataBuffer.back().keypoints, (dataBuffer.rbegin()++)->keypoints,
                             dataBuffer.back().descriptors, (dataBuffer.rbegin()++)->descriptors,
                             matches, descOpt, matcherType, selectorType, perf);
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.rbegin())->kptMatches = matches;

            std::cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            cv::Mat matchImg = ((dataBuffer.rbegin())->cameraImg).clone();
            cv::drawMatches((dataBuffer.rbegin()++)->cameraImg, (dataBuffer.rbegin()++)->keypoints,
                            dataBuffer.rbegin()->cameraImg, dataBuffer.rbegin()->keypoints,
                            matches, matchImg,
                            cv::Scalar::all(-1), cv::Scalar::all(-1),
                            vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string filename =
                    string("./results/img/exp_detector_") + GetFeatureTypeName(detectorType) + "_descriptor_" +
                    GetFeatureTypeName(descriptorType) + to_string(imgIndex) + ".jpg";
            cv::imwrite(filename, matchImg);
            if (bVis) {
                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
        }

        // Keep track of the performance results
        perfResults.emplace_back(perf);
    } // eof loop over all images

    // Write results using flat schema
    expResults
            << "keypoint_type,descriptor_type,match_algorithm,selection_algorithm, "
            << "num_keypoints,num_matches_raw,num_matches_final,"
            << "elapsed_detect,elapsed_desc_calc,elapsed_match\n";
    for (auto res : perfResults) {
        expResults << GetFeatureTypeName(detectorType) << ","
                   << GetFeatureTypeName(descriptorType) << ","
                   << matcherType << ","
                   << selectorType << ","
                   << res.numKeyPoints << ","
                   << res.numMatchesRaw << ","
                   << res.numMatchesFinal << ","
                   << (res.elapsedDetection) << ","
                   << (res.elapsedDescCalc) << ","
                   << res.elapsedMatch << "\n";
    }

    return 0;
}




























