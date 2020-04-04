#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

enum FeatureType
{
    SHITMOASI,
    HARRIS,
    FAST,
    BRISK,
    ORB,
    AKAZE,
    SIFT,
    FREAK,
    BRIEF,
    FEATURE_TYPE_UNKNOWN
};

struct FeaturePerformance2D
{
    int numKeyPoints{0};

    int numMatchesFinal{0};
    int numMatchesRaw{0};

    double elapsedDetection{0.0};
    double elapsedDescCalc{0.0};
    double elapsedMatch{0.0};
};

/* FeatureType helper functions */
std::string GetFeatureTypeName(const FeatureType featureType);
FeatureType GetFeatureType(const std::string& name);

/* Timing helper functions */
double StartTimer();
double EndTimer(const double start);
double EndKeyPointTimer(const double start, std::vector<cv::KeyPoint> &keypoints, FeatureType ftype);

/* FilterKeypointsInRect removes points from keypoints not in rect */
void FilterKeypointsInRect(std::vector<cv::KeyPoint> &keypoints, cv::Rect &rect);

/* VisualizeKeypoints shows a window diplaying given keypoints */
void VisualizeKeypoints(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, const std::string& featureTypeName);

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, FeaturePerformance2D &perf, bool bVis = false);
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, FeaturePerformance2D &perf, bool bVis = false);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, FeatureType detectorType, FeaturePerformance2D &perf, bool bVis = false);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, FeatureType descriptorType, FeaturePerformance2D &perf);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, const std::string& descriptorType, const std::string& matcherType, const std::string& selectorType, FeaturePerformance2D &perf);

#endif /* matching2D_hpp */
