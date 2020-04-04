#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                      cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, const std::string &descriptorType,
                      const std::string &matcherType,
                      const std::string &selectorType, FeaturePerformance2D &perf
) {
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    auto t = StartTimer();

    if (matcherType == "MAT_BF") {
        int normType = descriptorType == "DES_BINARY" ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    } else if (matcherType == "MAT_FLANN") {
        if (descSource.type() != CV_32F) {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType == "SEL_NN") {
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        perf.numMatchesFinal = perf.numMatchesRaw = matches.size();
    } else if (selectorType == "SEL_KNN") {
        constexpr double filterThreshold = 0.8;
        constexpr int numKNNMatches = 2;
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, numKNNMatches);
        perf.numMatchesRaw = knnMatches.size();
        for (auto &matchedPairs : knnMatches) {
            if (matchedPairs.size() > 1) {
                if (matchedPairs[0].distance < filterThreshold * matchedPairs[1].distance) {
                    matches.push_back(matchedPairs[0]);
                }
            }
        }
        perf.numMatchesFinal = matches.size();
        std::cout << "[INFO] SEL_KNN filtered " << matches.size() << " / " << knnMatches.size() << std::endl;
    }
    perf.elapsedMatch = EndTimer(t);
}

void FilterKeypointsInRect(std::vector<cv::KeyPoint> &keypoints, cv::Rect &rect) {
    std::vector<cv::KeyPoint> keep;
    for (auto &kpt : keypoints) {
        if (rect.contains(kpt.pt)) {
            keep.push_back(kpt);
        }
    }
    keypoints = keep;
}

double StartTimer() {
    return (double) cv::getTickCount();
}

double EndTimer(const double start) {
    double elapsed = (((double) cv::getTickCount() - start) / cv::getTickFrequency()) * 1000.0;
    return elapsed;
}

double EndKeyPointTimer(const double start, std::vector<cv::KeyPoint> &keypoints, FeatureType ftype) {
    double t = ((double) cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << GetFeatureTypeName(ftype)
              << " detection with n=" << keypoints.size()
              << " keypoints in " << 1000 * t / 1.0
              << " ms" << endl;
    return t * 1000.0;
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, FeatureType detectorType,
                        FeaturePerformance2D &perf, bool bVis) {
    auto t = StartTimer();

    cv::Ptr<cv::FeatureDetector> detector;
    switch (detectorType) {
        case BRISK:
            detector = cv::BRISK::create();
            break;
        case FAST:
            detector = cv::FastFeatureDetector::create();
            break;
        case AKAZE:
            detector = cv::AKAZE::create();
            break;
        case SIFT:
            detector = cv::xfeatures2d::SIFT::create();
            break;
        case ORB:
            detector = cv::ORB::create();
            break;

        default:
            std::cout << "[WARNING] invalid keypoint detector type give, defaulting to SIFT" << std::endl;
            detector = cv::xfeatures2d::SIFT::create();
            break;
    }

    detector->detect(img, keypoints);

    perf.elapsedDetection = EndTimer(t);
    EndKeyPointTimer(t, keypoints, detectorType);

    if (bVis) {
        VisualizeKeypoints(img, keypoints, GetFeatureTypeName(detectorType));
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, FeatureType descriptorType,
                   FeaturePerformance2D &perf) {
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;

    // BRISK parameters
    constexpr int briskThreshold = 30;
    constexpr int briskOctaves = 3;
    constexpr float briskPatternScale = 1.0f;

    switch (descriptorType) {
        case BRISK:
            extractor = cv::BRISK::create(briskThreshold, briskOctaves, briskPatternScale);
            break;

        case ORB:
            extractor = cv::ORB::create();

        case BRIEF:
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
            break;

        case FREAK:
            extractor = cv::xfeatures2d::FREAK::create();
            break;

        case AKAZE:
            extractor = cv::AKAZE::create();
            break;

        case SIFT:
            extractor = cv::xfeatures2d::SIFT::create();
            break;

        default:
            cout << "WARNING: descriptor type given cannnot be used , default to SIFT" << endl;
            extractor = cv::xfeatures2d::SIFT::create();
            break;
    }

    // perform feature description
    auto t = StartTimer();
    extractor->compute(img, keypoints, descriptors);
    perf.elapsedDescCalc = EndTimer(t);
    std::cout << descriptorType << " descriptor extraction in " << perf.elapsedDescCalc << " ms" << std::endl;
}

void VisualizeKeypoints(cv::Mat &img, vector<cv::KeyPoint> &keypoints, const std::string &featureTypeName) {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = featureTypeName + " Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, FeaturePerformance2D &perf, bool bVis) {
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    auto t = StartTimer();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto &corner : corners) {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(corner.x, corner.y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    perf.elapsedDetection = EndTimer(t);
    EndKeyPointTimer(t, keypoints, SHITMOASI);

    // visualize results
    if (bVis) {
        VisualizeKeypoints(img, keypoints, GetFeatureTypeName(FeatureType::SHITMOASI));
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, FeaturePerformance2D &perf, bool bVis) {
    constexpr int blockSize = 2;
    constexpr int ksize = 3;
    constexpr double k = 0.04;
    constexpr float threshold = 150.0;

    auto t = StartTimer();

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, ksize, k);

    cv::Mat dst_norm;
    cv::Mat dst_norm_scaled;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            if (dst_norm.at<float>(i, j) > threshold) {
                keypoints.emplace_back(cv::Point2f(j, i), dst.at<float>(i, j));
            }
        }
    }
    perf.elapsedDetection = EndKeyPointTimer(t, keypoints, HARRIS);

    // visualize results
    if (bVis) {
        VisualizeKeypoints(img, keypoints, GetFeatureTypeName(FeatureType::HARRIS));
    }
}

std::string GetFeatureTypeName(FeatureType featureType) {
    std::string name;
    switch (featureType) {
        case SHITMOASI:
            return "SHITOMASI";
        case ORB:
            return "ORB";
        case FAST:
            return "FAST";
        case BRISK:
            return "BRISK";
        case FREAK:
            return "FREAK";
        case HARRIS:
            return "HARRIS";
        case SIFT:
            return "SIFT";
        case BRIEF:
            return "BRIEF";
        case AKAZE:
            return "AKAZE";
        default:
            return "UNKNOWN";
    }
}

FeatureType GetFeatureType(const std::string &name) {
    if (name == "SHITOMASI") {
        return SHITMOASI;
    }
    if (name == "ORB") {
        return ORB;
    }
    if (name == "BRISK") {
        return BRISK;
    }
    if (name == "FREAK") {
        return FREAK;
    }
    if (name == "SIFT") {
        return SIFT;
    }
    if (name == "BRIEF") {
        return BRIEF;
    }
    if (name == "HARRIS") {
        return HARRIS;
    }
    if (name == "AKAZE") {
        return AKAZE;
    }
    if (name == "FAST") {
        return FAST;
    }
    return FEATURE_TYPE_UNKNOWN;
}