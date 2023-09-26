/** @file imageProcessing.cpp
 *
 *  Copyright © 2022 Oregon State University
 *
 *  Dominic W. Daprano
 *  Sheng Tse Tsai
 *  Moritz S. Schmid
 *  Christopher M. Sullivan
 *  Robert K. Cowen
 *
 *  Hatfield Marine Science Center
 *  Center for Qualitative Life Sciences
 *  Oregon State University
 *  Corvallis, OR 97331
 *
 *  This program is distributed WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 *  This program is distributed under the GNU GPL v 2.0 or later license.
 *
 *  Any User wishing to make commercial use of the Software must contact the authors
 *  or Oregon State University directly to arrange an appropriate license.
 *  Commercial use includes (1) use of the software for commercial purposes, including
 *  integrating or incorporating all or part of the source code into a product
 *  for sale or license by, or on behalf of, User to third parties, or (2) distribution
 *  of the binary or source code to third parties for use with a commercial
 *  product sold or licensed by, or on behalf of, User.
 *
 */

#include "imageProcessing.hpp"
#include <iostream>
#include <fstream> // write output csv files
#include <iomanip> // std::setw
#include <filesystem>

#include <opencv2/videoio.hpp> // used for the video preprocessing
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp> // simpleBlobDector
#include <opencv2/objdetect.hpp>  // groupRectangles
#include <opencv2/imgproc.hpp>    // gaussian blur

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <chrono> // timer

namespace fs = std::filesystem;

bool containExt(const std::string s, std::string arr[], int len)
{
    for (int i = 0; i < len; i++)
    {
        if (s == arr[i])
        {
            return true;
        }
    }
    return false;
}

bool isInt(std::string str)
{
    for (int i = 0; i < str.length(); i++)
    {
        if (!isdigit(str[i]))
            return false;
    }
    return true;
}

std::string convertInt(int number, int fill)
{
    std::stringstream ss;                                 // create a stringstream
    ss << std::setw(fill) << std::setfill('0') << number; // add number to the stream

    return ss.str(); // return a string with the contents of the stream
}

void getFrame(cv::VideoCapture cap, cv::Mat &img)
{
    cv::Mat frame;
    cap.read(frame);
    cv::cvtColor(frame, img, cv::COLOR_RGB2GRAY);
    frame.release();
}

void segmentImage(const cv::Mat &img, cv::Mat &imgCorrect, std::vector<cv::Rect> &bboxes, std::string imgDir, std::string imgName, std::ofstream &framePtr, Options options)
{
    // Flatfield the image to remove the vertical lines
    int effective_width = flatField(img, imgCorrect, options.outlierPercent);

    // If the SNR is less than options.signalToNoise then the image will have many false segments
    float imgSNR = SNR(imgCorrect);

    if (imgSNR > options.signalToNoise)
    {
        cv::Mat imgPreprocess;
        preprocess(imgCorrect, imgPreprocess, 1);
        mser(imgPreprocess, bboxes, options.minArea, options.maxArea, options.delta, options.variation, options.epsilon);
    }
    else if (imgSNR > options.signalToNoise * .75)
    {

        // Create a mask that includes all of the regions of the image with
        // the darkest pixels which MSER method can be performed on.
        cv::Mat imgPreprocess;
        preprocess(imgCorrect, imgPreprocess, 8);

        cv::Mat imgThresh;
        cv::threshold(imgPreprocess, imgThresh, options.threshold, 255, cv::THRESH_BINARY);

        cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
        cv::Mat imgCorrectMask(img.size(), img.type(), cv::Scalar(255));

        std::vector<std::vector<cv::Point>> contours; // Vector for storing contour
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(imgThresh, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_TC89_L1);

        // create mask based on darkest regions
        cv::Rect maskRect(0, 0, mask.cols, mask.rows); // use maskRect to make sure box doesn't go off the edge
        for (int i = 0; i < contours.size(); i++)
        {
            cv::Rect boundRect = cv::boundingRect(contours[i]);
            if (boundRect.area() > options.maxArea)
                continue;

            float scaleFactor = 1.2;
            cv::Rect largeRect = rescaleRect(boundRect, scaleFactor);
            cv::Rect intersectRect = boundRect & maskRect; // FIXME: remove & - really slow
            cv::Mat roi = mask(intersectRect);

            roi.setTo(255);
        }

        imgCorrect.copyTo(imgCorrectMask, mask);
        mser(imgCorrectMask, bboxes, options.minArea, options.maxArea, options.delta, options.variation, options.epsilon);
    }
    else
    {
        contourBbox(imgCorrect, bboxes, 90, options.minArea, options.maxArea);
    }

// Write details about the segmentation for this file.
#pragma omp critical(write)
    {
        framePtr << imgName << ", " << imgSNR << ", " << effective_width << std::endl;
    }
}

void contourBbox(const cv::Mat &img, std::vector<cv::Rect> &bboxes, int threshold, int minArea, int maxArea)
{
    cv::Mat imgPreprocess;
    preprocess(img, imgPreprocess, 8);

    cv::Mat imgThresh;
    cv::threshold(imgPreprocess, imgThresh, threshold, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours; // Vector for storing contour
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imgThresh, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_TC89_L1);

    // create mask based on darkest regions
    for (int i = 0; i < contours.size(); i++)
    {
        cv::Rect bbox = cv::boundingRect(contours[i]);
        if (bbox.area() < minArea || bbox.area() > maxArea)
            continue;

        bboxes.push_back(bbox);
    }
}

void saveCrops(const cv::Mat &img, const cv::Mat &imgCorrect, std::vector<cv::Rect> &bboxes, std::string imgDir, std::string imgName, std::ofstream &measurePtr, Options options)
{
    cv::Rect imgRect(0, 0, imgCorrect.cols, imgCorrect.rows); // use imgRect to make sure box doesn't go off the edge

    // Create crop directories
    std::string correctCropDir = imgDir + "/corrected_crop";
    std::string frameDir = imgDir + "/frame/";

    fs::create_directory(correctCropDir);
    if (options.fullOutput)
    {
        fs::create_directory(frameDir);
    }

    // Save image with bounding boxes
    cv::Mat imgBboxes;
    cv::cvtColor(imgCorrect, imgBboxes, cv::COLOR_GRAY2RGB);

    int numBboxes = bboxes.size();

    for (int k = 0; k < numBboxes; k++)
    {
        // Get measurement data
        float area = bboxes[k].area();

        // Determine if the bbox is too large or small
        if (area < options.minArea || area > options.maxArea)
            continue;

        float perimeter = bboxes[k].height + bboxes[k].width * 2.0;
        float x = bboxes[k].x;
        float y = bboxes[k].y;
        float major;
        float minor;
        float height = bboxes[k].height;
        float width = bboxes[k].width;

        // Get mean pixel value for the crop
        cv::Mat imgCropUnscaled = cv::Mat(imgCorrect, bboxes[k] & imgRect);
        double mean = cv::sum(imgCropUnscaled)[0] / (height * width);

        // Determine if box is irregularly shapped (Abnormally long and thin)
        int hwRatio = 10;
        if (width < 30 && height > hwRatio * width)
            continue;

        if (height > width)
        {
            major = height;
            minor = width;
        }
        else
        {
            major = width;
            minor = height;
        }

        // Re-scale the crop of the image after getting the measurement data written to a file
        cv::Rect scaledBbox = rescaleRect(bboxes[k], 1.2);

        // Create a new crop using the intersection of rectangle objects and the image
        std::string correctImgFile = correctCropDir + "/" + imgName + "_" + "crop_" + convertInt(k) + ".png";
        cv::Mat imgCropCorrect = cv::Mat(imgCorrect, scaledBbox & imgRect);
        cv::imwrite(correctImgFile, imgCropCorrect);

        // Draw the cropped frames on the image to be saved
        cv::rectangle(imgBboxes, bboxes[k], cv::Scalar(0, 0, 255));
        cv::rectangle(imgBboxes, scaledBbox, cv::Scalar(255, 0, 0));

// Write the image data to the measurement file
// Format: img,area,major,minor,perimeter,x,y,mean,height
#pragma omp critical(write)
        {
            measurePtr << correctImgFile << ","
                       << area << ","
                       << major << ","
                       << minor << ","
                       << perimeter << ","
                       << x << ","
                       << y << ","
                       << mean << ","
                       << height << std::endl;
        }
    }

    // Write full video frames to files
    if (options.fullOutput)
    {
        std::string correctedFrame = frameDir + "/" + imgName + "_corrected.png";
        std::string originalFrame = frameDir + "/" + imgName + "_original.png";
        std::string bboxFrame = frameDir + "/" + imgName + "_bboxes.png";

        cv::imwrite(correctedFrame, imgCorrect);
        cv::imwrite(originalFrame, img);
        cv::imwrite(bboxFrame, imgBboxes);
    }
}

cv::Rect rescaleRect(const cv::Rect &rect, float scale)
{
    float scaleWidth = rect.width * scale - rect.width;
    float scaleHeight = rect.height * scale - rect.height;
    cv::Rect scaledRect(rect.x - scaleWidth / 2, rect.y - scaleHeight / 2,
                        rect.width + scaleWidth, rect.height + scaleHeight);

    return scaledRect;
}

void groupRect(std::vector<cv::Rect> &rectList, int groupThreshold, double eps)
{
    if (rectList.empty())
    {
        return;
    }

    // Third argument of partion is a predicate operator that looks for a method of the class
    // that will return true when elements are apart of the same partition
    std::vector<int> labels;
    int nclasses = partition(rectList, labels, OverlapRects(eps));
    // int nclasses = partition(rectList, labels, cv::SimilarRects(eps));

    // labels correspond to the location of the rectangle in space
    std::vector<cv::Rect> rrects(nclasses);
    std::vector<int> rweights(nclasses, 0);
    int nlabels = (int)labels.size();
    for (int i = 0; i < nlabels; i++)
    {
        int cls = labels[i];
        rrects[cls].width = rectList[i].width;
        rrects[cls].height = rectList[i].height;
        rrects[cls].x = rectList[i].x;
        rrects[cls].y = rectList[i].y;
    }

    for (int i = 0; i < nlabels; i++)
    {
        int cls = labels[i];
        if (rectList[i].width > rrects[cls].width)
        {
            rrects[cls].width = rectList[i].width;
            rrects[cls].x = rectList[i].x;
        }
        if (rectList[i].height > rrects[cls].height)
        {
            rrects[cls].height = rectList[i].height;
            rrects[cls].y = rectList[i].y;
        }
        rweights[cls]++;
    }

    rectList.clear();

    for (int i = 0; i < nclasses; i++)
    {
        if (rweights[i] >= groupThreshold)
        {
            rectList.push_back(rrects[i]);
        }
    }
}

void mser(const cv::Mat &img, std::vector<cv::Rect> &bboxes, int minArea, int maxArea, int delta, int maxVariation, float eps)
{
    cv::Ptr<cv::MSER> detector = cv::MSER::create(delta, minArea, maxArea, maxVariation);
    std::vector<std::vector<cv::Point>> msers;
    detector->detectRegions(img, msers, bboxes);

    // merge the bounding boxes produced by MSER
    int minBboxes = 2;
    groupRect(bboxes, minBboxes, eps);
}

void preprocess(const cv::Mat &src, cv::Mat &dst, float erosion_size)
{
    // Perform image pre processing
    cv::Mat erodeElement = getStructuringElement(cv::MORPH_ERODE, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size));

    cv::morphologyEx(src, dst, cv::MORPH_OPEN, erodeElement); // open is a combination of erosion and dialation
}

float SNR(const cv::Mat &img)
{
    // perform histogram equalization
    cv::Mat imgHeq;
    cv::equalizeHist(img, imgHeq);

    // Calculate Signal To Noise Ratio (SNR)
    cv::Mat imgClean, imgNoise;
    cv::medianBlur(imgHeq, imgClean, 3);
    imgNoise = imgHeq - imgClean;
    double SNR = 20 * (cv::log(cv::norm(imgClean, cv::NORM_L2) / cv::norm(imgNoise, cv::NORM_L2)));

    return SNR;
}

int flatField(const cv::Mat &src, cv::Mat &dst, float percent)
{
    //cv::Mat imgBlack = cv::Mat::zeros(src.size(), src.type());
    cv::Mat imgCalib = cv::Mat::zeros(src.size(), src.type());

    // Get the calibration image
    int width = trimMean(src, imgCalib, percent);

    //cv::Mat imgCorrect(src.size(), src.type());    // creates mat of the correct size and type
    //cv::addWeighted(src, 1, imgBlack, -1, 0, src); // subtracts an all black array
    //cv::addWeighted(imgCalib, 1, imgBlack, -1, 0, imgCalib);
    cv::divide(src, imgCalib, dst, 255); // performs the flat fielding by dividing the arrays

    return width;
}

int fillSides(cv::Mat &img, int left, int right, int fill)
{
    if (left + right > img.cols)
    {
        std::cerr << "Error: The size of the left and right crops are larger then the image." << std::endl;
        return 1;
    }

    // Fill the sides with pixel "fill"
    for (int i = 0; i < left; i++)
    {
        img.col(i).setTo(fill);
    }
    for (int i = img.cols - right; i < img.cols; i++)
    {
        img.col(i).setTo(fill);
    }

    return 0;
}

int trimMean(const cv::Mat &img, cv::Mat &tMean, float percent)
{
    cv::Mat sort;
    cv::sort(img, sort, cv::SORT_EVERY_COLUMN);
    int height = img.rows, width = img.cols;

    // Get a subset of the matrix entries so that they can be averaged
    int k = round(img.rows * percent / 2); // Calculate the number of outlier elements

    // Create a mask with 0's for the top and bottom k elements
    cv::Mat maskCol = cv::Mat::ones(height, 1, CV_8UC1);
    for (int cnt1 = 0; cnt1 < k; cnt1++)
    {
        maskCol.at<int8_t>(cnt1, 0) = 0;
    }
    for (int cnt1 = (height - k); cnt1 < height; cnt1++)
    {
        maskCol.at<int8_t>(cnt1, 0) = 0;
    }
    cv::Mat mask;
    cv::repeat(maskCol, 1, width, mask);

    cv::Mat imgMask;
    sort.copyTo(imgMask, mask);

    // get the column-wise average of the image.
    cv::Mat average;
    cv::reduce(imgMask, average, 0, cv::REDUCE_AVG);

    int val = 0;
    for (int i = 0; i < width; i++) {
        if (average.at<int8_t>(0, i) <= 10)
        {
            val += 1;
        }
    }

    // Create the trimmed mean matrix
    cv::repeat(average, height, 1, tMean);

    return val;
}
