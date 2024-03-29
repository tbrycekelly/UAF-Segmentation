/** @file main.cpp
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

#include <iostream>
#include <fstream> // write output csv files
#include <iomanip>  // For the function std::setw

#include <filesystem>
#include <ctime>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "imageProcessing.hpp"

#if defined(WITH_OPENMP) and !defined(WITH_VISUAL)
    #include "omp.h" // OpenMP
#endif

namespace fs = std::filesystem;

void helpMsg(std::string executable, Options options) {
    std::cout << "Usage: " << executable << " -i <input> [OPTIONS]\n"
        << std::left << std::setw(30) << "Segment a directory of images by utilizing the MSER method.\n\n"
        << std::left << std::setw(30) << "  -i, --input" << "Directory of video files to segment\n"
        << std::left << std::setw(30) << "  -o, --output-directory" << "Output directory where segmented images should be stored (Default: " << options.outputDirectory << ")\n"
        << std::left << std::setw(30) << "  -s, --signal-to-noise" << "The cutoff signal to noise ratio that is used in determining which frames from\n"
        << std::left << std::setw(30) << "" << "the video file get segmented. Note: This will change as we change the outlier percent (Default: " << options.signalToNoise << ")\n"
        << std::left << std::setw(30) << "  -p, --outlier-percent" << "Percentage of darkest and lightest pixels to throw out before flat-fielding (Default: " << options.outlierPercent << ")\n"
        << std::left << std::setw(30) << "  -M, --maxArea" << "Maximum area of a segmented blob (Default: " << options.maxArea << ")\n"
        << std::left << std::setw(30) << "  -m, --minArea" << "Minimum area of a segmented blob. (Default: " << options.minArea << ")\n"
        << std::left << std::setw(30) << "  -d, --delta" << "Delta is a parameter for MSER. Delta is the number of steps (changes\n"
        << std::left << std::setw(30) << "" << "in pixel brightness) MSER uses to compare the size of connected regions.\n"
        << std::left << std::setw(30) << "" <<  "A smaller delta will produce more segments. (Default: " << options.delta << ")\n"
        << std::left << std::setw(30) << "  -v, --variation" << "Maximum variation of the region's area between delta threshold.\n"
        << std::left << std::setw(30) << "" <<  "Larger values lead to more segments. (Default: " << options.variation << ")\n"
        << std::left << std::setw(30) << "  -e, --epsilon" << "Float between 0 and 1 that represents the maximum overlap between\n"
        << std::left << std::setw(30) << "" << "two rectangle bounding boxes. 0 means that any overlap will mean\n"
        << std::left << std::setw(30) << "" << "that the bounding boxes are treated as the same. (Default: " << options.epsilon << ")\n"
        << std::left << std::setw(30) << "  -t, --threshold" << "Value to threshold the images for low signal to noise images \n"
        << std::left << std::setw(30) << "" <<  "(Default: " << options.threshold << ")\n"
        << std::left << std::setw(30) << "  -f, --full-ouput" << "If flag is included a directory of full frames is added to output\n"
        << std::left << std::setw(30) << "  -l, --left-crop" << "Crop this many pixels off of the left side of the image\n"
        << std::left << std::setw(30) << "  -r, --right-crop" << "Crop this many pixels off of the right side of the image" << std::endl;
}

int main(int argc, char **argv) {
    // Set the default options
    Options options;
    options.input = "";
    options.outputDirectory = "out";
    options.signalToNoise = 60;
    options.outlierPercent = .15;
    options.minArea = 50;
    options.maxArea = 400000;
    options.epsilon = 1;
    options.delta = 4;
    options.variation = 100;
    options.threshold = 160;
    options.fullOutput = false;
    options.left = 0;
    options.right = 0;

    std::cout << "Segmentation Version 2023.10.03" << std::endl;

    // TODO: more robust options with std::find may be worth it
    if (argc == 1) {
        helpMsg(argv[0], options);
        // Print the number of threads that will be used by this program
        #pragma omp parallel
        {
            #pragma omp single
            {
                #if defined(WITH_OPENMP)
                int nthreads = omp_get_num_threads();
                std::cout << "OMP Num Threads: " << nthreads << std::endl;
                #endif
            }
        }
    }

    int i = 1;
	while (i < argc) {
        // Display the help message
		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            helpMsg(argv[0], options);
            return 0;
        }
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            options.input = argv[i + 1];
            if ( !fs::exists(options.input) ) {
                std::cerr << options.input << " does not exist." << std::endl;
                return 1;
            }
            i+=2;
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output-directory") == 0) {
            options.outputDirectory = argv[i + 1];
            try {
                fs::create_directories(options.outputDirectory);
                fs::permissions(options.outputDirectory, fs::perms::all);
            }
            catch(fs::filesystem_error const& ex){
                std::cerr << ex.what() <<  std::endl;
                return 1;
            }
            i+=2;
		} else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--signal-to-noise") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                std::cerr << argv[i+1] << " is not a valid input. Signal To Noise ration must be an integer." << std::endl;
                return 1;
            }
            options.signalToNoise = std::stoi(argv[i+1]);
            i+=2;
		} else if (strcmp(argv[i], "-M") == 0 || strcmp(argv[i], "--maxArea") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                std::cerr << argv[i+1] << " is not a valid input. Maximum must be an integer." << std::endl;
                return 1;
            }
            options.maxArea = std::stoi(argv[i+1]);
            i+=2;
		} else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--minArea") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                std::cerr << argv[i+1] << " is not a valid input. Minimum must be an integer." << std::endl;
                return 1;
            }
            options.minArea = std::stoi(argv[i+1]);
            i+=2;
		} else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--epsilon") == 0) {
            // Validate the input type
            options.epsilon = std::stof(argv[i+1]); // FIXME: may throw error if not int

            if (options.epsilon < 0) {
                std::cerr << options.epsilon << " is not a valid input. Epsilon must be a non-negative float." << std::endl;
                return 1;
            }
            i+=2;
		} else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--delta") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                std::cerr << argv[i+1] << " is not a valid input. Delta must be an integer." << std::endl;
                return 1;
            }
            options.delta = std::stoi(argv[i+1]);
            i+=2;

		} else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--variation") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                std::cerr << argv[i+1] << " is not a valid input. Variation must be an integer." << std::endl;
                return 1;
            }
            options.variation = std::stoi(argv[i+1]);
            i+=2;
		} else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threshold") == 0) {
            // Validate the input type
            options.threshold = std::stof(argv[i+1]); // FIXME: may throw error if not int

            if (options.threshold < 0 && options.threshold > 255) {
                std::cerr << options.threshold << " is not a valid input. Threshold must be between 0 and 255 inclusive" << std::endl;
                return 1;
            }
            i+=2;
		} else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--outlier-percent") == 0) {
            // Validate the input type
            options.outlierPercent = std::stof(argv[i+1]); // FIXME: may throw error if not int

            if (options.outlierPercent > 1 or options.outlierPercent < 0) {
                std::cerr << options.outlierPercent << " is not a valid input. Outlier percent must be a float between 0 and 1." << std::endl;
                return 1;
            }
            i+=2;
		} else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--full-output") == 0) {
            // If flag exists then add output
            options.fullOutput = true;

            i+=1;
		} else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--left-crop") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                std::cerr << argv[i+1] << " is not a valid input. Left crop must be a positive integer." << std::endl;
                return 1;
            } else if (std::stoi(argv[i+1]) < 0) {
                std::cerr << argv[i+1] << " is not a valid input. Left crop must be a positive integer." << std::endl;
                return 1;
            }

            options.left = std::stoi(argv[i+1]);
            i+=2;
		} else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--right-crop") == 0) {
            // Validate the input type
            if ( !isInt(argv[i+1]) ) {
                std::cerr << argv[i+1] << " is not a valid input. Right crop must be a positive integer." << std::endl;
                return 1;
            } else if (std::stoi(argv[i+1]) < 0) {
                std::cerr << argv[i+1] << " is not a valid input. Right crop must be a positive integer." << std::endl;
                return 1;
            }
            options.right = std::stoi(argv[i+1]);
            i+=2;
		} else {
            // Display invalid option message
            std::cerr << argv[0] << ": invalid option \'" << argv[i] << "\'" <<
                "\nTry \'" << argv[0] << " --help\' for more information." << std::endl;

            return 1;
            i+=2;
		}
	}
    if ( options.input == "" ) {
        std::cerr << "Must have either an input directory or an input file for segmentation to be run on" << std::endl;
        return 1;
    }


    auto start = std::chrono::system_clock::now();

    // TBK: Add details to log
    std::cout << "Input Directory: " << options.input << std::endl;
    std::cout << "Output Directory: " << options.outputDirectory << std::endl;
    std::cout << "SNR: " << options.signalToNoise << std::endl;
    std::cout << "Outlier Percentage: " << options.outlierPercent << std::endl;
    std::cout << "Min Area: " << options.minArea << std::endl;
    std::cout << "Max Area: " << options.maxArea << std::endl;
    std::cout << "Epsilon: " << options.epsilon << std::endl;
    std::cout << "Delta: " << options.delta << std::endl;
	std::cout << "Variation: " << options.variation << std::endl;
	std::cout << "Threshold: " << options.threshold << std::endl;
	std::cout << "Full Output: " << options.fullOutput << std::endl;


    // create output directories
    fs::create_directory(options.outputDirectory);
    fs::permissions(options.outputDirectory, fs::perms::all);

	std::string measureDir = options.outputDirectory + "/measurements";
    fs::create_directory(measureDir);
    fs::permissions(measureDir, fs::perms::all);

	std::string segmentDir = options.outputDirectory + "/segmentation";
    fs::create_directory(segmentDir);
    fs::permissions(segmentDir, fs::perms::all);

    std::vector<fs::path> files;
    if (fs::is_directory(options.input)) {
        for(auto& p: fs::directory_iterator{options.input}) {
            fs::path file(p);

            std::string ext = file.extension();

            std::string valid_ext[] = {".avi", ".mp4"};
            int len = sizeof(valid_ext)/sizeof(valid_ext[0]);
            if (!containExt(ext, valid_ext, len)) {
                continue;
            }
            files.push_back(file);
        }
    }
    else {
        files.push_back(fs::path(options.input));
    }

    int numFiles = files.size();
    if (numFiles < 1 ) {
        std::cerr << "No files to segment were found." << std::endl;
        return 1;
    }

    std::cout << "Found " << numFiles << " files." << std::endl;

    for (int i=0; i<numFiles; i++) {
        std::cout << "Starting file " << i + 1 << "..." << std::endl;
        fs::path file = files[i];
        std::string fileName = file.stem();

        // Create a measurement file to save crop info to
        std::string measureFile = measureDir + "/" + fileName + ".csv";
        std::ofstream measurePtr(measureFile);
        measurePtr << "image,area,major,minor,perimeter,x,y,mean,height" << std::endl;

        // Create a metadata file to save frame info into
        std::string frameFile = measureDir + "/" + fileName + ".meta.csv";
        std::ofstream framePtr(frameFile);
        framePtr << "frame, snr, valid_width" << std::endl;

        cv::VideoCapture cap(file.string());
        if (!cap.isOpened()) {
            std::cerr << "Invalid file: " << file.string() << std::endl;
            continue;
        }

	    int image_stack_counter = 0;
        int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
		std::cout << "Found video file with " << totalFrames << " frames." << std::endl; // TBK
        int elementCount = 0;

        #pragma omp parallel for
        for (int j=0; j<totalFrames; j++) {
            cv::Mat imgGray;
            #pragma omp critical(getImage)
            {
                getFrame(cap, imgGray);
                image_stack_counter += 1;
            }

            
			std::string imgName = fileName + "_" + convertInt(image_stack_counter, 4);
            std::string imgDir = segmentDir + "/" + fileName;
            fs::create_directories(imgDir);
            fs::permissions(imgDir, fs::perms::all);

            int fill = fillSides(imgGray, options.left, options.right);
            if (fill != 0) {
                exit(1);
            }

            cv::Mat imgCorrect;
            std::vector<cv::Rect> bboxes;
            segmentImage(imgGray, imgCorrect, bboxes, imgDir, imgName, framePtr, options);
            saveCrops(imgGray, imgCorrect, bboxes, imgDir, imgName, measurePtr, options);
            elementCount += bboxes.size();

            imgGray.release();
            imgCorrect.release();
	    }

	    // When video is done being processed release the capture object
	    cap.release();
        std::cout << "Done with file. Found " << elementCount << " ROIs." << std::endl; // TBK
        measurePtr.close();
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
 
    std::cout << "Finished segmentation at " << std::ctime(&end_time) << std::endl << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    return 0;
}
