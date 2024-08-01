#include "ANSCV.h"
#include <iostream>
#include <filesystem>
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/foreach.hpp"
#include "boost/optional.hpp"

template<typename T>
T GetOptionalValue(const boost::property_tree::ptree& pt, std::string attribute, T defaultValue) {
    if (pt.count(attribute)) {
        return pt.get<T>(attribute);
    }
    return defaultValue;
}

template <typename T>
T GetData(const boost::property_tree::ptree& pt, const std::string& key)
{
    T ret;
    if (boost::optional<T> data = pt.get_optional<T>(key))
    {
        ret = data.get();
    }
    else
    {
        throw std::runtime_error("Could not read the data from ptree: [key: " + key + "]");
    }
    return ret;
}
unsigned char* CVMatToBytes(cv::Mat image, unsigned int& bufferLengh)
{
    int size = image.total() * image.elemSize();
    unsigned char* bytes = new unsigned char[size];  // you will have to delete[] that later
    std::memcpy(bytes, image.data, size * sizeof(unsigned char));
    bufferLengh = size * sizeof(unsigned char);
    return bytes;
}
int FireDetectionTest() {
    boost::property_tree::ptree root;
    boost::property_tree::ptree detectionObjects;
    boost::property_tree::ptree pt;
   
    std::string videoFilePath = "C:\\Projects\\ANSCV\\Resources\\fire_4.mp4";
    std::string result;
    ANSCENTER::ANSFireDetector* infHandle;
    int createResult = CreateANSFireDetectorHandle(&infHandle,0.6);

    cv::VideoCapture capture(videoFilePath);
    if (!capture.isOpened()) {
        printf("could not read this video file...\n");
        return -1;
    }

    /*
    NOTE:
        - Variables to measure average processing time over the duration of the 
        test video.
    */
    long double total = 0;
    long int counter = 0;

    while (true)
    {
        counter++;

        cv::Mat frame;
        if (!capture.read(frame)) // if not success, break loop
        {
            std::cout << "\n Cannot read the video file. please check your video.\n";
            break;
        }
        unsigned int bufferLength = 0;
        unsigned char* jpeg_string = CVMatToBytes(frame, bufferLength);
        int height = frame.rows;
        int width = frame.cols;
        auto start = std::chrono::system_clock::now();
        int detectionResult= RunDetectorBinary(&infHandle, jpeg_string, width, height, result);

        delete jpeg_string;
        if (!result.empty()) {
            pt.clear();
            std::stringstream ss;
            ss.clear();
            ss << result;
            boost::property_tree::read_json(ss, pt);
            BOOST_FOREACH(const boost::property_tree::ptree::value_type & child, pt.get_child("results"))
            {
                const boost::property_tree::ptree& result = child.second;
                /*
                NOTE:
                    - Try catch error when nothing is stored in json string
                */
                try {
                    auto class_id = GetData<int>(result, "classId");
                    auto class_name = GetData<std::string>(result, "className");
                }
                catch (const std::exception& ex) {
                    /*
                    NOTE:
                        - Commenting out code to `stdout` and `stderr` to correctly
                        measure performance.
                    */
                    //std::cerr << "Error: " << ex.what() << std::endl;
                }

            }
            /*
            NOTE:
                - Commenting out code to `stdout` and `stderr` to correctly
                measure performance.
            */
            //std::cout << "Result=" << result;
        }
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        total += elapsed.count();

        /*
        NOTE:
            - Commenting out code to `stdout` and `stderr` to correctly
            measure performance.
        */
        //printf("Time = %lld ms\n", static_cast<long long int>(elapsed.count()));

        cv::imshow("ANS Fire Detector", *(infHandle->GetMovingContouredFrame()));
        if (cv::waitKey(30) == 27) // Wait for 'esc' key press to exit
        {
            break;
        }
    }
    capture.release();
    cv::destroyAllWindows();
    ReleaseANSFireDetectorHandle(&infHandle);

    std::cout << "Average time per frame: " << (float) total / counter << std::endl;
    std::cout << "End of program.\n";
    return 0;
}
int main()
{
    return FireDetectionTest();
}

