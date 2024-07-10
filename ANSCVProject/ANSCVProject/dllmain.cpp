// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "ANSCV.h"
BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

extern "C" ANSCV_API int	CreateANSFireDetectorHandle(ANSCENTER::ANSFireDetector * *Handle, double HSVTheshold) {
    (*Handle) = new ANSCENTER::ANSFireDetector();
    if (*Handle == NULL) return 0;
    else {
        bool result = (*Handle)->Initialise(HSVTheshold);
        if (result) return 1;
        else return 0;
    }
}
extern "C" ANSCV_API int    ReleaseANSFireDetectorHandle(ANSCENTER::ANSFireDetector * *Handle) {
    (*Handle)->Destroy();
    delete* Handle;
    if (*Handle == NULL) return 0;
    else return 1;
}
extern "C" ANSCV_API int	RunDetector(ANSCENTER::ANSFireDetector * *Handle, unsigned char* jpeg_string, unsigned int bufferLength, std::string & result) {
    cv::Mat frame = cv::imdecode(cv::Mat(1, bufferLength, CV_8UC1, jpeg_string), cv::IMREAD_COLOR);
    if (frame.empty()) { 
        result = "";
        return 0;
    }
    std::vector<ANSCENTER::Object> outputs = (*Handle)->Detect(frame);
    result= ANSCENTER::ANSFireDetector::ConvertResultsToString(&outputs);
    return 1;

}
extern "C" ANSCV_API int	RunDetectorBinary(ANSCENTER::ANSFireDetector * *Handle, unsigned char* jpeg_bytes, unsigned int width, unsigned int height, std::string & result) {
    cv::Mat frame = cv::Mat(height, width, CV_8UC3, jpeg_bytes).clone(); // make a copy
    if (frame.empty()) {
        result = "";
        return 0;
    }
    std::vector<ANSCENTER::Object> outputs = (*Handle)->Detect(frame);
    result = ANSCENTER::ANSFireDetector::ConvertResultsToString(&outputs);

    /*
    NOTE:
        - Delete later in production. This is only for model training purposes.
    */
    (*Handle)->WriteResultsToFileCSV("results.csv", &outputs, std::ios::app | std::ios::out);
    return 1;
}
extern "C" ANSCV_API int	RunDetectorImagePath(ANSCENTER::ANSFireDetector * *Handle, const char* imageFilePath, std::string & result) {
    std::string stImageFileName(imageFilePath);
    cv::Mat frame = cv::imread(stImageFileName, cv::ImreadModes::IMREAD_COLOR);
    if (frame.empty()) {
        result = "";
        return 0;
    }
    std::vector<ANSCENTER::Object> outputs = (*Handle)->Detect(frame);
    result = ANSCENTER::ANSFireDetector::ConvertResultsToString(&outputs);
    return 1;
}