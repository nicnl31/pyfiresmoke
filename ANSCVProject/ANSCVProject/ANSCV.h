#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#ifndef ANSCV_H
#define ANSCV_H
#define ANSCV_API __declspec(dllexport)

#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/*
NOTE:
	- Including the boost library. Must specify the include directory in the project
	configuration for this reference to work.
*/
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/optional.hpp>
#include <boost/format.hpp>

/*
NOTE:
	- Including the movement detection library. Must specify the include directory
	in the project configuration for this reference to work.
*/
#include <MoveDetect.h>

namespace pt = boost::property_tree;

namespace ANSCENTER {
	/*
	PURPOSE:
		- To make code more readable.
	*/
	enum DetectionType {
		FIRE, SMOKE, NONE
	};

	/*
	PURPOSE:
		- The `Object` struct is used to keep track of the information regarding the
		object of interest. In this case, the region of interest indicates smoke or
		fire.
	*/
	struct Object
	{
		int classId{ 0 };			/* Fire is of value 0, smoke is of value 1, and default or unident-
										ified is of value 2. Enumeration is used. */
		std::string className{};	/* Value is set to either "Smoke" or "Fire" or "None" */
		cv::Rect box{};				/* Region of interest ROI (bounding box) or polygon*/
		long unsigned int counter;	/* Number of frames from the first frame (offset), 0-based index */
	};

	class ANSCV_API ANSFireDetector {
	private:
		double _hsvThreshold;
		MoveDetect::Handler _handler;
		long unsigned int _counter;

		/*
		PURPOSE:
			- Returns a bounding box from the `contour` argument provided by the caller.
		*/
		cv::Rect BoundingBoxFromContour(std::vector<cv::Point> contour);

		/*
		PURPOSE:
			- Filters the HSV threshold of fire (hard-coded values inside the func-
			tion). This function returns a Boolean that denotes whether the fire col-
			our is detected given the threshold. It also changes the frame argument
			by masking it.

		ARGUMENTS:
			- The `area_threshold` argument is the percentage of the total area that
			contains the fire colour.
		*/
		bool FireColourInFrame(cv::Mat* frame, float area_threshold = 0.5);

		/*
		PURPOSE:
			- Filters the HSV threshold of smoke (hard-coded values inside the func-
			tion). This function returns a Boolean that denotes whether the smoke col-
			our is detected given the threshold. It also changes the frame argument
			by masking it.

		ARGUMENTS:
			- The `area_threshold` argument is the percentage of the total area that
			contains the smoke colour.
		*/
		bool SmokeColourInFrame(cv::Mat* frame, float area_threshold = 0.4);

		/*
		PURPOSE:
			- Initializes return objects based on the specifications written above.
		*/
		void InitializeObject(Object* object);

	public:
		/*
		PURPOSE:
			- Constructors and destructors.
		*/
		ANSFireDetector();
		~ANSFireDetector();

		/*
		PURPOSE:
			- The `Initialise()` function is basically a setter. The `Destroy()`
			methods might be to deallocate any dynamic memory.
		*/
		bool Initialise(double HSVThreshold);
		bool Destroy();

		/*
		PURPOSE:
			- Detects fire/smoke. The method returns a list of objects (regions of int-
			erest) that might indicate fire/smoke. 
		*/
		std::vector<Object> Detect(cv::Mat& input);

		/*
		PURPOSE:
			- Returns the pointer to the `cv::Mat` object inside the handler with
			the boxed/contoured frame.
		*/
		cv::Mat* GetMovingContouredFrame();

		/*
		PURPOSE:
			- Uses the JSON library (JSON boost) to convert the objects from
			`Object` to `std::string`.
		*/
		static std::string ConvertResultsToString(std::vector<Object> *objects);

		/*
		PURPOSE:
			- Outputs to a `.csv` text file. The default action is to append the 
			output to the file specified by the `path`. This function is temporary
			and should be deleted in production. This is only for model training or
			testing and should be deleted later.
		*/
		bool WriteResultsToFileCSV(std::string path, std::vector<Object>* objects, 
			std::ios_base::openmode mode);
		
	};

	/*
	MAYBE:
		- Other classes.
	*/
}

// API interfaces (external application can call)
// Fire Detector APIs
extern "C" ANSCV_API int	CreateANSFireDetectorHandle(ANSCENTER::ANSFireDetector * *Handle, double HSVTheshold);
extern "C" ANSCV_API int    ReleaseANSFireDetectorHandle(ANSCENTER::ANSFireDetector * *Handle);
extern "C" ANSCV_API int	RunDetector(ANSCENTER::ANSFireDetector * *Handle, unsigned char* jpeg_string, unsigned int bufferLength, std::string & result);
extern "C" ANSCV_API int	RunDetectorBinary(ANSCENTER::ANSFireDetector * *Handle, unsigned char* jpeg_bytes, unsigned int width, unsigned int height, std::string & result);
extern "C" ANSCV_API int	RunDetectorImagePath(ANSCENTER::ANSFireDetector * *Handle, const char* imageFilePath, std::string & result);

#endif