#include "ANSCV.h"
#include <MoveDetect.h>
#include <Windows.h>

namespace ANSCENTER {
	ANSFireDetector::ANSFireDetector() {
		/*
		NOTE:
			- Unknown what this HSV threshold is. 
		*/
		this->_hsvThreshold = 0.5;	

		/*
		NOTE:
			- Defining the colour ranges for fire and smoke.
		*/
		this->_fire_colour.push_back(std::pair(
			cv::Scalar(0, 0, 200), cv::Scalar(255, 255, 255)) // 0,0,179 -> 255, 255, 255
		);

		this->_smoke_colour.push_back(std::pair(
			cv::Scalar(0, 0, 51), cv::Scalar(180, 128, 204)) // 0,0,0 -> 255, 255, 127
		);
		
		/*
		NOTE:
			- PSNR value set to high for more subtle movement detection.
		*/
		this->_handler = MoveDetect::Handler();
		this->_handler.psnr_threshold = 45.0;
		this->_counter = 0;
	}

	ANSFireDetector::~ANSFireDetector() {
		/*
		NOTE:
			- (To-do): clean up.
		*/
		Destroy();
	}

	bool ANSFireDetector::Initialise(double HSVThreshold) {
		/*
		NOTE:
			- Custom threshold. This is basically a setter method. 
		*/
		bool success = true;
		_hsvThreshold = HSVThreshold;
		return success;
	}

	bool ANSFireDetector::Destroy() {
		bool success = true;
		/*
		NOTE:
			- (To-do): clean up.
		*/
		return success;
	}

	std::vector<Object> ANSFireDetector::Detect(cv::Mat& input) {
		std::vector<Object> objects;
		objects.clear();
		
		/*
		NOTE:
			- Enabling boxing and contouring. After `detect()` is called, it stores
			the output result (i.e., with boxings/contourings) into `output`. 
		*/
		this->_handler.contours_enabled = true;
		this->_handler.detect(input);
		this->_handler.output = input;
		
		/*
		NOTE:
			- Getting the bounding box for each contour.
		*/
		for (int i = 0; i < this->_handler.contours.size(); i++) {
			cv::Rect box = this->BoundingBoxFromContour(this->_handler.contours[i]);
			cv::Mat cropped = input(box);

			/*
			NOTE:
				- Initializing object to default values.
			*/
			Object object;
			this->InitializeObject(&object);
			object.box = box;

			/*
			NOTE:
				- Checking if the cropped frame has fire or smoke colour. We need
				to copy to `cropped` again because the methods used below modify
				the frames.
			*/

			bool fcol_detected = this->MajorityColourInFrame(cropped, this->_fire_colour, 0.15);
			bool scol_detected = this->MajorityColourInFrame(cropped, this->_smoke_colour);

			/*
			NOTE:
				- If fire colour detected or smoke colour detected. Might need to
				look into smoke colour method of detection as it empirically creates
				a lot of false positives.
			*/
			if (fcol_detected || scol_detected) {
				Object object;
				this->InitializeObject(&object);

				if (fcol_detected) {
					object.classId = FIRE;
					object.className = "Fire";
				}else if(scol_detected) {
					object.classId = SMOKE;
					object.className = "Smoke";
				}

				object.box = box;
				object.counter = this->_counter;

				/*
				NOTE:
					- Anything less than THRESHOLD % of the frame's total area gets ignored and
					discarded. This is to avoid too many unwanted ROIs.
				*/
				const float SMALL_ROI_FILTER_THRESHOLD = 0.02;
				if (float((float) (object.box.width * object.box.height) / 
					(float)(input.rows * input.cols)) < SMALL_ROI_FILTER_THRESHOLD) {

					continue;
				}
				objects.push_back(object);
				
			}
		}

		/*
		NOTE:
			- Drawing bounding boxes around the ROI in the output. This code here is
			only temporary as we only need the information in the JSON string. Delete
			this later in the final release.
		*/
		for (int i = 0; i < objects.size(); i++) {
			int OFFSET = 5;
			cv::Scalar box_colour;

			/*
			NOTE:
				- Setting the colour of the bounding box based on the type of detection.
			*/
			if (objects[i].classId == FIRE) {
				box_colour = cv::Scalar(255, 127, 127);
			} else if (objects[i].classId == SMOKE) {
				box_colour = cv::Scalar(255, 127, 255);
			}

			cv::rectangle(this->_handler.output, objects[i].box, box_colour, 1);

			/*
			NOTE:
				- Outputting text, with a custom shadow.
			*/
			cv::Point shadowpos(objects[i].box.x - 1, objects[i].box.y - OFFSET - 1);
			cv::putText(this->_handler.output, cv::format("%s",
				objects[i].className.c_str()), shadowpos, 0, 1,
				cv::Scalar(0, 0, 0), 0);

			cv::Point textpos(objects[i].box.x, objects[i].box.y - OFFSET);
			cv::putText(this->_handler.output, cv::format("%s",
				objects[i].className.c_str()), textpos, 0, 1,
				cv::Scalar(255, 255, 255), 0);
		}

		/*
		NOTE:
			- Increment the counter to keep track of which frame offset we are at.
			This is also temporary code. The counter should be deleted in product-
			ion.
		*/
		this->_counter++;

		return objects;
	}

	cv::Mat* ANSFireDetector::GetMovingContouredFrame() {
		/*
		NOTE:
			- The `MoveDetect::detect()` function stores the output here. Using a
			pointer to avoid copying the entire object.
		*/
		return &this->_handler.output;
	}

	std::string ANSFireDetector::ConvertResultsToString(std::vector<Object> *objects) {
		/*
		NOTE:
			- (To-do): use the JSON library to convert JSON object into a string.
		*/
		std::ostringstream intermediate;

		pt::ptree root;
		pt::ptree rnode;

		for (size_t i = 0; i < objects->size(); i++) {
			long int counter = (*objects)[i].counter;

			pt::ptree node;
			/*
			NOTE:
				- ID of `0` denotes none type. ID of `1` denotes a possibility of smoke
				and ID of `2` denotes possibility of fire.
			*/
			node.put<int>("id", (*objects)[i].classId);
			node.put<std::string>("class", (*objects)[i].className);

			/*
			NOTE:
				- Current ROI box and converting the coordinates to string.
			*/
			cv::Rect* box = &(*objects)[i].box;

			node.put<long int>("counter", counter);

			std::string top_left = std::to_string(box->x) + "," +
				std::to_string(box->y);
			node.put<std::string>("top-left", top_left);

			std::string bottom_right = std::to_string(box->x + box->width) + "," +
				std::to_string(box->y + box->height);
			node.put<std::string>("bottom-right", bottom_right);

			/*
			NOTE:
				- Region of interest.
			*/

			rnode.push_back(std::make_pair("ROI", node));
		}

		root.add_child("results", rnode);
		/*
		NOTE:
			- Align each node to a tree then translate that property to JSON file.
		*/
		pt::write_json(intermediate, root);
		std::string result = intermediate.str();
		return result;
	}

	bool ANSFireDetector::WriteResultsToFileCSV(std::string path, std::vector<Object>* objects,
		std::ios_base::openmode mode = (std::ios::out | std::ios::app)) {
		
		bool success = true;

		/*
		NOTE:
			- Default action. 
		*/
		std::ofstream file;

		file.open(path, mode);
		if (!file.is_open()) {
			std::cerr << "Unable to open the file\n";
			success = false;
			return success;
		}

		/*
		NOTE:
			Simply overwriting the file (non-default action) or appending to the
			end of the file (default action). The format is CSV.
		*/
		std::string output;

		/*
		NOTE:
			- Writing out the information corresponding to each column when the counter
			is initially incremented.
		*/
		if (!(this->_counter - 1)) {
			const std::string HEADER = "id,class,min_x,min_y,max_x,max_y,counter";
			file << HEADER + "\n";
		}
		
		/*
		NOTE:
			- Writing information of each object to the file. 
		*/
		for (Object object : *objects) {
			file	<< std::to_string(object.classId) 
					<< "," << object.className
					<< "," << std::to_string(object.box.x)
					<< "," << std::to_string(object.box.y)
					<< "," << std::to_string(object.box.x + object.box.width)
					<< "," << std::to_string(object.box.y + object.box.height)
					<< "," << std::to_string(this->_counter)
					<< "\n";
		}
		
		file.close();

		return success;
	}

	cv::Rect ANSFireDetector::BoundingBoxFromContour(std::vector<cv::Point> contour) {
		if (contour.size() <= 0) {
			cv::Rect empty;
			return empty;
		}
		if (cv::contourArea(contour) < 1000) {
			cv::Rect empty;
			return empty;
		}
		/*
		NOTE:
			- Finding exremas.
		*/
		cv::Point minp(contour[0].x, contour[0].y),
			maxp(contour[0].x, contour[0].y);

		for (unsigned int i = 0; i < contour.size(); i++) {
			/*
			NOTE:
				- Updating local minima.
			*/
			if (contour[i].x < minp.x) {
				minp.x = contour[i].x;
			}
			if (contour[i].y < minp.y) {
				minp.y = contour[i].y;
			}

			/*
			NOTE:
				- Updating local maxima.
			*/
			if (contour[i].x > maxp.x) {
				maxp.x = contour[i].x;
			}
			if (contour[i].y > maxp.y) {
				maxp.y = contour[i].y;
			}
		}

		/*
		NOTE:
			- Creating a box and returning the box based on the extremas found.
		*/
		cv::Rect box(minp, maxp);

		return box;
	}

	bool ANSFireDetector::MajorityColourInFrame(cv::Mat frame, Range range, float area_threshold)
	{
		/*
		NOTE:
			- Try catch memory leak when all frames are read.
		*/

		cv::Mat img_hsv;
		try {
			/*
			NOTE:
				- Convert colour into gray colour.
			*/
			cv::cvtColor(frame, img_hsv, cv::COLOR_BGR2HSV);

			/*
			NOTE:
				- The first of the redefined `Range` pair is always the lower
				threshold and the second is the upper threshold.
			*/
			cv::inRange(img_hsv, range.first, range.second, frame);
		}
		catch (cv::Exception& e) {
			return false;
		}

		return (cv::countNonZero(frame) / (frame.rows * frame.cols)) > area_threshold;
	}

	bool ANSFireDetector::MajorityColourInFrame(cv::Mat frame, std::vector<Range> ranges, float area_threshold)
	{
		/*
		NOTE:
			- Try catch memory leak when all frames are read.
		*/
		unsigned int current_area = 0;

		cv::Mat img_hsv;
		try {
			/*
			NOTE:
				- Convert colour into gray colour.
			*/
			cv::cvtColor(frame, img_hsv, cv::COLOR_BGR2HSV);

			/*
			NOTE:
				- The first of the redefined `Range` pair is always the lower
				threshold and the second is the upper threshold.
			*/
			for (const Range &range : ranges) {
				cv::inRange(img_hsv, range.first, range.second, frame);
				current_area += cv::countNonZero(frame);
			}
		}
		catch (cv::Exception& e) {
			return false;
		}

		return (float)current_area / ((float)frame.rows * frame.cols) > area_threshold;
	}

	void ANSFireDetector::InitializeObject(Object* object) {
		/*
		NOTE:
			- Initializing based on header specifications and comments.
		*/
		object->classId = DetectionType::NONE;
	}
}