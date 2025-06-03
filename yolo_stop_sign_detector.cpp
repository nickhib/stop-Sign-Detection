#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
/*
citation  https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49
i also used ispiration from sam own yolo example
*/
constexpr float CONFIDENCE_THRESHOLD = .85;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;
using namespace cv;
using namespace std;
// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors)/sizeof(colors[0]);

int main()
{
	
	int stop_sign_id = -1;
	VideoCapture vcap;
	VideoCapture cap("./2IMG_shortend.mp4");
	if (!cap.isOpened()) 
	{
        	std::cout << "Error" << std::endl;
        	return -1;
    	}
	int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	double fps = cap.get(cv::CAP_PROP_FPS);
	cv::VideoWriter writer("result.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'),
                           fps, cv::Size(frame_w, frame_h), true);
	if (!writer.isOpened()) {
        	std::cerr << "Error opening video writer" << std::endl;
        	return -1;
	}
	std::vector<std::string> class_names;
	{
		std::ifstream class_file("coco.names");
		if (!class_file)
		{
			std::cerr << "failed to open coco.names\n";
			return 0;
		}
		std::string line;
		int stop_sign_num = 0;
		while (std::getline(class_file, line)){
			class_names.push_back(line);
			if( line == "stop sign")
			{
				stop_sign_id = stop_sign_num;
				cout << stop_sign_id << endl;
			}
			stop_sign_num++;
		}
	}

	auto net = cv::dnn::readNetFromDarknet("yolov3.cfg", "yolov3.weights");
	if (cuda::getCudaEnabledDeviceCount() > 0) {
		cout << "cuda" << endl;
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	else
	{
    		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
   		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}

	auto output_names = net.getUnconnectedOutLayersNames();

	cv::Mat frame, blob;
	std::vector<cv::Mat> detections;
	int64  work_begin;
	int64 work_end;
	double work_fps;

	while(1)
	{

		work_begin =getTickCount();


		if(!cap.read(frame)) {
			std::cout << "No frame" << std::endl;
			break;
			cv::waitKey();
		}
        	cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);

        	net.setInput(blob);
        	net.forward(detections, output_names);
        	std::vector<int> indices;
        	std::vector<cv::Rect> boxes;
        	std::vector<float> scores;
        	for (int k = 0 ; k < detections.size();k++)
        	{
            		const auto num_boxes = detections[k].rows;

            		for (int i = 0; i < num_boxes; i++)
            		{
                		auto centerx = detections[k].at<float>(i, 0) * frame.cols;
                		auto centery = detections[k].at<float>(i, 1) * frame.rows;
                		auto width = detections[k].at<float>(i, 2) * frame.cols;
                		auto height = detections[k].at<float>(i, 3) * frame.rows;
                		cv::Rect rect(centerx - width/2, centery - height/2, width, height);

                    		auto confidence = *detections[k].ptr<float>(i, 5 + stop_sign_id);
                    		if (confidence >= CONFIDENCE_THRESHOLD)
                    		{
                        		boxes.push_back(rect);
                        		scores.push_back(confidence);
                    		}
            		}
        	}
		cv::dnn::NMSBoxes(boxes, scores, 0.0, NMS_THRESHOLD, indices);
        
            	for (size_t i = 0; i < indices.size(); ++i)
            	{
                	const auto color = colors[stop_sign_id % NUM_COLORS];

                	auto idx = indices[i];
                	const auto& rect = boxes[idx];
                	cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                	std::ostringstream label_ss;
                	label_ss << class_names[stop_sign_id] << ": " << std::fixed << std::setprecision(2) << scores[idx];
                	auto label = label_ss.str();
                
                	int baseline;
                	auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                	cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                	cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
            	}
    
		std::ostringstream stats_ss;
        	auto stats = stats_ss.str();
            
        	int baseline;
        	auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        	cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
        	cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));
		putText(frame, "FPS (total): " + to_string(work_fps), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
		work_end  =(getTickCount() - work_begin);
		work_fps = (getTickFrequency()/ work_end);
		writer.write(frame);
        	cv::namedWindow("output");
        	cv::imshow("output", frame);
        	cv::waitKey(5);
    }


    return 0;
}
