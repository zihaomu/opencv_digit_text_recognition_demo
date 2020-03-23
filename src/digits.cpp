#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> 


using namespace cv;
using namespace std;
using namespace cv::dnn;

/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
	Mat probMat = probBlob.reshape(1, 1);
	Point classNumber;
	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}
vector<Rect> ROIposition;
vector<Rect> ROI;

Mat gammaCorrection(const Mat &img, const double gamma_)
{
	CV_Assert(gamma_ >= 0);
	//! [changing-contrast-brightness-gamma-correction]
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);

	Mat res = img.clone();
	LUT(img, lookUpTable, res);
	//! [changing-contrast-brightness-gamma-correction]
	Mat img_gamma_corrected;
	hconcat(img, res, img_gamma_corrected);
	return img_gamma_corrected;
}


void predictor(dnn::Net net, Mat &roi, int &class_id, double &probability) {

	Mat pred;
	Mat inputBlob = dnn::blobFromImage(roi, 1, Size(28, 28), Scalar(), false); //Convert Mat to batch of images
	net.setInput(inputBlob, "data");//set the network input, "data" is the name of the input layer

	pred = net.forward("prob");//compute output, "prob" is the name of the output layer
	//cout << pred << endl; 
	getMaxClass(pred, &class_id, &probability);

}

int main()
{
	// load model 
	string modelTxt = "./lenet3.prototxt";
	string modelBin = "./lenet.caffemodel";

	dnn::Net net;
	try {
		net = dnn::readNetFromCaffe(modelTxt, modelBin);
	}
	catch (cv::Exception &ee) {
		cerr << "Exception: " << ee.what() << endl;
		if (net.empty()) {
			cout << "Can't load the network by using the flowing files:" << endl;
			cout << "modelTxt: " << modelTxt << endl;
			cout << "modelBin: " << modelBin << endl; exit(-1);
		}
	}

	namedWindow("result",WINDOW_AUTOSIZE);
	namedWindow("raw",WINDOW_AUTOSIZE);

	Mat labels, img_color, stats, centroids;
	Point positiosn;

	Rect g_rectangle;
	bool g_bDrawingBox = false;

	int class_id = 0;
	double probability = 0;

	VideoCapture cap(0);

	// set camera resolution
	cap.set(CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CAP_PROP_FRAME_HEIGHT, 480);

	Rect basicRact = Rect(0, 0, 640, 480);
	Mat raw_image;
	
	double fps = 0;
	char string_fps[10];

	if (cap.isOpened())
	{
		TickMeter cvtm;

		while (true)
		{
			string fps_string("FPS:");
			cvtm.reset();
			cvtm.start();
			cap >> raw_image;

			Mat image = raw_image.clone();

			// preprocessing 
			cvtColor(image, image, COLOR_BGR2GRAY);
			GaussianBlur(image, image, Size(3, 3),2, 2);
			adaptiveThreshold(image, image, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, 10);
			bitwise_not(image, image);
			
			// find connected component
			int nccomps = cv::connectedComponentsWithStats(image, labels, stats,centroids);

			for (int i = 1; i < nccomps; i++) {
				g_bDrawingBox = false;

				// extend the bounding box of connected component
				if (stats.at<int>(i - 1, CC_STAT_AREA) > 80 && stats.at<int>(i - 1, CC_STAT_AREA) < 3000) {
					g_bDrawingBox = true;
					int left = stats.at<int>(i - 1, CC_STAT_HEIGHT) / 4;
					g_rectangle = Rect(stats.at<int>(i - 1, CC_STAT_LEFT) - left, stats.at<int>(i - 1, CC_STAT_TOP)- left, stats.at<int>(i - 1, CC_STAT_WIDTH) +2* left, stats.at<int>(i - 1, CC_STAT_HEIGHT)+2*left);
					g_rectangle &= basicRact;
				}

				if (g_bDrawingBox) {
					Mat roi = image(g_rectangle);
					predictor(net,roi, class_id, probability);

					if (probability < 0.5)
						continue;
					cout << "probability : "<<probability << endl;
					
					rectangle(raw_image, g_rectangle, Scalar(128, 255, 128), 2);

					positiosn = Point(g_rectangle.br().x - 7, g_rectangle.br().y + 25);
					putText(raw_image, to_string(class_id), positiosn, 3, 1.0, Scalar(128, 128, 255), 2);

				}

			}
			
			cvtm.stop();
			fps = 1 / cvtm.getTimeSec();
			sprintf_s(string_fps, "%.2f", fps);
			fps_string += string_fps;
			putText(image, fps_string, Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(128, 255, 128));
	
			printf("time = %gms\n", cvtm.getTimeMilli());
			imshow("result", image);
			imshow("raw", raw_image);
			waitKey(30);

		}
	}
	return 0;
}
