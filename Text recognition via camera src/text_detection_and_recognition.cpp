//  This example can not only detect text through the camera, but also perform real-time text recognition. Finally, display the results in the bounding box.
//  Text detection uses EAST and text recognition uses VGG Net+CTC.
//  
//  You can follow this guide to train by yourself using the MNIST dataset.
//  https://github.com/intel/caffe/blob/a3d5b022fe026e9092fc7abc7654b1162ab9940d/examples/mnist/readme.md
//
//  You can also download already trained model directly.
//  EAST in https://github.com/argman/EAST
//  VGG Net in  


#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::dnn;


const char* keys =
    "{ help  h     | | Print help message. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ modelDet    | /home/moo/Desktop/opencv_project/model/frozen_east_text_detection.pb | Path to a binary .pb file contains the Text detection model.}"
    "{ modelRec    | /home/moo/Desktop/ocr/clovaai/new_model/VGG_CTC.onnx | Path to a .onnx file contains the Text recognition model.}"
    "{ widthDet    | 320 | Preprocess input image by resizing to a specific width. It should be multiple by 32. }"
    "{ heightDet   | 320 | Preprocess input image by resizing to a specific height. It should be multiple by 32. }"
    "{ widthRec    | 100 | Preprocess input image by resizing to a specific width. }"
    "{ heightRec   | 32 | Preprocess input image by resizing to a specific height. }"
    "{ thr         | 0.5 | Confidence threshold. }"
    "{ nms         | 0.4 | Non-maximum suppression threshold. }"
    ;

void decodeDet(const Mat& scores, const Mat& geometry, float scoreThresh,
            std::vector<RotatedRect>& detections, std::vector<float>& confidences);

const std::string vocabulary = "0123456789abcdefghijklmnopqrstuvwxyz";

std::string decodeRec(Mat prediction);

int main(int argc, char** argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of "
                  "EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)");
    // if (argc == 1 || parser.has("help"))
    // {
    //     parser.printMessage();
    //     return 0;
    // }

    float confThreshold = parser.get<float>("thr");
    float nmsThreshold = parser.get<float>("nms");
    int inpWidthDet = parser.get<int>("widthDet");
    int inpHeightDet = parser.get<int>("heightDet");
    std::string modelDet = parser.get<String>("modelDet");
    
    int inpWidthRec = parser.get<int>("widthRec");
    int inpHeightRec = parser.get<int>("heightRec");
    std::string modelRec = parser.get<String>("modelRec");


    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    CV_Assert(!modelDet.empty());

    // Load text dectction network.
    Net netDet = readNet(modelDet);

    CV_Assert(!modelRec.empty());

    // Load text recognition network.
    Net netRec = readNetFromONNX(modelRec);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(0);

    static const std::string kWinName = "EAST+VGG Text detection and recognition.";
    namedWindow(kWinName, WINDOW_NORMAL);
    std::vector<Mat> outs;
    std::vector<String> outNames(2);
    outNames[0] = "feature_fusion/Conv_7/Sigmoid";
    outNames[1] = "feature_fusion/concat_3";

    Mat frame, blob;
    int frameHeight, frameWidth;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }

        blobFromImage(frame, blob, 1.0, Size(inpHeightDet, inpHeightDet), Scalar(123.68, 116.78, 103.94), true, false);
        netDet.setInput(blob);
        netDet.forward(outs, outNames);

        Mat scores = outs[0];
        Mat geometry = outs[1];

        //Detection Decode predicted bounding boxes.
        std::vector<RotatedRect> boxes;
        std::vector<float> confidences;
        decodeDet(scores, geometry, confThreshold, boxes, confidences);

        // Apply non-maximum suppression procedure.
        std::vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        // Render detections.
        Point2f ratio((float)frame.cols / inpWidthDet, (float)frame.rows / inpHeightDet);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            RotatedRect& box = boxes[indices[i]];
            
            Point2f vertices[4];
            box.points(vertices);
            
            Point2f dstVertices[] = {
                Point2f(0, inpHeightRec-1),
                Point2f(0,0),
                Point2f(inpWidthRec-1,0),
                Point2f(inpWidthRec-1, inpHeightRec-1),
            };

            for (int j = 0; j < 4; ++j)
            {
                vertices[j].x *= ratio.x;
                vertices[j].y *= ratio.y;
            }

            // Crop detection result 
            Mat transMatrix = getPerspectiveTransform(vertices, dstVertices);
            Mat warpMat, predRec;
            warpPerspective(frame, warpMat, transMatrix, Size(inpWidthRec, inpHeightRec));
            cvtColor(warpMat, warpMat, COLOR_BGR2GRAY);

            Mat blobImg = dnn::blobFromImage(warpMat,1.0,Size(100,32),Scalar(),true);

            netRec.setInput(blobImg);
            predRec = netRec.forward();
            std::string decodeSeq = decodeRec(predRec);
            std::cout<<decodeSeq<<std::endl;

            putText(frame, decodeSeq, vertices[0], 3, 1.0, Scalar(128, 128, 255), 2);
            
            for (int j = 0; j < 4; ++j)
                line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 1);
        }

        // Put efficiency information.
        std::vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = netDet.getPerfProfile(layersTimes) / freq;
        std::string label = format("Inference time: %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        imshow(kWinName, frame);
    }
    return 0;
}

void decodeDet(const Mat& scores, const Mat& geometry, float scoreThresh,
            std::vector<RotatedRect>& detections, std::vector<float>& confidences)
{
    detections.clear();
    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
    CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y)
    {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < scoreThresh)
                continue;

            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                           offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
            Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
            RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}

std::string decodeRec(Mat prediction)
{
    std::string decodeSeq = "";
    int maxLoc = 0;
    bool ctcFlag = true;
    for (int i = 0; i < prediction.size[0]; i++) {

        Mat predRow = prediction.row(i);
        cv::MatIterator_<float> maxVal = std::max_element(predRow.begin<float>(),predRow.end<float>());

        if ((*maxVal) > *predRow.begin<float>()){
            maxLoc = maxVal - predRow.begin<float>();
        }

        if (maxLoc > 0) {
            char currentChar = vocabulary[maxLoc - 1];
            if (currentChar != decodeSeq.back() || ctcFlag) {
                decodeSeq += currentChar;
                ctcFlag = false;
            }
        } else {
            ctcFlag = true;
        }
    }
    return decodeSeq;
}

