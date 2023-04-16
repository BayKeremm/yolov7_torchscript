#include <torch/script.h> // One-stop header.

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include<string>
#include <iostream>
#include <memory>
static std::vector<std::string> class_names{"blue","large orange", "small orange", "other cone", "yellow"};
void show_image(cv::Mat& img, std::string title)
{
    cv::imshow(title + " type:", img);
    cv::waitKey(0);
}
void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 2);
	//Get the label for the class name and its confidence
	std::string label = cv::format("%.2f", conf);
	label = class_names[classid] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = cv::max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 1);
}

auto ToTensor(cv::Mat img, bool show_output = false, bool unsqueeze=false, int unsqueeze_dim = 0)
{
    std::cout << "image shape: " << img.size() << std::endl;
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kFloat32);

    return tensor_image;
}

auto transpose(at::Tensor tensor, c10::IntArrayRef dims = { 0, 3, 1, 2 })
{
    //std::cout << "############### transpose ############" << std::endl;
    //std::cout << "shape before : " << tensor.sizes() << std::endl;
    tensor = tensor.permute(dims);
    //std::cout << "shape after : " << tensor.sizes() << std::endl;
    //std::cout << "######################################" << std::endl;
    return tensor;
}
auto ToInput(at::Tensor tensor_image,const torch::Device& device)
{
    // Create a vector of inputs.
    tensor_image = tensor_image.to(device);
    return std::vector<torch::jit::IValue>{tensor_image};
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
    
    // check if CUDA is available
    if(!torch::hasCUDA()){
        std::cerr << "CUDA is not available\n";
        return 1;
    }

    torch::Device device(torch::kCUDA, 0);


    torch::jit::script::Module yolo;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        yolo = torch::jit::load(argv[1]);
        yolo.to(device);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "model ok\n";

    std::string image_filepath = "../all.jpeg";
    cv::Mat image = cv::imread(image_filepath);
    cv::resize(image,image,cv::Size(640,640));
    cv::cvtColor( image , image , cv::COLOR_RGB2BGR );//opencv uses bgr
    cv::Mat newImage;
    image.convertTo(newImage, CV_32FC3, 1.0 / 255.0);

    auto tensor = ToTensor(newImage);
    tensor = transpose(tensor, { (2),(0),(1) });
    //add batch dim (an inplace operation just like in pytorch)
    tensor.unsqueeze_(0);
    //std::cout << "shape after tensor conversion : " << tensor.sizes() << std::endl;

    std::vector<torch::jit::IValue> input_to_net = ToInput(tensor, device);
    
    // forward the image
    auto output = yolo.forward(input_to_net).toTuple();
    //std::cout << output->elements()[0].toTensor().sizes() << "\n\n";
    at::Tensor preds = output->elements()[0].toTensor();

    preds = preds.to(torch::kCPU);

    std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
    float ratioh = (float)image.rows / 640, ratiow = (float)image.cols / 640; 
    const int proposals = 25200; // default for 640x640 images, I think it is related to the amount of anchors

    for(int i = 1; i <= proposals; i++){
        at::Tensor pred = preds.slice(1,i-1,i);
        float box_score = pred.index({0,0,4}).item<float>();
        if(box_score > 0.8){// larger than confidence threshold
            at::Tensor scores = pred.slice(2,5,10);
            float max = torch::max(scores).item<float>();
            at::Tensor max_index = (scores == max).nonzero();
            max *= box_score;
               
            if(max > 0.8) { // conf threshold
                //std::cout << scores << " is the scores\n";
                const int class_id = max_index.index({0,2}).item<int>();
                //std::cout << class_id << "is the max index\n";
                float cx = pred.index({0,0,0}).item<float>() * ratiow;
                float cy = pred.index({0,0,1}).item<float>() * ratioh;
                float w = pred.index({0,0,2}).item<float>() * ratiow;
                float h = pred.index({0,0,3}).item<float>() * ratioh;

                int left = int(cx - 0.5*w);
                int top = int(cy - 0.5*h);
                cv::Rect b = cv::Rect(left, top, (int)(w), (int)(h));
                confidences.push_back((float)max);
                boxes.push_back(b);
                classIds.push_back(class_id);
            } 
        }
    }
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, 0.8, 0.6, indices);
    for (size_t i = 0; i < indices.size(); ++i)
	{
		//count++;
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		drawPred(confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, image, classIds[idx]);
	}
    cv::cvtColor( image , image , cv::COLOR_BGR2RGB );//opencv uses bgr
    show_image(image,"maybe");
}
    


