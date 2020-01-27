#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::cuda;
using namespace std;

const string TEST_IMAGE = "../assets/Im3.png";
const int GRADIENT_THRESH[2] = {20,100};
const int L_CHANNEL_THRESH[2] = {130,255};
const int B_CHANNEL_THRESH[2] = {170,1210};


int main( int argc, char** argv )
{
	Mat src = imread("TEST_IMAGE",IMREAD_COLOR);
	
	GpuMat gpu;
	gpu.upload(src);
	
	if (gpu.type==CV_8UC1) 
		cout << gpu.type << endl;
	else
		cv::cuda::cvtColor(gpu,gpu,COLOR_BGR2RGB);
	
	
	
	imshow("test", gpu);
	waitKey(0);
	
    return 0;
}
