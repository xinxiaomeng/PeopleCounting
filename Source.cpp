#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cv.h>
#include "opencv/cxcore.hpp"
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <sstream>

using namespace std;
using namespace cv;

static const bool IS_DEBUG = false;				//設定是否顯示Debug訊息
static const int DIFFERENT_IMAGE_THRESHOLD = 5;	//前景的Threshold
static const int BINARIZED_THRESHOLD = 20;		//二直化的Threshold

VideoCapture capture;
Size frameSize;

int frameNum;
int frameIndex;

//Kalman filter
const int stateNum=4;
const int measureNum=2;

Mat	backgroundImage;
Mat iMatMax;
Mat iMatMin;

void Initial()
{
	capture = VideoCapture("FileName.mp4");

	frameSize.height = (int) capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	frameSize.width = (int) capture.get(CV_CAP_PROP_FRAME_WIDTH);

	frameNum = (int) capture.get(CV_CAP_PROP_FRAME_COUNT);
	frameIndex = 0;
}

void InitialImageMaxMin(const Mat* grayFrame)
{
	grayFrame->copyTo(iMatMax);
	grayFrame->copyTo(iMatMin);
}

void UpdateImageMaxMin(Mat* grayFrame)
{
	int index = 0;

	for(int i=0;i<grayFrame->rows;i++)
	{
        for(int j=0;j<grayFrame->cols;j++)
		{
			index = grayFrame->channels()*grayFrame->cols*i + j;

			if(grayFrame->data[index] > iMatMax.data[index])
				iMatMax.data[index] = grayFrame->data[index];

			if(grayFrame->data[index] < iMatMin.data[index])
				iMatMin.data[index] = grayFrame->data[index];
        }
    }
}

void UpdateBackgroundImage()
{
	uchar diffImage;
	uchar meanImage;

	uchar newBackgroundImage;
	uchar okdBackgroundImage;

	int index = 0;

	for(int i=0;i<backgroundImage.rows;i++)
	{
		for(int j=0;j<backgroundImage.cols;j++)
		{
			index = backgroundImage.channels()*backgroundImage.cols*i + j;

			diffImage = iMatMax.data[index] - iMatMin.data[index];

			if(diffImage < DIFFERENT_IMAGE_THRESHOLD)
			{
				meanImage = (uchar)((iMatMax.data[index] + iMatMin.data[index]) / 2);

				okdBackgroundImage = backgroundImage.data[index];

				newBackgroundImage = okdBackgroundImage *0.3+ meanImage *0.7;

				backgroundImage.data[index] = newBackgroundImage;

			}
        }
    }
}

void GetBackgroundMask(Mat* backgroundMask,  Mat* grayFrame)
{
	uchar diffMax = 0;
	uchar diffMin = 255;

	uchar tempImageValue;

	int index = 0;

	for(int i=0;i<grayFrame->rows;i++)
	{
        for(int j=0;j<grayFrame->cols;j++)
		{
			index = grayFrame->channels()*grayFrame->cols*i + j;

            tempImageValue = grayFrame->data[index] - backgroundImage.data[index];

			if(tempImageValue > diffMax)
			{
				diffMax = tempImageValue;
			}

			if(tempImageValue <  diffMin)
			{
				diffMin = tempImageValue;
			}
        }
    }

	for(int i=0;i<grayFrame->rows;i++)
	{
        for(int j=0;j<grayFrame->cols;j++)
		{
			index = grayFrame->channels()*grayFrame->cols*i + j;

            tempImageValue = abs(grayFrame->data[index] - backgroundImage.data[index]);

			tempImageValue = 255 * ((tempImageValue - diffMin)/(double)(diffMax - diffMin));

			backgroundMask->data[index] = tempImageValue;
        }
    }


	int erosion_size = 1;
	Mat element = getStructuringElement(MORPH_ELLIPSE,
                  Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                  Point(erosion_size, erosion_size) );

	erode(*backgroundMask, *backgroundMask, element, Point(-1,-1), 1, 0);

	erosion_size = 1;
	element = getStructuringElement(MORPH_ELLIPSE,
              Size(2 * erosion_size + 1, 2 * erosion_size + 1),
              Point(erosion_size, erosion_size) );

	dilate(*backgroundMask, *backgroundMask, element, Point(-1,-1), 3, 0);


	threshold(*backgroundMask, *backgroundMask, BINARIZED_THRESHOLD, 255, CV_THRESH_BINARY);


}

void SobelXY(Mat* sobel, Mat* grayFrame)
{
	Mat blur;
	Mat grad;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	GaussianBlur(*grayFrame, blur, Size(3,3), 0, 0, BORDER_DEFAULT );

	Sobel(blur, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	Sobel(blur, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, *sobel );

}

class Detected_point
{
	public:
		Detected_point(Point pvalue, int cvalue[][5])
		{
			for(int i=0;i<5;i++)
				for(int j=0;j<5;j++)
					color[i][j]=cvalue[i][j];
			p.x=pvalue.x;
			p.y=pvalue.y;

			Point p;

			life=0;
			detected_life=0;
			count=0;
		}
		Detected_point() {}
		~Detected_point () {}
		int color[5][5];
		int life, detected_life;
		Point p;
		bool count;
};

int template_size;

vector<Detected_point> newPoint;

void HeadDetection(Mat thisForceEdgeImage, Mat thisFrame)
{
	Mat templateImage[5];
	templateImage[0] = imread("head_2.jpg", 0);
	templateImage[1] = imread("head_5.jpg", 0);
	templateImage[2] = imread("head_6.jpg", 0);
	templateImage[3] = imread("head_7.jpg", 0);
	templateImage[4] = imread("head_8.jpg", 0);



	//Code
	template_size = templateImage[0].cols;
	//End

	//樣板大小 200*200
	for(int k=0;k<5;k++)
	{
		bool stop=0;

		Mat res(frameSize.width - templateImage[k].cols + 1, frameSize.height - templateImage[k].rows + 1,CV_32FC1);

		matchTemplate(thisForceEdgeImage, templateImage[k], res, CV_TM_CCORR_NORMED);

		waitKey(5);
		threshold(res,res, 0.5, 255,0);

		while(true)
		{
			double minval,maxval;
			double threshold = 0.5;
			Point minLoc, maxLoc;

			minMaxLoc(res, &minval, &maxval, &minLoc, &maxLoc, Mat());

			if(maxval >= threshold)
			{
				//Code
				int color[5][5];
				for(int i = 0; i < 5; i++)
				{
					for(int j = 0; j < 5; j++)
					{
						int x = maxLoc.x + i*templateImage[k].cols / 5;
						int y = maxLoc.y + j*templateImage[k].rows / 5;
						color[i][j]=thisForceEdgeImage.at<uchar>(y, x);
					}
				}

				newPoint.push_back(Detected_point(Point(maxLoc.x + templateImage[k].cols, maxLoc.y + templateImage[k].rows), color));

				stop = 1;
				//End

				floodFill(res,maxLoc,Scalar(0),0,Scalar(.1),Scalar(1.));
			}
			else
				break;
		}
		if(stop)
			break;
	}
}

bool valid_zone(Point data, Point newp)
{
	return (newp.x <= data.x+template_size/2 && newp.x >= data.x-template_size/2)
		&& (newp.y <= data.y+template_size && newp.y >= data.y-template_size);
}

int min_Value(int p1[][5], int p2[][5])
{
	int sum=0;
	for(int i=0;i<5;i++)
		for(int j=0;j<5;j++)
			sum+=pow(p1[i][j]-p2[i][j], 2.0);

	return pow(sum, 0.5);
}


int main()
{
	Initial();

	Mat thisFrame;
	Mat thisGrayFrame;
	Mat thisBackgroundMask;
	Mat thisSobel;
	Mat thisForceEdgeImage;

	/* Code */
	vector<Detected_point> data;

	int up=0;
	int down=0;

	while(frameIndex < frameNum)
	{
		capture >> thisFrame;

		frameIndex = (int) capture.get(CV_CAP_PROP_POS_FRAMES);

		cout << frameIndex << "/" << frameNum << endl;

		cvtColor(thisFrame, thisGrayFrame, CV_BGR2GRAY);

//		imshow("Video", thisGrayFrame);

		if(frameIndex == 1)
		{
			thisGrayFrame.copyTo(backgroundImage);
			thisGrayFrame.copyTo(thisBackgroundMask);
		}

		if(frameIndex % 10 ==1)
		{
			InitialImageMaxMin(&thisGrayFrame);
		}
		else if(frameIndex % 10 == 0)
		{
			UpdateBackgroundImage();
			if(IS_DEBUG) imshow("Background", backgroundImage);
		}
		else
		{
			UpdateImageMaxMin(&thisGrayFrame);
		}

		GetBackgroundMask(&thisBackgroundMask, &thisGrayFrame);
//		if(IS_DEBUG) imshow("Background Mask", thisBackgroundMask);

		SobelXY(&thisSobel, &thisGrayFrame);
//		if(IS_DEBUG) imshow("Soble", thisSobel);

		bitwise_and(thisSobel, thisBackgroundMask, thisForceEdgeImage);

		HeadDetection(thisForceEdgeImage, thisFrame);

		/* Code Start */

		//追蹤
		for(int i=0;i<data.size();i++)
		{
			int min = 2147483647;

			for(int j = 0; j < newPoint.size(); j++)
				if(valid_zone(data[i].p, newPoint[j].p))
					if(min_Value(data[i].color, newPoint[j].color)<min)
						min = min_Value(data[i].color, newPoint[j].color);

			for(int j = 0; j < newPoint.size(); j++)
			{
				if(min_Value(data[i].color, newPoint[j].color) == min && min < 2147483647)
				{
					if(!data[i].count)
					{
						//up
						if(data[i].p.y > 360 && newPoint[j].p.y < 360 && data[i].life > 0)
						{
							data[i].count = 1;
							up++;
						}
						//down
						else if(data[i].p.y < 360 && newPoint[j].p.y > 360 && data[i].life > 0)
						{
							data[i].count = 1;
							down++;
						}
					}

					//更新座標
					data[i].p.x = newPoint[j].p.x;
					data[i].p.y = newPoint[j].p.y;

					//偵測次數
					data[i].detected_life++;

					break;
				}
			}
		}

		//資料清除週期
		for(int i=0;i<data.size();i++)
			data[i].life++;

		//清除過期資料與誤判
		while(1)
		{
			int size = 0;
			for(int i = 0; i < data.size(); i++)
			{
				if(data[i].life>25 || data[i].life-data[i].detected_life>4)
					data.erase(data.begin()+i);
				else size++;
			}
			if(data.size() == size)
				break;
		}

		//更新 新偵測到的點
		for(int i = 0; i < newPoint.size(); i++)
		{
			bool add = 1;
			for(int j = 0; j < data.size(); j++)
			{
				if(newPoint[i].p == data[j].p)
				{
					add = 0;
					break;
				}
			}
			if(add)
				data.push_back(Detected_point(newPoint[i].p, newPoint[i].color));
		}

		//清除重疊誤判
		for(int i = 0;i < data.size(); i++)
		{
			while(1)
			{
				int size = 0;
				for(int j= i + 1; j < data.size(); j++)
				{
					if(abs(data[i].p.x-data[j].p.x) < template_size / 2
						|| abs(data[i].p.y-data[j].p.y) < template_size / 2)
					{
						if(data[i].life > data[j].life)
							data.erase(data.begin() + j);
						else
							data.erase(data.begin() + i);
					}
					else size++;
				}
				if(size == data.size() - i - 1)
					break;
			}
		}

		//標出頭部
		for(int i = 0;i < data.size(); i++)
		{
			rectangle(thisFrame, Point(data[i].p.x - template_size, data[i].p.y - template_size),
					  Point(data[i].p.x + template_size / 8, data[i].p.y + template_size / 8),
					  CV_RGB(255,255,255), 2);

			rectangle(thisForceEdgeImage, Point(data[i].p.x - template_size, data[i].p.y - template_size),
					  Point(data[i].p.x + template_size / 8, data[i].p.y + template_size / 8),
					  CV_RGB(255,255,255), 2);
		}

		//清除newPoint
		while(newPoint.size() > 0)
			newPoint.erase(newPoint.begin());

		//顏色
		CvScalar yellow = CV_RGB(255, 255, 0);
		CvScalar red = CV_RGB(255, 0, 0);

		//int轉string
		string s1, s2;
		stringstream ss1(s1);
		ss1 << up;
		s1 = ss1.str();

		stringstream ss2(s2);
		ss2 << down;
		s2 = ss2.str();

		CvFont font = cvFont(1, 1); //Scale, Thickness

		//印出人數
		/*
		putText(thisFrame, "Up : ", Point(1000, 100), CV_FONT_HERSHEY_COMPLEX, 1, red);
		putText(thisFrame, s1, Point(1200, 100), CV_FONT_HERSHEY_COMPLEX, 1, red);
		putText(thisFrame, "Down : ", Point(1000, 150), CV_FONT_HERSHEY_COMPLEX, 1, red);
		putText(thisFrame, s2, Point(1200, 150), CV_FONT_HERSHEY_COMPLEX, 1, red);

		putText(thisForceEdgeImage, "Up : ", Point(1000, 100), CV_FONT_HERSHEY_COMPLEX, 1, red);
		putText(thisForceEdgeImage, s1, Point(1200, 100), CV_FONT_HERSHEY_COMPLEX, 1, red);
		putText(thisForceEdgeImage, "Down : ", Point(1000, 150), CV_FONT_HERSHEY_COMPLEX, 1, red);
		putText(thisForceEdgeImage, s2, Point(1200, 150), CV_FONT_HERSHEY_COMPLEX, 1, red);
		*/

		imshow("Video", thisFrame);
		imshow("Video2", thisForceEdgeImage);

		/* Code End */

		//save Image
		if(frameIndex == 28)
		{
			imwrite("ForceEdgeImage1.jpg", thisForceEdgeImage);
			imwrite("thisFrame1.jpg", thisFrame);
		}

		if(frameIndex == 35)
		{
			imwrite("ForceEdgeImage2.jpg", thisForceEdgeImage);
			imwrite("thisFrame2.jpg", thisFrame);
		}

        if(waitKey(33) == 27) break;
	}
}