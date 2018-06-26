#include <cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "caliberate.h"	
#include <iostream>
#include <ctime>				
#include <cmath>


using namespace std;
using namespace cv;
VideoCapture capture;		
Mat frameImg;				
Mat g_image;				

double dist(Point2i x1, Point2i x2)	
{	
	return sqrt( pow((x1.x-x2.x),2) + pow((x1.y-x2.y),2) ); 
}

int main() 
{
	string fileName = "traffic.avi";
	capture.open(fileName);		
	if( !capture.isOpened() )
	{	
		cerr<<"video opening error\n"; waitKey(0); system("pause");  
	}

	Mat frameImg_origSize;							
	namedWindow( "out"	  , CV_WINDOW_AUTOSIZE);	
	namedWindow( "trackbar", CV_WINDOW_AUTOSIZE);	
	resizeWindow( "trackbar", 300, 600);			
	
	capture>>frameImg_origSize;
	if( frameImg_origSize.empty() ) { cout<<"something wrong"; }
	
	resize(frameImg_origSize, frameImg, SMALL_SIZE, 0, 0, CV_INTER_AREA);	

	g_image = Mat(SMALL_SIZE, CV_8UC1);	g_image.setTo(0);	
	Mat roadImage = Mat(SMALL_SIZE, CV_8UC3);	
	roadImage = findRoadImage();
	
	calibPolygon();	

	Mat binImage = Mat(SMALL_SIZE,CV_8UC1);	
	Mat finalImage = Mat(SMALL_SIZE, CV_8UC3);	
	time_t T = time(0);	
	float fps = 0, lastCount = 0;	

	int thresh_r = 43, thresh_g = 43, thresh_b = 49;						
	createTrackbar( "Red Threshold", "trackbar", &thresh_r, 255, 0 );		
	createTrackbar( "Green Threshold", "trackbar", &thresh_g, 255, 0 );		
	createTrackbar( "Blue Threshold", "trackbar", &thresh_b, 255, 0 );		

	int dilate1=1, erode1=2, dilate2=5;	
	Mat imgA = Mat(SMALL_SIZE, CV_8SC3);	
	int win_size = 20;	
	int corner_count = MAX_CORNERS;	
	vector<Point2i> cornersA, cornersB;

	frameImg.copyTo(imgA);

	int arrowGap = 5;	
	
	createTrackbar("dilate 1","trackbar", &dilate1, 15, 0);	
	createTrackbar("erode 1","trackbar", &erode1, 15, 0);	
	createTrackbar("dilate 2","trackbar", &dilate2, 15, 0);	
	
	
	Mat dilate1_element = getStructuringElement(MORPH_ELLIPSE , Size(2 * dilate1 + 1, 2 * dilate1 + 1), Point(-1,-1) );
	Mat erode1_element = getStructuringElement(MORPH_ELLIPSE , Size(2 * erode1 + 1, 2 * erode1 + 1), Point(-1,-1) );
	Mat dilate2_element = getStructuringElement(MORPH_ELLIPSE , Size(2 * dilate2 + 1, 2 * dilate2 + 1), Point(-1,-1) );
	
	vector< Vec4i > hierarchy;
	vector< vector<Point> > contours;
	vector< uchar > vstatus; 
	vector< float >verror;

	while(true)
	{
		++fps;

		capture>>frameImg_origSize; 
		if( frameImg_origSize.empty() ) break; 

		resize(frameImg_origSize, frameImg, frameImg.size()); 
		imshow("Original Video", frameImg);
		
		for( int i=0; i<HEIGHT_SMALL; ++i) 
		{
			for(int j=0; j<WIDTH_SMALL; ++j)
			{
				if(	abs(roadImage.at<Vec3b>(i,j)[0]-frameImg.at<Vec3b>(i,j)[0])<thresh_r &&
					abs(roadImage.at<Vec3b>(i,j)[1]-frameImg.at<Vec3b>(i,j)[1])<thresh_g &&
					abs(roadImage.at<Vec3b>(i,j)[2]-frameImg.at<Vec3b>(i,j)[2])<thresh_b ) 
				{	binImage.at<uchar>(i,j) = 0;
					
				}	
				else
				{	binImage.at<uchar>(i,j) = 255;
					
				}	
		    }
		}
		
		frameImg.copyTo(finalImage);
		
		bitwise_and(binImage, polygonImg, binImage, noArray());	

		imshow("Binary Image", binImage);	
		
		dilate(binImage, binImage, dilate1_element);
		erode(binImage, binImage, erode1_element);
		dilate(binImage, binImage, dilate2_element);
		imshow("noise removed", binImage);	

		binImage.copyTo(g_image);
		
		findContours( g_image, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		

		double  percentArea = 0; 
		double contoursArea = 0;

		cout<<"("<<contours.size()<<") ";	

		for(unsigned int idx=0; idx<contours.size(); idx++)
		{
			if( !contours.at(idx).empty() )	
			{
				contoursArea += contourArea( contours.at(idx) );
			}
			Scalar color( rand()&255, rand()&255, rand()&255 );
			drawContours(finalImage, contours, idx, color);
		}

		contours.clear();
		hierarchy.clear();

		percentArea = contoursArea/polyArea;
		cout<<(int)(percentArea*100)<<"% ";

		int xCorners = 0; 
		for(int i=0; i<HEIGHT_SMALL; i+=arrowGap) 
		{
			for(int j=0; j<WIDTH_SMALL; j+=arrowGap)
			{
				if( xCorners >= MAX_CORNERS-1 ) break; 
				if( binImage.at<uchar>(i,j) == 255 )	
				{
					cornersA.push_back(Point2i(i,j));
					++xCorners;
				}
			}
		}
		cornersB.reserve(xCorners);		
		
		calcOpticalFlowPyrLK(imgA,frameImg,cornersA,cornersB,vstatus, verror, Size( win_size,win_size ),5, cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, .3 ),0); 
		double avgDist = 0; 
		for( int i=0; i<xCorners; i++ ) 
		{
			avgDist += dist(cornersA[i], cornersB[i]); 
			line( finalImage, Point(cornersA[i].x,cornersA[i].y), Point(cornersB[i].x,cornersB[i].y) , CV_RGB(0,0,255),1 , CV_AA); 
		}
		avgDist /= xCorners;
		cout<<setprecision(2)<<avgDist;
		frameImg.copyTo(imgA);

		cornersA.clear();
		cornersB.clear();
		vstatus.clear();
		verror.clear();

		line(finalImage, pts[0], pts[1], CV_RGB(0,255,0),1,CV_AA); 
		line(finalImage, pts[1], pts[2], CV_RGB(0,255,0),1,CV_AA);
		line(finalImage, pts[2], pts[3], CV_RGB(0,255,0),1,CV_AA);
		line(finalImage, pts[3], pts[0], CV_RGB(0,255,0),1,CV_AA);
		imshow( "Final Output", finalImage); 
		waitKey(33);
		if(time(0) >= T+1)
		{
			cout<<" ["<<fps<<"]";
			fps = 0;
			T = time(0);
		}
		cout<<endl;
	}
	return 0;
}
