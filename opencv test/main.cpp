#include <vector>
#include <string>
#include "cv.h"
#include "cvaux.h"
#include "highgui.h"

static CvMemStorage* storage = 0;
//static CvHaarClassifierCascade* cascade = 0;
static CvScalar colors[] = {
    {{0,0,255}},{{0,128,255}},{{0,255,255}},{{0,255,0}},
    {{255,128,0}},{{255,255,0}},{{255,0,0}},{{255,0,255}}
};
void detect_and_draw( IplImage* image );

const char* cascade_name =
"/Users/kouruiri/Desktop/cascadefina.xml";

void detect_and_draw(IplImage* img,const CvHaarClassifierCascade* cascade )
{
    storage = cvCreateMemStorage(0);
	cvClearMemStorage( storage );
    
    
    //Image Preparation
    //
    double scale=0.5;
    CvSize dst_cvsize;
    IplImage* gray1 = (IplImage*) img;
    dst_cvsize.width = img->width * scale;
    dst_cvsize.height = img->height * scale;
    IplImage* small_img= cvCreateImage( dst_cvsize, img->depth,img->nChannels);
    
    
    IplImage* gray;
    gray = cvCreateImage(cvGetSize(small_img), IPL_DEPTH_8U, 1);
    //IplImage* small_img=cvCreateImage(cvSize(img->width/scale,img->height/scale),8,1);
    
    cvResize(img, small_img);
    cvCvtColor(small_img,gray, CV_BGR2GRAY);
    cvEqualizeHist(gray,gray);
    cvClearMemStorage(storage);
    
    double t = (double)cvGetTickCount();
    CvSeq* objects = cvHaarDetectObjects(gray,
                                         (CvHaarClassifierCascade*)cascade,
                                         storage,
                                         1.1,
                                         2,
                                         CV_HAAR_DO_CANNY_PRUNING/*CV_HAAR_DO_CANNY_PRUNING*/,
                                         cvSize(30,30));
    
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %gms\n   %d objects\n", t/((double)cvGetTickFrequency()*1000.) ,objects->total);
    
    //Loop through found objects and draw boxes around them
    for(int i=0;i<objects->total;++i)
    {
        CvRect* r=(CvRect*)cvGetSeqElem(objects,i);
        cvRectangle(gray1, cvPoint(r->x/scale,r->y/scale), cvPoint((r->x+r->width)/scale,(r->y+r->height)/scale), colors[i%8]);
        
        CvFont font;
        cvInitFont(&font,CV_FONT_HERSHEY_PLAIN, 1.0, 1.0, 0,1,CV_AA);
    }
    
    cvShowImage( "result", gray1 );
    cvReleaseImage(&gray);
//    cvReleaseImage(&gray1);
    cvReleaseImage(&small_img);
    cvReleaseMemStorage(&storage);
    }

int main( int argc, char** argv )
{
    CvHaarClassifierCascade* faceCascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0 );
	if( !faceCascade ) {
		printf("ERROR in recognizeFromCam(): Could not load Haar cascade Face detection classifier.\n");
		exit(1);
	}
    
    IplImage* pFrame = NULL;
    CvCapture* pCapture = cvCreateCameraCapture(0);
    
    cvNamedWindow("video", 1);
    
    while(1)
    {
        pFrame=cvQueryFrame( pCapture );
        if(!pFrame)break;
        detect_and_draw(pFrame,faceCascade);
        //cvShowImage("video",pFrame);
        char c=cvWaitKey(33);
        if(c==27)break;
    }
    cvReleaseCapture(&pCapture);
    cvDestroyWindow("video");
}
