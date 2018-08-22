#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include <fstream> 
#include <string>
#include <iomanip> 
using namespace cv;
using namespace std;
using namespace detail;
#define  PI 3.14159

bool OpenCVWarped(int flag = 3);

//����ͶӰ
Mat cylinder(Mat& src) {

	Mat img_result = src.clone();
	for (int i = 0; i < img_result.rows; i++)
	{
		for (int j = 0; j < img_result.cols; j++)
		{
			img_result.at<Vec3b>(i, j) = 0;
		}
	}
	int W = src.cols;
	int H = src.rows;
	float r = W / (2 * tan(PI / 6));
	float k = 0;
	float fx = 0;
	float fy = 0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			k = sqrt((float)(r*r + (W / 2 - j)*(W / 2 - j)));
			fx = r*sin(PI / 6) + r*sin(atan((j - W / 2) / r));
			fy = H / 2 + r*(i - H / 2) / k;
			int ix = (int)fx;
			int iy = (int)fy;
			if (ix < W&&ix >= 0 && iy < H&&iy >= 0)
			{
				img_result.at<Vec3b>(iy, ix) = src.at<Vec3b>(i, j);
			}

		}
	}

	return img_result;
}

//����ͶӰ
Mat spherical(Mat src) {


	int width = src.cols, height = src.rows;
	Mat dst(height, width, CV_8UC3, Scalar::all(0));

	Mat map_x(height, width, CV_32FC1), map_y(height, width, CV_32FC1);

	double radius1 = height / 2;
	double radius2 = radius1*radius1;
	//��������
	double x, y;
	//ԭ����
	int x1, y1;
	float xx, yy;
	double middle = 2 * radius1 / PI;
	double matan;
	double oa;
	for (int i = 0; i < height; i++)
	{
		uchar* row_src = src.ptr(i);
		uchar* row_dst = dst.ptr(i);
		for (int j = 0; j < width; j++)
		{
			//�ƶ������ᣬʹԭ����ͼ������
			x1 = j - width / 2;
			y1 = height / 2 - i;

			if (x1 != 0)
			{
				oa = middle*asin(sqrt(y1*y1 + x1*x1) / radius1);
				matan = atan2(y1, x1);
				x = cos(matan)*oa;
				y = sin(matan)*oa;
			}
			else
			{
				y = asin(y1 / radius1)*middle;
				x = 0;
			}
			//����ת��
			yy = (height / 2 - y);
			xx = (x + width / 2);

			//������ȷ��ͼ���С������¿����ò������ɣ���������������Ч��
			map_x.at<float>(i, j) = xx;
			map_y.at<float>(i, j) = yy;
		}
	}

	remap(src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 0, 0));

	return dst;
}

//flag---ͶӰ����
bool OpenCVWarped(int flag) {

	//����ͼ��
	vector<Mat> imgs;
	Mat img = imread("D:/imags/01.jpg");
	imgs.push_back(img);
	img = imread("D:/imags/02.jpg");
	imgs.push_back(img);

	//�������
	Ptr<FeaturesFinder> finder;
	//finder = new SurfFeaturesFinder();
	finder = new OrbFeaturesFinder();
	vector<ImageFeatures> features(2);
	(*finder)(imgs[0], features[0]);
	(*finder)(imgs[1], features[1]);

	//����ƥ��
	vector<MatchesInfo> pairwise_matches;
	BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);
	matcher(features, pairwise_matches);

	//�����������
	HomographyBasedEstimator estimator;
	vector<CameraParams> cameras;
	estimator(features, pairwise_matches, cameras);
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}

	//����ƽ�����ȷ�������
	Ptr<detail::BundleAdjusterBase> adjuster;
	adjuster = new detail::BundleAdjusterReproj();
	adjuster->setConfThresh(1);
	(*adjuster)(features, pairwise_matches, cameras);

	vector<Mat> rmats;
	for (size_t i = 0; i < cameras.size(); ++i)
		rmats.push_back(cameras[i].R.clone());
	//����У��
	waveCorrect(rmats, WAVE_CORRECT_HORIZ);
	for (size_t i = 0; i < cameras.size(); ++i)
		cameras[i].R = rmats[i];

	vector<Point> corners(2);    //��ʾӳ��任��ͼ������Ͻ�����
	vector<Mat> masks_warped(2);    //��ʾӳ��任���ͼ������
	vector<Mat> images_warped(2);    //��ʾӳ��任���ͼ��
	vector<Size> sizes(2);    //��ʾӳ��任���ͼ��ߴ�
	vector<Mat> masks(2);    //��ʾԴͼ������

	for (int i = 0; i < 2; ++i)    //��ʼ��Դͼ������
	{
		masks[i].create(imgs[i].size(), CV_8U);    //����ߴ��С
		masks[i].setTo(Scalar::all(255));    //ȫ����ֵΪ255����ʾԴͼ����������ʹ��
	}

	//����ͼ��ӳ��任������
	Ptr<WarperCreator> warper_creator;
	switch (flag)
	{
	case 1:
		//ƽ��ͶӰ
		warper_creator = new cv::PlaneWarper();
		break;
	case 2:
		//����ͶӰ
		warper_creator = new cv::CylindricalWarper();
		break;
	case 3:
		//����ͶӰ
		warper_creator = new cv::SphericalWarper();
		break;
	case 4:
		//����ͶӰ
		warper_creator = new cv::FisheyeWarper();
		break;
	case 5:
		//������ͶӰ
		warper_creator = new cv::StereographicWarper();
		break;
	default:
		cout << "ѡ�����ʹ���" << endl;
		return false;
		break;
	}

	//����ͼ��ӳ��任��������ӳ��ĳ߶�Ϊ����Ľ��࣬��������Ľ��඼��ͬ
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(cameras[0].focal));
	for (int i = 0; i < 2; ++i)
	{
		Mat_<float> K;
		//ת������ڲ�������������
		cameras[i].K().convertTo(K, CV_32F);
		//�Ե�ǰͼ����ͶӰ�任���õ��任���ͼ���Լ���ͼ������Ͻ�����
		corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		//�õ��ߴ�
		sizes[i] = images_warped[i].size();
		//�õ��任���ͼ������
		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}
	//ͨ�����룬ֻ�õ�ӳ��任���ͼ��
	for (int k = 0; k < 2; k++)
	{
		for (int i = 0; i < sizes[k].height; i++)
		{
			for (int j = 0; j < sizes[k].width; j++)
			{
				if (masks_warped[k].at<uchar>(i, j) == 0)    //����
				{
					images_warped[k].at<Vec3b>(i, j)[0] = 0;
					images_warped[k].at<Vec3b>(i, j)[1] = 0;
					images_warped[k].at<Vec3b>(i, j)[2] = 0;
				}
			}
		}
	}

	imwrite("warp1.jpg", images_warped[0]);
	imwrite("warp2.jpg", images_warped[1]);

	system("warp1.jpg");
	system("warp2.jpg");
	return true;

}


void DealWithImgData(uchar *srcdata, uchar *drcdata, int width, int height)//����һΪԭͼ�����������ָ�룬������Ϊ���Ϻ�ͼ�����������ָ�룬������Ϊͼ��Ŀ�������Ϊͼ��ĸ�
{

	int l_width = width * 4;//����λͼ��ʵ�ʿ�Ȳ�ȷ����Ϊ4byte�ı��� 
	double radius1 = height / 2;//��������뾶
	double radius2 = radius1*radius1;//�뾶��ƽ��
	double x1, y1;//Ŀ����������ͼ�е�����λ�ã��������꣩
	double x, y;//Ŀ����������ͼ�ж�Ӧԭͼ������λ�ã�ԭ���꣩
	double middle2 = 2 * radius1 / 3.1416;//�������ʽ��
	double matan;//Ŀ����Բ��������x��ļн�
	int pixel_point;//����ͼ��ָ��
	int pixel_point_row;//����ͼ������ָ��
	double oa;//���Ӧ������

	//˫���Բ�ֵ�㷨��ر���
	int i_original_img_hnum, i_original_img_wnum;//Ŀ�������
	double distance_to_a_y, distance_to_a_x;//��ԭͼ������a���ˮƽ����  
	int original_point_a, original_point_b, original_point_c, original_point_d;

	for (int hnum = 0; hnum < height; hnum++)
	{
		pixel_point_row = hnum*l_width;
		for (int wnum = 0; wnum < width; wnum++)
		{
			if ((hnum - height / 2)*(hnum - height / 2) + (wnum - width / 2)*(wnum - width / 2) < radius2)//�������ӳ��ڲŴ���
			{
				pixel_point = pixel_point_row + wnum * 3;//����λ��ƫ��������Ӧ��ͼ��ĸ����ص�RGB�����
														 /***********��������***********/
				x1 = wnum - width / 2;
				y1 = height / 2 - hnum;

				if (x1 != 0)
				{
					oa = middle2*asin(sqrt(y1*y1 + x1*x1) / radius1);//������ȷ��ͼ���С������¿����ò������ɣ���������������Ч��
					matan = atan2(y1, x1);
					x = cos(matan)*oa;
					y = sin(matan)*oa;
				}
				else
				{
					y = asin(y1 / radius1)*middle2;
					x = 0;
				}
				/***********��������***********/

				/***********˫���Բ�ֵ�㷨***********/
				i_original_img_hnum = (height / 2 - y);
				i_original_img_wnum = (x + width / 2);
				distance_to_a_y = (height / 2 - y) - i_original_img_hnum;
				distance_to_a_x = (x + width / 2) - i_original_img_wnum;//��ԭͼ������a��Ĵ�ֱ����  

				original_point_a = i_original_img_hnum*l_width + i_original_img_wnum * 3;//����λ��ƫ��������Ӧ��ͼ��ĸ����ص�RGB�����,�൱�ڵ�A    
				original_point_b = original_point_a + 3;//����λ��ƫ��������Ӧ��ͼ��ĸ����ص�RGB�����,�൱�ڵ�B  
				original_point_c = original_point_a + l_width;//����λ��ƫ��������Ӧ��ͼ��ĸ����ص�RGB�����,�൱�ڵ�C   
				original_point_d = original_point_c + 3;//����λ��ƫ��������Ӧ��ͼ��ĸ����ص�RGB�����,�൱�ڵ�D  

				if (hnum == height - 1)
				{
					original_point_c = original_point_a;
					original_point_d = original_point_b;
				}
				if (wnum == width - 1)
				{
					original_point_a = original_point_b;
					original_point_c = original_point_d;
				}

				drcdata[pixel_point + 0] =
					srcdata[original_point_a + 0] * (1 - distance_to_a_x)*(1 - distance_to_a_y) +
					srcdata[original_point_b + 0] * distance_to_a_x*(1 - distance_to_a_y) +
					srcdata[original_point_c + 0] * distance_to_a_y*(1 - distance_to_a_x) +
					srcdata[original_point_c + 0] * distance_to_a_y*distance_to_a_x;
				drcdata[pixel_point + 1] =
					srcdata[original_point_a + 1] * (1 - distance_to_a_x)*(1 - distance_to_a_y) +
					srcdata[original_point_b + 1] * distance_to_a_x*(1 - distance_to_a_y) +
					srcdata[original_point_c + 1] * distance_to_a_y*(1 - distance_to_a_x) +
					srcdata[original_point_c + 1] * distance_to_a_y*distance_to_a_x;
				drcdata[pixel_point + 2] =
					srcdata[original_point_a + 2] * (1 - distance_to_a_x)*(1 - distance_to_a_y) +
					srcdata[original_point_b + 2] * distance_to_a_x*(1 - distance_to_a_y) +
					srcdata[original_point_c + 2] * distance_to_a_y*(1 - distance_to_a_x) +
					srcdata[original_point_c + 2] * distance_to_a_y*distance_to_a_x;
				/***********˫���Բ�ֵ�㷨***********/
			}
		}
	}
}


//˫���Բ�ֵ�������㷨
void BGRBilinearScale(const Mat src, Mat& dst) {

	double dstH = dst.rows;  //Ŀ��ͼƬ�߶�
	double dstW = dst.cols;  //Ŀ��ͼƬ���
	double srcW = src.cols;  //ԭʼͼƬ��ȣ������int���ܻᵼ��(srcH - 1)/(dstH - 1)��Ϊ��
	double srcH = src.rows;  //ԭʼͼƬ�߶�
	double xm = 0;      //ӳ���x
	double ym = 0;      //ӳ���y
	int xi = 0;         //ӳ��x��������
	int yi = 0;         //ӳ��y��������
	int xl = 0;         //xi + 1
	int yl = 0;         //yi + 1
	double xs = 0;
	double ys = 0;

	/* ΪĿ��ͼƬÿ�����ص㸳ֵ */
	for (int i = 0; i < dstH; i++) {
		for (int j = 0; j < dstW; j++) {
			//���Ŀ��ͼ��(i,j)�㵽ԭͼ���е�ӳ������(mapx,mapy)
			xm = (srcH - 1) / (dstH - 1) * i;
			ym = (srcW - 1) / (dstW - 1) * j;
			/* ȡӳ�䵽ԭͼ��xm���������� */
			xi = (int)xm;
			yi = (int)ym;
			/* ȡƫ���� */
			xs = xm - xi;
			ys = ym - yi;

			xl = xi + 1;
			yl = yi + 1;
			//��Ե��
			if ((xi + 1) > (srcH - 1)) xl = xi - 1;
			if ((yi + 1) > (srcW - 1)) yl = yi - 1;

			//b
			dst.at<Vec3b>(i, j)[0] = (int)(src.at<Vec3b>(xi, yi)[0] * (1 - xs)*(1 - ys) +
				src.at<Vec3b>(xi, yl)[0] * (1 - xs)*ys +
				src.at<Vec3b>(xl, yi)[0] * xs*(1 - ys) +
				src.at<Vec3b>(xl, yl)[0] * xs*ys);
			//g
			dst.at<Vec3b>(i, j)[1] = (int)(src.at<Vec3b>(xi, yi)[1] * (1 - xs)*(1 - ys) +
				src.at<Vec3b>(xi, yl)[1] * (1 - xs)*ys +
				src.at<Vec3b>(xl, yi)[1] * xs*(1 - ys) +
				src.at<Vec3b>(xl, yl)[1] * xs*ys);
			//r
			dst.at<Vec3b>(i, j)[2] = (int)(src.at<Vec3b>(xi, yi)[2] * (1 - xs)*(1 - ys) +
				src.at<Vec3b>(xi, yl)[2] * (1 - xs)*ys +
				src.at<Vec3b>(xl, yi)[2] * xs*(1 - ys) +
				src.at<Vec3b>(xl, yl)[2] * xs*ys);

		}
	}
}


int main(int argc, char** argv)
{
	Mat img_1 = imread("D:/imags/horse.jpg");
	//Mat dst;
	
	// 	imwrite("res.jpg", cylinder(img_1));
	// 	system("res.jpg");

	//opencvŤ������
	//OpenCVWarped();
	Mat dst = spherical(img_1);
	imwrite("res.jpg", dst);
	system("res.jpg");

	return 0;
}