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

//柱形投影
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

//球面投影
Mat spherical(Mat src) {


	int width = src.cols, height = src.rows;
	Mat dst(height, width, CV_8UC3, Scalar::all(0));

	Mat map_x(height, width, CV_32FC1), map_y(height, width, CV_32FC1);

	double radius1 = height / 2;
	double radius2 = radius1*radius1;
	//球面坐标
	double x, y;
	//原坐标
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
			//移动坐标轴，使原点在图像中心
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
			//坐标转换
			yy = (height / 2 - y);
			xx = (x + width / 2);

			//这里在确定图像大小的情况下可以用查表法来完成，这样会大大的提高其效率
			map_x.at<float>(i, j) = xx;
			map_y.at<float>(i, j) = yy;
		}
	}

	remap(src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 0, 0));

	return dst;
}

//flag---投影类型
bool OpenCVWarped(int flag) {

	//输入图像
	vector<Mat> imgs;
	Mat img = imread("D:/imags/01.jpg");
	imgs.push_back(img);
	img = imread("D:/imags/02.jpg");
	imgs.push_back(img);

	//特征检测
	Ptr<FeaturesFinder> finder;
	//finder = new SurfFeaturesFinder();
	finder = new OrbFeaturesFinder();
	vector<ImageFeatures> features(2);
	(*finder)(imgs[0], features[0]);
	(*finder)(imgs[1], features[1]);

	//特征匹配
	vector<MatchesInfo> pairwise_matches;
	BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);
	matcher(features, pairwise_matches);

	//相机参数评估
	HomographyBasedEstimator estimator;
	vector<CameraParams> cameras;
	estimator(features, pairwise_matches, cameras);
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}

	//光束平差法，精确相机参数
	Ptr<detail::BundleAdjusterBase> adjuster;
	adjuster = new detail::BundleAdjusterReproj();
	adjuster->setConfThresh(1);
	(*adjuster)(features, pairwise_matches, cameras);

	vector<Mat> rmats;
	for (size_t i = 0; i < cameras.size(); ++i)
		rmats.push_back(cameras[i].R.clone());
	//波形校正
	waveCorrect(rmats, WAVE_CORRECT_HORIZ);
	for (size_t i = 0; i < cameras.size(); ++i)
		cameras[i].R = rmats[i];

	vector<Point> corners(2);    //表示映射变换后图像的左上角坐标
	vector<Mat> masks_warped(2);    //表示映射变换后的图像掩码
	vector<Mat> images_warped(2);    //表示映射变换后的图像
	vector<Size> sizes(2);    //表示映射变换后的图像尺寸
	vector<Mat> masks(2);    //表示源图的掩码

	for (int i = 0; i < 2; ++i)    //初始化源图的掩码
	{
		masks[i].create(imgs[i].size(), CV_8U);    //定义尺寸大小
		masks[i].setTo(Scalar::all(255));    //全部赋值为255，表示源图的所有区域都使用
	}

	//定义图像映射变换创造器
	Ptr<WarperCreator> warper_creator;
	switch (flag)
	{
	case 1:
		//平面投影
		warper_creator = new cv::PlaneWarper();
		break;
	case 2:
		//柱面投影
		warper_creator = new cv::CylindricalWarper();
		break;
	case 3:
		//球面投影
		warper_creator = new cv::SphericalWarper();
		break;
	case 4:
		//鱼眼投影
		warper_creator = new cv::FisheyeWarper();
		break;
	case 5:
		//立方体投影
		warper_creator = new cv::StereographicWarper();
		break;
	default:
		cout << "选择类型错误！" << endl;
		return false;
		break;
	}

	//定义图像映射变换器，设置映射的尺度为相机的焦距，所有相机的焦距都相同
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(cameras[0].focal));
	for (int i = 0; i < 2; ++i)
	{
		Mat_<float> K;
		//转换相机内参数的数据类型
		cameras[i].K().convertTo(K, CV_32F);
		//对当前图像镜像投影变换，得到变换后的图像以及该图像的左上角坐标
		corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		//得到尺寸
		sizes[i] = images_warped[i].size();
		//得到变换后的图像掩码
		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}
	//通过掩码，只得到映射变换后的图像
	for (int k = 0; k < 2; k++)
	{
		for (int i = 0; i < sizes[k].height; i++)
		{
			for (int j = 0; j < sizes[k].width; j++)
			{
				if (masks_warped[k].at<uchar>(i, j) == 0)    //掩码
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


void DealWithImgData(uchar *srcdata, uchar *drcdata, int width, int height)//参数一为原图像的数据区首指针，参数二为贴合后图像的数据区首指针，参数三为图像的宽，参数四为图像的高
{

	int l_width = width * 4;//计算位图的实际宽度并确保它为4byte的倍数 
	double radius1 = height / 2;//贴合球面半径
	double radius2 = radius1*radius1;//半径的平方
	double x1, y1;//目标在球正视图中的坐标位置（球面坐标）
	double x, y;//目标在球正视图中对应原图的坐标位置（原坐标）
	double middle2 = 2 * radius1 / 3.1416;//计算过程式子
	double matan;//目标与圆心连线与x轴的夹角
	int pixel_point;//遍历图像指针
	int pixel_point_row;//遍历图像行首指针
	double oa;//点对应弧长度

	//双线性插值算法相关变量
	int i_original_img_hnum, i_original_img_wnum;//目标点坐标
	double distance_to_a_y, distance_to_a_x;//在原图像中与a点的水平距离  
	int original_point_a, original_point_b, original_point_c, original_point_d;

	for (int hnum = 0; hnum < height; hnum++)
	{
		pixel_point_row = hnum*l_width;
		for (int wnum = 0; wnum < width; wnum++)
		{
			if ((hnum - height / 2)*(hnum - height / 2) + (wnum - width / 2)*(wnum - width / 2) < radius2)//在球体视场内才处理
			{
				pixel_point = pixel_point_row + wnum * 3;//数组位置偏移量，对应于图像的各像素点RGB的起点
														 /***********球面贴合***********/
				x1 = wnum - width / 2;
				y1 = height / 2 - hnum;

				if (x1 != 0)
				{
					oa = middle2*asin(sqrt(y1*y1 + x1*x1) / radius1);//这里在确定图像大小的情况下可以用查表法来完成，这样会大大的提高其效率
					matan = atan2(y1, x1);
					x = cos(matan)*oa;
					y = sin(matan)*oa;
				}
				else
				{
					y = asin(y1 / radius1)*middle2;
					x = 0;
				}
				/***********球面贴合***********/

				/***********双线性插值算法***********/
				i_original_img_hnum = (height / 2 - y);
				i_original_img_wnum = (x + width / 2);
				distance_to_a_y = (height / 2 - y) - i_original_img_hnum;
				distance_to_a_x = (x + width / 2) - i_original_img_wnum;//在原图像中与a点的垂直距离  

				original_point_a = i_original_img_hnum*l_width + i_original_img_wnum * 3;//数组位置偏移量，对应于图像的各像素点RGB的起点,相当于点A    
				original_point_b = original_point_a + 3;//数组位置偏移量，对应于图像的各像素点RGB的起点,相当于点B  
				original_point_c = original_point_a + l_width;//数组位置偏移量，对应于图像的各像素点RGB的起点,相当于点C   
				original_point_d = original_point_c + 3;//数组位置偏移量，对应于图像的各像素点RGB的起点,相当于点D  

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
				/***********双线性插值算法***********/
			}
		}
	}
}


//双线性插值法缩放算法
void BGRBilinearScale(const Mat src, Mat& dst) {

	double dstH = dst.rows;  //目标图片高度
	double dstW = dst.cols;  //目标图片宽度
	double srcW = src.cols;  //原始图片宽度，如果用int可能会导致(srcH - 1)/(dstH - 1)恒为零
	double srcH = src.rows;  //原始图片高度
	double xm = 0;      //映射的x
	double ym = 0;      //映射的y
	int xi = 0;         //映射x整数部分
	int yi = 0;         //映射y整数部分
	int xl = 0;         //xi + 1
	int yl = 0;         //yi + 1
	double xs = 0;
	double ys = 0;

	/* 为目标图片每个像素点赋值 */
	for (int i = 0; i < dstH; i++) {
		for (int j = 0; j < dstW; j++) {
			//求出目标图像(i,j)点到原图像中的映射坐标(mapx,mapy)
			xm = (srcH - 1) / (dstH - 1) * i;
			ym = (srcW - 1) / (dstW - 1) * j;
			/* 取映射到原图的xm的整数部分 */
			xi = (int)xm;
			yi = (int)ym;
			/* 取偏移量 */
			xs = xm - xi;
			ys = ym - yi;

			xl = xi + 1;
			yl = yi + 1;
			//边缘点
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

	//opencv扭曲方法
	//OpenCVWarped();
	Mat dst = spherical(img_1);
	imwrite("res.jpg", dst);
	system("res.jpg");

	return 0;
}