#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
using namespace std;
bool detectred(cv::Mat &img, double &centerX, double &centerY, bool hasPrev,
               double prevX, double prevY) { //图像甄别标记函数
  bool found = false;
  cv::Mat hsv;
  cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
  /*std::cout << "宽=" << img.cols << " 高=" << img.rows
            << " 通道=" << img.channels() << std::endl
            <<"总像素数="<<img.total()<<"\n";*/
  cv::Mat dst, clean, redlow, redhigh;
  vector<vector<cv::Point>> contours;
  cv::inRange(hsv, cv::Scalar(0, 80, 80), cv::Scalar(10, 255, 255),
              redlow); //检测范围
  cv::inRange(hsv, cv::Scalar(170, 80, 80), cv::Scalar(179, 255, 255),
              redhigh);   //检测范围
  dst = redhigh | redlow; //两张和到一起
  /*cv::Vec3b p1=dst.at<cv::Vec3b> (240,100);
  cv::Vec3b p2=dst.at<cv::Vec3b> (240,460);
  cv::Vec3b p3=dst.at<cv::Vec3b> (100,100);*/
  cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)); //侵蚀核
  cv::morphologyEx(dst, clean, cv::MORPH_OPEN, kernel);             //侵蚀
  cv::findContours(clean, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE); //找图形
  double area;
  cv::Rect rect;
  double bestArea = 0;
  double bestDistance2 = 1e18;
  for (size_t i = 0; i < contours.size(); i++) {
    area = cv::contourArea(contours[i]);
    if (area < 500) {
      continue; //图形太小了
    }
    found = true;
    rect = cv::boundingRect(contours[i]);
    double candidateX = rect.x + rect.width / 2.0;
    double candidateY = rect.y + rect.height / 2.0;
    double diffX = candidateX - prevX;
    double diffY = candidateY - prevY;
    double distance2 = diffX * diffX + diffY * diffY;
    if ((!hasPrev && area > bestArea) ||
        (hasPrev && distance2 < bestDistance2)) {
      bestArea = area; //当前面积比最佳面积大的话，把当前的面积作为最佳面积标准
      bestDistance2=distance2;//切换最佳距离
      centerX = candidateX;
      centerY = candidateY;
    } //找中心
    // cout << area << ' ' << rect << " " << centerX << " " << centerY << "\n";
    cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2); //边框
    cv::putText(img, "RED", cv::Point(rect.x, rect.y - 20),
                cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
  }
  /*cout<<(int)p1[0]<<" "<<(int)p1[1]<<" "<<(int)p1[2]<<"\n";
  cout<<(int)p2[0]<<" "<<(int)p2[1]<<" "<<(int)p2[2]<<"\n";
  cout<<(int)p3[0]<<" "<<(int)p3[1]<<" "<<(int)p3[2]<<"\n";
  uchar p1=clean.at<uchar> (434,254);
  uchar p2=clean.at<uchar> (64,124);
  uchar p3=clean.at<uchar> (403,563);
  out<<(int)p1<<"\n";
  cout<<(int)p2<<"\n";
  cout<<(int)p3<<"\n";
  cv::imshow("111", img);
  cv::waitKey(0);*/
  return found;
}
int main() {
  cv::VideoCapture cap("data/test8_area_swap.avi");
  if (!cap.isOpened()) {
    cout << "视频打开失败"
         << "\n";
    return -1;
  }
  cv::Mat frame;
  long long sum = 0;
  cv::VideoWriter writer("data/result_area_swap_01.avi",
                         cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
                         cv::Size(640, 480));
  if (!writer.isOpened()) {
    cout << "视频写入程序失败";
    return -1;
  }
  double prevX = 0, prevY = 0;
  bool hasPrev = false;
  double lastX = 0, lastY = 0;
  bool hasPrediction = false;
  while (cap.read(frame)) { //读一帧
    cv::Mat img = frame;
    if (img.empty()) { // frame读不到
      cout << "读图失败，检查路径" << std::endl;
      return -1;
    }
    double centerX = 0, centerY = 0;
    bool found = detectred(frame, centerX, centerY, hasPrev, prevX,
                           prevY); //检测目标，取当前帧中的图形中心
    if (found) {
      if (hasPrediction) { //上一帧有预测
        cout << "误差=" << centerX - lastX << " " << centerY - lastY << "\n";
      }              //本帧中心与上一帧预测中心的误差
      if (hasPrev) { //有上一帧，然后预测
        double dx = centerX - prevX; // x方向的位移
        double dy = centerY - prevY; // Y方向的位移
        double predictedx = centerX + dx;
        double predictedy = centerY + dy; //预测下一帧的
        cv::circle(frame, cv::Point(cvRound(predictedx), cvRound(predictedy)),
                   5, cv::Scalar(0, 255, 255), -1);
        lastX = predictedx;
        lastY = predictedy; //给下一轮来计算误差
        hasPrediction = true;
        cout << prevX << " " << prevY << " " << centerX << " " << centerY << " "
             << dx << " " << dy << " " << predictedx << " " << predictedy
             << "\n";
      }
      prevX = centerX, prevY = centerY; //存这一帧的中心，用于下次的位移计算
      hasPrev = true;
    } else {
      hasPrev = false;
      hasPrediction = false;
      cout << "目标丢失"
           << "\n";
    }
    writer.write(frame);
    sum++;
    /*bool saved =
        cv::imwrite("data/result_video_frame_01.png", img);
    if (saved) {
      cout << "saved success" << "\n";
    } else {
      cout << "saved fail" << "\n";
      ;
    }*/
  }
  writer.release();
  cout << "处理帧数=" << sum << "\n";
  return 0;
}
