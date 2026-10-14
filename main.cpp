#include <cmath>
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
  cv::Mat dst, clean, redlow, redhigh;
  cv::Mat hsv;
  int kernelWidth = 9, kernelHeight = 9; //侵蚀核大小
  cv::Size kernelSize(kernelWidth, kernelHeight);
  cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
  double area = 0;
  cv::Rect rect, trueRect;
  double bestArea = 0;
  double bestDistance2 = 1e18;//
  double miniarea = 500;                         //最小侦测面积
  int maxDetectWidth = 80, maxDetectHeight = 80; //最大预测距离的区域
  double maxDetectDistance = maxDetectHeight * maxDetectWidth; //最大预测距离
  cv::Scalar hsv1Lower(170, 60, 50), hsv1Upper(179, 255, 255),
      hsv2Lower(0, 60, 50),
      hsv2Upper(10, 255, 255); //双颜色防止红色逆时针找不到
  /*std::cout << "宽=" << img.cols << " 高=" << img.rows
            << " 通道=" << img.channels() << std::endl
            <<"总像素数="<<img.total()<<"\n";*/
  vector<vector<cv::Point>> contours;
  cv::inRange(hsv, hsv1Lower, hsv1Upper,
              redlow); //检测颜色范围
  cv::inRange(hsv, hsv2Lower, hsv2Upper,
              redhigh); //检测颜色范围，后期可考虑用变量代替Scalar(已改)
  dst = redhigh | redlow; //两张和到一起
  /*cv::Vec3b p1=dst.at<cv::Vec3b> (240,100);
  cv::Vec3b p2=dst.at<cv::Vec3b> (240,460);
  cv::Vec3b p3=dst.at<cv::Vec3b> (100,100);*/
  cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, kernelSize); //侵蚀核
  cv::morphologyEx(dst, clean, cv::MORPH_OPEN, kernel);         //侵蚀
  cv::findContours(clean, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE); //找图形
  for (size_t i = 0; i < contours.size(); i++) {
    area = cv::contourArea(contours[i]);
    // cout<<"候选面积="<<area<<"位置"<<'\n';
    if (area < miniarea) {
      continue; //图形太小了
    }
    rect = cv::boundingRect(contours[i]);
    double candidateX = rect.x + rect.width / 2.0;
    double candidateY = rect.y + rect.height / 2.0;
    double diffX = candidateX - prevX;
    double diffY = candidateY - prevY;
    double distance2 = diffX * diffX + diffY * diffY;
    if ((!hasPrev && area > bestArea) ||
        (hasPrev && distance2 < bestDistance2 &&
         distance2 <=
             maxDetectDistance)) { //上一帧没有的话选面积大的，有的话确认与上一帧目标的距离，防止目标面积突变出现跳转，并且将误差设为80*80,过大的话消失后容易侦测到其他的，太小的话容易抓不到正确的
      bestArea =
          area; //当前面积比最佳面积大的话，把当前的面积作为最佳面积标准比
      bestDistance2 = distance2; //切换最佳距离
      centerX = candidateX;
      centerY = candidateY;
      trueRect = rect;
      found = true;
    } //找中心
    // cout << area << ' ' << rect << " " << centerX << " " << centerY << "\n";
  }
  if (found) {
    cv::rectangle(img, trueRect, cv::Scalar(0, 255, 0), 2); //边框
    cv::putText(img, "RED", cv::Point(trueRect.x, trueRect.y - 20),
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
  cv::VideoCapture cap("/home/snire/rob/data/realtest.mp4"); //视频源文件
  int fps = 30;                          //针对视频情况设置帧率
  int width = 720, height = 1280;        //设置分辨率
  double HoldSeconds = 0.5;              //目标丢失后尝试跟踪时间
  int maxLostFrames = HoldSeconds * fps; //尝试跟踪的帧数
  double prevX = 0, prevY = 0;
  bool hasPrev = false;
  double lostFrames = 0;
  double lastX = 0, lastY = 0;
  bool hasPrediction = false;
  double errorsum = 0, RightTimes = 0; //用于检测整体的预测情况如何
  double errormax = 0;                 //最大误差
  double errornow = 0;                 //当前帧误差
  double recoveryErrorsum = 0;
  int recoveryCount = 0;           //能计算找回误差的次数
  int totalRecoveryCount = 0;      //实际找回次数
  double lastDx = 0, lastDy = 0;   //上一帧预测值
  bool hasSpeed = false;           //速度是否有效
  double searchX = 0, searchY = 0; //搜索中心
  if (!cap.isOpened()) {
    cout << "视频打开失败"
         << "\n";
    return -1;
  }
  cv::Mat frame;
  long long sum = 0;
  cv::VideoWriter writer(
      "/home/snire/rob/data/result_real_test_01.avi", //保存目录
      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
      cv::Size(width, height));
  if (!writer.isOpened()) {
    cout << "视频写入程序失败";
    return -1;
  }
  while (cap.read(frame)) { //读一帧
    cv::Mat img = frame;
    if (img.empty()) { // frame读不到
      cout << "读图失败，检查路径" << std::endl;
      return -1;
    }
    double centerX = 0, centerY = 0; //中心点坐标
    searchX = prevX, searchY = prevY;
    if (hasPrev && lostFrames > 0 && hasSpeed) {
      searchX = prevX + lastDx * (lostFrames + 1);
      searchY = prevY + lastDy * (lostFrames + 1);
    }
    bool found = detectred(frame, centerX, centerY, hasPrev, searchX,
                           searchY); //检测目标，取当前帧中的图形中心
    if (found) {
      if (lostFrames != 0) {
        totalRecoveryCount++; //丢失后重新找到就算找回；
        if (hasSpeed) {
          double recoveryError = hypot(centerX - searchX, centerY - searchY);
          recoveryErrorsum += recoveryError;
          recoveryCount++;
          cout << "找回误差=" << recoveryError << "\n";
        }
        hasSpeed = false;
      }
      if (hasPrediction) { //上一帧有预测
        cout << "误差=" << centerX - lastX << " " << centerY - lastY << "\n";
        RightTimes++;                                       //预测次数++
        errornow = hypot(centerX - lastX, centerY - lastY); //当前误差距离
        errorsum += errornow;                               //误差距离汇总
        if (errornow > errormax) {
          errormax = errornow; //判断最大误差值
        }
      } //本帧中心与上一帧预测中心的误差
      if (hasPrev && lostFrames == 0) { //有上一帧，未丢失，则if内进行预测
        double dx = centerX - prevX; // x方向的位移
        double dy = centerY - prevY; // Y方向的位移
        double predictedx = centerX + dx;
        double predictedy = centerY + dy; //预测下一帧的中心位置
        cv::circle(frame, cv::Point(cvRound(predictedx), cvRound(predictedy)),
                   5, cv::Scalar(0, 255, 255), -1); //在预测的下一帧中心加圆点
        lastX = predictedx;
        lastY = predictedy; //给下一轮来计算误差
        lastDx = dx;
        lastDy = dy;          //存入帧位移，用于下一帧预测
        hasSpeed = true;      //速度有效
        hasPrediction = true; //有预测
        cout << prevX << " " << prevY << " " << centerX << " " << centerY << " "
             << dx << " " << dy << " " << predictedx << " " << predictedy
             << "\n";
      }
      prevX = centerX, prevY = centerY; //存这一帧的中心，用于下次的位移计算
      lostFrames = 0;                   //丢失帧数归零
      hasPrev = true;
    } else if (hasPrev) {
      lostFrames++;

      if (lostFrames > maxLostFrames) {
        hasPrev = false;
        lostFrames = 0;
        hasSpeed = false;
        cout << "目标确认丢失\n";
      } else {
        if (hasSpeed) { //丢失后预测中心点
          double predictedX = prevX + lastDx * lostFrames;
          double predictedY = prevY + lastDy * lostFrames;
          cv::circle(frame, cv::Point(cvRound(predictedX), cvRound(predictedY)),
                     5, cv::Scalar(0, 255, 255), -1);
        }
        cout << "目标暂时丢失\n";
      }

      hasPrediction = false;
    }

    writer.write(frame);
    sum++;
    /*bool saved =
        cv::imwrite("/home/snire/rob/data/result_video_frame_01.png", img);
    if (saved) {
      cout << "saved success" << "\n";
    } else {
      cout << "saved fail" << "\n";
      ;
    }*/
  }
  writer.release();
  cout << "处理帧数=" << sum << "\n";
  if (RightTimes > 0) {
    cout << "预测次数=" << RightTimes << "\n"
         << "平均预测误差=" << errorsum / RightTimes << "\n"
         << "最大误差=" << errormax << "\n";
  }
  cout << "实际找回次数=" << totalRecoveryCount << "\n";
  if (recoveryCount > 0) {
    cout << "找回样本数=" << recoveryCount << "\n"
         << "平均找回误差=" << recoveryErrorsum / recoveryCount << "\n";
  } else {
    cout << "找回样本数=" << recoveryCount << "\n";
  }
  return 0;
}