#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "ELSED.h"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

inline void
drawSegments(cv::Mat img,
             upm::Segments segs,
             const cv::Scalar &color,
             int thickness = 1,
             int lineType = cv::LINE_AA,
             int shift = 0) {
  for (const upm::Segment &seg: segs)
    cv::line(img, cv::Point2f(seg[0], seg[1]), cv::Point2f(seg[2], seg[3]), color, thickness, lineType, shift);
}

int main(int argc, const char** argv) {
  std::cout << "******************************************************" << std::endl;
  std::cout << "******************* ELSED main demo ******************" << std::endl;
  std::cout << "******************************************************" << std::endl;

  // Using default parameters (long segments)
  cv::Mat img = cv::imread(argv[1]);
  if (img.empty()) {
    std::cerr << "Error reading input image" << std::endl;
    return -1;
  }

  cv::TickMeter tm;
  tm.start();
  upm::ELSED elsed;
  upm::Segments segs = elsed.detect(img);
  tm.stop();
  std::cout << "ELSED detected: " << segs.size() << " (large) segments : " << tm << std::endl;

  drawSegments(img, segs, CV_RGB(0, 255, 0), 2);
  cv::imwrite("ELSED_long.png", img);


  img = cv::imread(argv[1]);

  // Not using jumps (short segments)
  upm::ELSEDParams params;
  params.listJunctionSizes = {};
  tm.reset();
  tm.start();
  upm::ELSED elsed_short(params);
  segs = elsed_short.detect(img);
  tm.stop();
  std::cout << "ELSED detected: " << segs.size() << " (short) segments : " << tm << std::endl;

  drawSegments(img, segs, CV_RGB(0, 255, 0), 2);
  cv::imwrite("ELSED_short.png", img);

  img = cv::imread(argv[1]);
  Mat gray;
  cvtColor(img, gray, COLOR_BGR2GRAY);

  tm.reset();
  tm.start();
  Ptr<EdgeDrawing> ed = createEdgeDrawing();
  //ed->params.EdgeDetectionOperator = EdgeDrawing::SOBEL;
  //ed->params.GradientThresholdValue = 36;
  //ed->params.AnchorThresholdValue = 8;

  // Detect edges
  //you should call this before detectLines() and detectEllipses()
  ed->detectEdges(gray);
  tm.stop();
  std::cout << "ED detectEdges()                     : " << tm << std::endl;

  tm.reset();
  tm.start();
  ed->detectEdges(gray);
  tm.stop();
  std::cout << "ED detectEdges() (second call)       : " << tm << std::endl;

  // Detect lines
  vector<Vec4f> lines;
  tm.reset();
  tm.start();
  ed->detectLines(lines);
  tm.stop();
  std::cout << "ED detectLines()                     : " << tm << std::endl;

  tm.reset();
  tm.start();
  ed->detectLines(lines);
  tm.stop();
  std::cout << "ED detectLines() (second call)       : " << tm << std::endl;

  drawSegments(img, lines, CV_RGB(0, 255, 0), 2);
  cv::imwrite("ED_lines.png", img);

  return 0;
}