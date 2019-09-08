#include <iostream>
#include <memory>
#include <string>
#include <fstream>

#include "track.h"
#include "tracker.h"
#include "imagemanager.h"
#include "kalman_param.h"

#include <opencv2/highgui/highgui.hpp>  // Video write

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;


string PRED_FOLDER("../DS8/pred");


Mat drawOccupancyGrid(int i, cv::Mat& occupancy_image, Entities tracks);


std::map<int, std::vector< std::vector< std::string > > > petsReading(const std::string& _gt)
{
    std::ifstream file(_gt);
    if(!file.is_open())
    {
      std::cerr << "Error: cannot read the file: " << _gt << std::endl;
      exit(-1);
    }
    
    
    std::string line;
    std::string delimiter(",");
    std::map<int, std::vector< std::vector< std::string > > > pets;
   

    while (std::getline(file, line))
    {
      if(line.size() == 0) continue;
      auto start = 0U;
      auto end = line.find(delimiter);
      std::vector<std::string> row;
      while (end != std::string::npos)
      {
	row.push_back(line.substr(start, end - start));
	start = end + delimiter.length();
	end = line.find(delimiter, start);
      }
      row.push_back(line.substr(start, end - start));
      
      const uint& n = atoi(row[1].c_str());
      std::vector< std::vector<std::string> > detections;
      uint j = 2;
      for(uint i = 0; i < n; ++i)
      {
	std::vector<std::string> currDetections;
	try
	{
	  currDetections.push_back(row[j]);
	  currDetections.push_back(row[++j]);
	  currDetections.push_back(row[++j]);
	  currDetections.push_back(row[++j]);
	}
	catch(...)
	{
	  std::cerr << "Error: cannot read parse:\n " << line << std::endl;
	  exit(-1);
	}
	++j;
	detections.push_back(currDetections);
      }
  
      pets.insert(std::make_pair(atoi(row[0].c_str()), detections));
    }

    return pets;
}

int main(int argc, char** argv)
{
  if(argc != 4)
  {
    std::cerr << "Usage: " << std::endl;
    std::cout << "\t ./" << argv[0] << " detection_file_name.txt image_folder kalman_param.txt" << std::endl;
  }
  
  std::map<int, std::vector< std::vector< std::string > > > detections = petsReading(argv[1]);
  cv::Mat image, imageClone, imageTracks;
  cv::Mat rgb_image;
  
  std::vector< std::vector < std::string > > curr;
  ImageManager img(argv[2]);
  std::string param_filename(argv[3]);
  ATracker::KalmanParam param;
  param.read(param_filename);
  
  ATracker::Tracker tr(param);
  
  std::vector<cv::Rect> rects;
  std::vector<cv::Point2f> points;
  std::vector<ATracker::Detection> dets;
  
  cv::Rect rect;
  const double& milliseconds = 1000 / 7;
  
  cv::VideoWriter video_d("detection.avi", CV_FOURCC('M','J','P','G'), 3, cv::Size(640,480));
  cv::VideoWriter video_t("tracking.avi", CV_FOURCC('M','J','P','G'), 3, cv::Size(640,480));
  cv::VideoWriter video_o("occupancy.avi", CV_FOURCC('M','J','P','G'), 3, cv::Size(640,480));
  
  for(uint i = 0; i < detections.size(); ++i)
  {
    rects.clear();
    points.clear();
    image = cv::imread(img.next(1));
    imageClone = image.clone();
    rgb_image = image.clone();
    imageTracks = image.clone();
    int w = image.cols;
    int h = image.rows;
    
    curr = detections[i+1];
    std::stringstream ss;
    int j = 0;
    for(const auto &c : curr)
    {
      rect = cv::Rect(cvRound(atof(c[0].c_str())), cvRound(atof(c[1].c_str())),
                  cvRound(atof(c[2].c_str())), cvRound(atof(c[3].c_str())));
      
      rects.push_back(rect);
      
      points.push_back(cv::Point2f(rect.x + (rect.width >> 2), rect.y + rect.height));
      ATracker::Detection d(rect.x + (rect.width >> 2), rect.y + rect.height, rect.width, rect.height);
      dets.push_back(d);
      
      cv::rectangle(imageClone, rect, cv::Scalar(0, 0, 255), 2 );
      
      ss.str("");
      ss << j;
      
      //cv::putText(imageClone, ss.str(), cv::Point(cvRound(atof(c[0].c_str())), cvRound(atof(c[1].c_str()))), cv::FONT_HERSHEY_SIMPLEX,
      //		  0.55, cv::Scalar(0, 255, 0), 2, CV_AA);
      
      ++j;
    }
    tr.setSize(w, h);
    tr.track(dets, w, h, image);
    
    Entities tracks = tr.getTracks();
    for(auto& track : tracks)
    {
      track->drawTrack(imageTracks);
    }

    Mat occupancy_image = drawOccupancyGrid(i, rgb_image, tracks);

    cv::imshow("DETECTIONS", imageClone);
    cv::imshow("TRACKS", imageTracks);
    video_d.write(imageClone);
    video_t.write(imageTracks);
    video_o.write(occupancy_image);
    
    cv::waitKey(cvRound(milliseconds));
    dets.clear();  
  }
  video_d.release();
  video_t.release();
  video_o.release();
}


Mat drawOccupancyGrid(int i, cv::Mat& rgb_image, Entities tracks) {

  char window[] = "occupancy map";

  std::string n;
  std::stringstream ss;
  ss << (i+1);
  n = ss.str();

  Mat rgb = rgb_image;

  if(!rgb.data)                              
  {
      cout <<  "Could not open or find the rgb image" << std::endl ;
      exit(-1);
  }

  Mat pred = imread(PRED_FOLDER + "//" + n +".png", CV_LOAD_IMAGE_COLOR);

  if(!pred.data)                              
  {
      cout <<  "Could not open or find the horizon image" << std::endl ;
      exit(-1);
  }

  imshow("pred",pred);

  Mat occupancy_image = rgb.clone();

  int limit = 0;

  for(int j = 0; j < pred.cols; j++) { 
      for(int i = 0; i < pred.rows; i++) {            
          if(pred.at<cv::Vec3b>(i,j)[0] == 255 &&
             pred.at<cv::Vec3b>(i,j)[1] == 0 &&
             pred.at<cv::Vec3b>(i,j)[2] == 255 )
          {
              occupancy_image.at<cv::Vec3b>(i,j)[0] = 255;
              occupancy_image.at<cv::Vec3b>(i,j)[1] = 0;
              occupancy_image.at<cv::Vec3b>(i,j)[2] = 255;
              if(i > limit)
                  limit = i;
          }
      }
  }
  
  cout << limit << endl;

  Rect first(0, limit, occupancy_image.cols/4, (occupancy_image.rows-limit)/3);
  cout << first << endl;
  Rect second(occupancy_image.cols/4, limit, occupancy_image.cols/2, (occupancy_image.rows-limit)/3);
  cout << second << endl;
  Rect third(occupancy_image.cols*0.75, limit, occupancy_image.cols/4, (occupancy_image.rows-limit)/3);
  cout << third << endl;
  Rect fourth(0, limit+((occupancy_image.rows-limit)*0.34), occupancy_image.cols/4, (occupancy_image.rows-limit)*0.67);
  cout << fourth << endl;
  Rect fifth(occupancy_image.cols/4, limit+((occupancy_image.rows-limit)*0.34), occupancy_image.cols/2, (occupancy_image.rows-limit)*0.67);
  cout << fifth << endl;
  Rect sixth(occupancy_image.cols*0.75, limit+((occupancy_image.rows-limit)*0.34), occupancy_image.cols/4, (occupancy_image.rows-limit)*0.67);
  cout << sixth << endl;

  /*
line(occupancy_image,
       Point(0, limit),
       Point(occupancy_image.cols-1, limit),
       Scalar(0,0,0),
       2);
*/
  
  /*
  rectangle(occupancy_image,first,Scalar(0,0,0), 2);
  rectangle(occupancy_image,second,Scalar(0,0,0), 2);
  rectangle(occupancy_image,third,Scalar(0,0,0), 2);
  rectangle(occupancy_image,fourth,Scalar(0,0,0), 2);
  rectangle(occupancy_image,fifth,Scalar(0,0,0), 2);
  rectangle(occupancy_image,sixth,Scalar(0,0,0), 2);
  */

  bool occupied_1 = false;
  bool occupied_2 = false;
  bool occupied_3 = false;
  bool occupied_4 = false;
  bool occupied_5 = false;
  bool occupied_6 = false;

  for(auto& track: tracks)
  {
      
      Rect r = track->getRect();
      Point br = r.br();
      int x0 = br.x-(r.width/2);
      int y0 = br.y;

      cout << "track: " << r << endl;
      cout << x0 << " " << y0 << endl;

      int a, b, c, d;
      
      //first
      a = first.tl().x;
      b = first.tl().y;
      c = first.width;
      d = first.height;
      if(x0 > a && x0 < (a+c) &&
         y0 > b && y0 < (b+d) )
      {
           rectangle(occupancy_image,first,Scalar(0,0,255), 4);
           occupied_1 = true;  
      }
      else {
           if(!occupied_1)
               rectangle(occupancy_image,first,Scalar(0,255,0), 2);  
      }
            
      //second
      a = second.tl().x;
      b = second.tl().y;
      c = second.width;
      d = second.height;

      if(x0 > a && x0 < (a+c) &&
         y0 > b && y0 < (b+d) )
      {
           rectangle(occupancy_image,second,Scalar(0,0,255), 4); 
           
           occupied_2 = true;
      }
      else {
           if(!occupied_2)
               rectangle(occupancy_image,second,Scalar(0,255,0), 2);  
      }

      //third
      a = third.tl().x;
      b = third.tl().y;
      c = third.width;
      d = third.height;
      if(x0 > a && x0 < (a+c) &&
         y0 > b && y0 < (b+d) )
      {
           rectangle(occupancy_image,third,Scalar(0,0,255), 4); 
           occupied_3 = true; 
      }
      else {
           if(!occupied_3)
               rectangle(occupancy_image,third,Scalar(0,255,0), 2);  
      }
      //fourth
      a = fourth.tl().x;
      b = fourth.tl().y;
      c = fourth.width;
      d = fourth.height;
      if(x0 > a && x0 < (a+c) &&
         y0 > b && y0 < (b+d) )
      {
           rectangle(occupancy_image,fourth,Scalar(0,0,255), 4);
           occupied_4 = true;  
      }
      else {
           if(!occupied_4)
               rectangle(occupancy_image,fourth,Scalar(0,255,0), 2);  
      }
      //fifth
      a = fifth.tl().x;
      b = fifth.tl().y;
      c = fifth.width;
      d = fifth.height;
      if(x0 > a && x0 < (a+c) &&
         y0 > b && y0 < (b+d) )
      {
           rectangle(occupancy_image,fifth,Scalar(0,0,255), 4);
           occupied_5 = true;  
      }
      else {
           if(!occupied_5)
               rectangle(occupancy_image,fifth,Scalar(0,255,0), 2);  
      }
      //sixth
      a = sixth.tl().x;
      b = sixth.tl().y;
      c = sixth.width;
      d = sixth.height;
      if(x0 > a && x0 < (a+c) &&
         y0 > b && y0 < (b+d) )
      {
           rectangle(occupancy_image,sixth,Scalar(0,0,255), 4);
           occupied_6 = true;  
      }
      else {
           if(!occupied_6)
               rectangle(occupancy_image,sixth,Scalar(0,255,0), 2);  
      }
      
      track->drawTrack(occupancy_image);
  }

  imshow(window, occupancy_image );
  moveWindow( window, 200, 200 );
  waitKey( 30 );
  return occupancy_image;
}

