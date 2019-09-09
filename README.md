# asv_obstacle_tracking
obstacle tracking on images coming from an autonomous surface vessel

## Building
mkdir build
cd build
cmake ../
make

## Running
From bin diretory launch the program with the following command:

./atracker /path/to/the/detection_file.txt /path/to/the/image_folder ../config/kalman_param.txt

Example:
./atracker ../DS8/detection.txt ../DS8/rgb/ ../config/kalman_param.txt 


