# Real-Time Face Detection Using OpenCV
1. face_detection_ssd_parallel.py script is used for performing real-time face detection on input stream from webcam connected to a laptop or desktop
2. Pre-trained deep learning model for face detection from OpenCV is used <a href='https://github.com/opencv/opencv/tree/master/samples/dnn' > Link </a>
3. Imutils library is used for reading frames from webcam in a multi-threaded approach for achieving higher FPS <a href='https://github.com/PyImageSearch/imutils'> Link </a>
4. Model architecture is a Single Shot Detector (SSD) framework with a ResNet backbone 
5. Model files are in caffe format
   * deploy.prototxt.txt - defines model architecture 
   * res10_300x300_ssd_iter_140000.caffemodel - contains trained model weights 
6. Model files can also be directly downloaded from OpenCV repository. Useful links: <a href='https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector'> Link1 </a> <a href='https://github.com/opencv/opencv/tree/master/samples/dnn'> Link2 </a> <a href='https://github.com/opencv/opencv/blob/master/samples/dnn/models.yml'> Link3 </a>
  
## Programming language and libraries used
1. Python programming language
2. OpenCV library 
3. Imutils library (install using -> pip install imutils)
4. Other Python libraries including Numpy library

