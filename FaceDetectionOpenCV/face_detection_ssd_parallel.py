# importing required libaries 
import cv2 
import numpy as np  
from imutils.video import WebcamVideoStream, FPS # pip install imutils (if imutils library not already installed)

# defining parameters and helper functions for performing face detection
model_config_filepath   = 'deploy.prototxt.txt' 
model_weights_filepath  = 'res10_300x300_ssd_iter_140000.caffemodel'
confidence_thresh       = 0.9 # threshold for filtering weak detections 

# loading inference model using cv2's dnn module 
model = cv2.dnn.readNet(model=model_weights_filepath, config=model_config_filepath) 

# defining function for detecting faces in a single input image/video frame 
def detect_faces(frame):
    # original frame resolution 
    orig_h, orig_w = frame.shape[:2]
    
    # preprocessing input frame
    h , w = 300 , 300 # required height and width after resizing 
    resized_frame = cv2.resize(frame, (w,h) ) # performing resizing 
    # performing mean subtraction and reshaping to a blob/image of shape 1x3x300x300 
    blob = cv2.dnn.blobFromImage(resized_frame, scalefactor=1, size=(w,h), mean=(104, 177, 123)) # mean values are in BGR ordering 
    
    # performing inference 
    model.setInput(blob) 
    detections = model.forward() # returned detections are of shape (1,1,num_detections,7). 
    
    # looping over all detections and annotating input frame with high confidence detections 
    for i in range(detections.shape[2]) :
        confidence = detections[0,0,i,2]   # index 2 stores the confidence/probability of the detection
        if confidence < confidence_thresh :
            continue
        
        # indices 3,4,5,6 store the bounding box coordinates in order [xmin, ymin, xmax, ymax] with values in the range 0-1 
        bbox = detections[0,0,i,3:7] * np.array([orig_w, orig_h, orig_w, orig_h]) #  scaling bounding box coordinates back to original frame dimensions 
        bbox = bbox.astype(np.int) # type casting and rounding to int type 
        cv2.rectangle(frame, (bbox[0], bbox[1]) , (bbox[2], bbox[3]) , (0,0,255) , 2) # drawing rectangular bounding boxes around detections 
        
    return frame 

# setting up input video stream for reading from webcam 
webcam_stream = WebcamVideoStream(0) # opening video stream from primary camera 
webcam_stream.start()
fps = FPS() # for computing frames processed per second 

# processing video frames 
fps.start()
while True :
    # reading next frame from input stream 
    frame = webcam_stream.read()
    fps.update()
    
    # detecting faces in the read frame  
    frame_with_detections = detect_faces(frame)

    # displaying the frame 
    cv2.imshow('Detected Faces', frame_with_detections)
    key_pressed = cv2.waitKey(1) # a 1 millisecond delay 
    if key_pressed  == ord('q'):
        break 
fps.stop()

# closing open streams, etc
webcam_stream.stop()
cv2.destroyAllWindows()

# printing stats - fps
print("FPS:{}".format(fps.fps()))
