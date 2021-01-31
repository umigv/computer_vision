# class ZEDRecording
# cam = ZEDRecording('HD_whatever.svo')
# cam.grab()
import cv2
import sys
import pyzed.sl as sl
import time 

class ZEDRecording:
    
    def __init__(self, filepath):
        input_type = sl.InputType()
        input_type.set_from_svo_file(filepath)
        init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)

        # Create a ZED camera object
        self.zed = sl.Camera()

        #Open the camera
    
        err = self.zed.open(init)
        if (err != sl.ERROR_CODE.SUCCESS):
            exit(-1)

    class GrabError(Exception):
        pass
        
    def grab(self):
        image = sl.Mat()
        grab = self.zed.grab()
        if (grab == sl.ERROR_CODE.SUCCESS):
            # A new image is available if grab() returns SUCCESS
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            # Create am open CV object
            imageCV = image.get_data()
            # Show the image
            return imageCV
        else:
            raise self.GrabError("Failed to grab image")
        
            
    def close(self):
        self.zed.close()

