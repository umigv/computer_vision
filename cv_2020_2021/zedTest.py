# Add all the necessary libraries
import sys
import pyzed.sl as sl
import cv2
import time
from zedRecording import ZEDRecording

def main(): 
    if len(sys.argv) != 2:
        print("Please specify path to .svo file.")
        exit()
        
    filepath = sys.argv[1]
    print("Reading SVO file: {0}".format(filepath))

    cam = ZEDRecording(filepath)

    # runtime_parameters = sl.RuntimeParameters()
    while True:
        try:
            grab = cam.grab()
            cv2.imshow("Image", grab)
            cv2.waitKey(0)
        except ZEDRecording.GrabError:
            print("Video ended")
            break

    cam.close()

if __name__ == "__main__":
    main()

        
