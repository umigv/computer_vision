# edit
from chessboard import ChessBoard
import opencv as cv2
from birdseye import BirdsEye
from lanefilter import LaneFilter
from zedRecording import ZEDRecording

# Step 1: CAMERA CALIBRATION

# Let's initialize 20 chessboards, note that at instatiation, 
# it finds all chessboard corners and object points
chessboards = []
'''
for n in range(20):
  this_path = 'camera_cal/calibration' + str(n + 1) + '.jpg'
  chessboard = ChessBoard(i = n, path = this_path, nx = 9, ny = 6)
  chessboards.append(chessboard)
  '''

this_path = '/home/umarv/computer_vision/testing/Explorer_HD720_SN15835_15-57-49.png'
chessboard = ChessBoard(i = 0, path = this_path, nx = 9, ny = 6)
chessboards.append(chessboard)

# We use these corners and object points (and image dimensions) 
# from all chessboards to estimate the calibration parameters 
points, corners, shape = [], [], chessboards[0].dimensions

for chessboard in chessboards:    
  if chessboard.has_corners: 
    points.append(chessboard.object_points)
    corners.append(chessboard.corners)

r, matrix, distortion_coef, rv, tv = cv2.calibrateCamera(points, corners, shape, None, None)

# Let's store these parameters somewhere so we can use them later
calibration_data = { "camera_matrix": matrix, "distortion_coefficient": distortion_coef }


# Step 2: PERSPECTIVE TRANSFORMATION

source_points = [(580, 460), (205, 720), (1110, 720), (703, 460)]
destination_points = [(320, 0), (320, 720), (960, 720), (960, 0)]

birdsEye = BirdsEye(source_points, destination_points, 
                    matrix, distortion_coef)

undistorted_image = birdsEye.undistort(this_path)
sky_view_image = birdsEye.sky_view(this_path)



# Step 3: GRADIENT AND COLOR THRESHOLDING

p = { 'sat_thresh': 120, 'light_thresh': 40, 'light_thresh_agr': 205,
      'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20 }


laneFilter = LaneFilter(p)

binary = laneFilter.apply(undistorted_image)
lane_pixels = np.logical_and(birdsEye.sky_view(binary), roi(binary))
