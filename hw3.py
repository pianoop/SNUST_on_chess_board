import numpy as np
import cv2 as cv

BLUE = (139, 0, 0)
RED = (0, 0, 255)
GRAY = (192, 192, 192)
# The given video and calibration data
input_file = 'chess.mp4 '
K = np.array([[1.21244393e+03, 0.00000000e+00, 6.80331146e+02],
              [0.00000000e+00, 1.19699883e+03, 3.59772924e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeff = np.array([ 0.33872153, -2.02265297, -0.00551924,  0.01612134,  5.17598023])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(input_file)
assert video.isOpened(), 'Cannot read the given input, ' + input_file

# Prepare a 3D box for simple AR
box_lower1 = board_cellsize * np.array([[3, 3, 0], [0, 3, 0], [0, 5, 0], [1, 5, 0], [1, 6, 0], [3, 6, 0]])
box_upper1 = board_cellsize * np.array([[3, 3, -1], [0, 3, -1], [0, 5, -1], [1, 5, -1], [1, 6, -1], [3, 6, -1]])
box_lower2 = board_cellsize * np.array([[3, 3, 0],[3, 0, 0], [7, 0, 0], [7, 3, 0], [6, 3, 0], [5, 2, 0], [4, 3, 0]])
box_upper2 = board_cellsize * np.array([[3, 3, -1],[3, 0, -1], [7, 0, -1], [7, 3, -1], [6, 3, -1], [5, 2, -1], [4, 3, -1]])
box_lower3 = board_cellsize * np.array([[7, 3, 0], [10, 3, 0], [10, 4, 0], [9, 4, 0], [9, 5, 0], [10, 5, 0], [10, 6, 0], [7, 6, 0]])
box_upper3 = board_cellsize * np.array([[7, 3, -1], [10, 3, -1], [10, 4, -1], [9, 4, -1], [9, 5, -1], [10, 5, -1], [10, 6, -1], [7, 6, -1]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    complete, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if complete:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the box on the image
        line_lower1, _ = cv.projectPoints(box_lower1, rvec, tvec, K, dist_coeff)
        line_lower2, _ = cv.projectPoints(box_lower2, rvec, tvec, K, dist_coeff)
        line_lower3, _ = cv.projectPoints(box_lower3, rvec, tvec, K, dist_coeff)
        line_upper1, _ = cv.projectPoints(box_upper1, rvec, tvec, K, dist_coeff)
        line_upper2, _ = cv.projectPoints(box_upper2, rvec, tvec, K, dist_coeff)
        line_upper3, _ = cv.projectPoints(box_upper3, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(line_lower1)], True, BLUE, 2)
        cv.polylines(img, [np.int32(line_upper1)], True, BLUE, 2)
        for b, t in zip(line_lower1, line_upper1):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), BLUE, 2)
        cv.polylines(img, [np.int32(line_lower2)], True, RED, 2)    
        cv.polylines(img, [np.int32(line_upper2)], True, RED, 2)  
        for b, t in zip(line_lower2, line_upper2):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), RED, 2)  
        cv.polylines(img, [np.int32(line_lower3)], True, GRAY, 2)    
        cv.polylines(img, [np.int32(line_upper3)], True, GRAY, 2)  
        for b, t in zip(line_lower3, line_upper3):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), GRAY, 2) 

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) scipy.spatial.transform.Rotation
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()