import numpy as np
import cv2
import cv2.aruco as aruco
#Camera paramters
camera_matrix = np.array([
    [486.6518,    0,     334.4018],  #!!!! R E P L A C E !!!! WHEN REPLACING CAMERA (cameraMatrix from MATLAB)
    [   0,     486.7650, 182.3772],  
    [   0,        0,        1    ]
])

#  !!!! R E P L A C E !!!! WHEN CHANGING CAMERA (Distortion coefficients extracted from matlab)
dist_coeffs = np.array([0.0378, -0.0745, 0, 0, 0])

# Load the video file
video_capture = cv2.VideoCapture('cal_testing2.mp4') #####SOURCE VIDEO

# Create a dictionary of markers
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# Create detector parameters
parameters = cv2.aruco.DetectorParameters()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("End of Video Clip")
        break
    # Convert frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters) #DETECT
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids) #DRAW
    # Pose estimation and axis drawing
    if corners is not None:
        # Estimate pose for each marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.112, camera_matrix, dist_coeffs)

        # Draw the axis on each detected marker
        for rvec, tvec in zip(rvecs, tvecs):
            frame_markers = cv2.drawFrameAxes(frame_markers, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # Extract the x, y, z translation components from tvec
            x, y, z = tvec[0]

            # Display the estimated position on the frame
            position_text = f"Position (x, y, z): ({x:.2f}, {y:.2f}, {z:.2f}) meters"
            cv2.putText(frame_markers, position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display the frame with detected markers and pose estimation
    cv2.imshow('Frame with Pose Estimation', frame_markers)

    if cv2.waitKey(1) & 0xFF == ord('q'): #CLOSE VIDEO WITH 'q'
        break

# Rlease and close
video_capture.release()
cv2.destroyAllWindows()
