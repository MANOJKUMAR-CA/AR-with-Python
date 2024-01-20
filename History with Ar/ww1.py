import cv2
import cv2.aruco as aruco
import numpy as np
import os

cap = cv2.VideoCapture(0)
video_augment = cv2.VideoCapture("markers/25.mp4")

detection = False
frame_count = 0

height_marker, width_marker = 100, 100
_, image_video = video_augment.read()
image_video = cv2.resize(image_video, (width_marker, height_marker))


def augmentAruco(bbox, img, img_augment):
    top_left = bbox[0][0][0], bbox[0][0][1]
    top_right = bbox[0][1][0], bbox[0][1][1]
    bottom_right = bbox[0][2][0], bbox[0][2][1]
    bottom_left = bbox[0][3][0], bbox[0][3][1]

    height, width, _, = img_augment.shape

    points_1 = np.array([top_left, top_right, bottom_right, bottom_left])
    points_2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    matrix, _ = cv2.findHomography(points_2, points_1)
    image_out = cv2.warpPerspective(img_augment, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, points_1.astype(int), (0, 0, 0))
    image_out = img + image_out

    return image_out

while True:
    success, frame = cap.read()
    # frame = cv2.rotate(frame,cv2.ROTATE_180)

    if detection == False:
        video_augment.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
    else:
        if frame_count == video_augment.get(cv2.CAP_PROP_FRAME_COUNT):
            video_augment.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
        _, image_video = video_augment.read()
        image_video = cv2.resize(image_video, (width_marker, height_marker))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    arucoParam = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict,arucoParam)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and ids[0] == 25:
        detection = True
        # aruco.drawDetectedMarkers(frame, corners)
        frame = augmentAruco(np.array(corners)[0], frame, image_video)
        # player = MediaPlayer("5.mp4")  # audio
        # audio_frame, val = player.get_frame() # audio


    cv2.imshow("World War 1",frame)
    # if val != 'eof' and audio_frame is not None:
    #     img , t = audio_frame
    

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_count += 1
  
cv2.destroyAllWindows()

