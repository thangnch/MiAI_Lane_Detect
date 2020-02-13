import numpy as np
import cv2


# Ham tinh toan duong thang
def getLines(offset1, offset2, rho, theta, direct=1):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 - offset1 * (-b))
    y1 = int(y0 - offset1 * a)
    x2 = int(x0 + direct * offset2 * (-b))
    y2 = int(y0 + direct * offset2 * a)
    return x1, y1, x2, y2


# Ham tim lan duong tu anh dau vao
def findLaneLines(crop, edged):
    # Tim tat ca cac lines trong hinh
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 25)

    # Dinh nghia cac bien luu
    rho_left = []
    theta_left = []
    rho_right = []
    theta_right = []

    # Neu nhu tim duoc lines
    if lines is not None:

        # Lap qua cac line
        for i in range(0, len(lines)):
            for rho, theta in lines[i]:
                # Left line la cac line co theta nam trong khoang [pi/2 , pi/4]
                if np.pi / 2 > theta > np.pi / 4:
                    rho_left.append(rho)
                    theta_left.append(theta)

                # Right line la cac line co theta nam trong khoang [pi/2, 3*pi/4]
                if np.pi / 2 < theta < 3 * np.pi / 4:
                    rho_right.append(rho)
                    theta_right.append(theta)

    # Tinh trung binh de lay line trai va line phai
    left_rho = np.median(rho_left)
    left_theta = np.median(theta_left)
    right_rho = np.median(rho_right)
    right_theta = np.median(theta_right)

    # Ve line trai
    if left_theta > np.pi / 4:
        x1, y1, x2, y2 = getLines(180, 800, left_rho, left_theta, 1)
        cv2.line(crop, (x1, y1), (x2, y2), (0, 0, 255), 6)

    # Ve line phai
    if right_theta > np.pi / 4:
        x3, y3, x4, y4 = getLines(250, 800, right_rho, right_theta, -1)
        cv2.line(crop, (x3, y3), (x4, y4), (0, 0, 255), 6)

    # Ve lan duong
    if left_theta > np.pi / 4 and right_theta > np.pi / 4:
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
        overlay = crop.copy()
        cv2.fillConvexPoly(overlay, pts, (255, 0, 0))
        opacity = 0.5
        cv2.addWeighted(overlay, opacity, crop, 1 - opacity, 0, crop)

    return crop


# Read video from file
vid_file = "xe10s.mp4"
video = cv2.VideoCapture(vid_file)

# Read frame by frame
while video.isOpened():

    ret, frame = video.read()
    # Check if frame is read ok
    if not ret:
        break

    cv2.imshow("Input", frame)

    # Crop and display main ROI view of video
    crop = frame[350:550, 300:950]
    mask = np.zeros((crop.shape[0], crop.shape[1]), dtype="uint8")
    pts = np.array([[25, 190], [200, 50], [340, 50], [650, 190]], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    masked = cv2.bitwise_and(crop, crop, mask=mask)
    cv2.imshow("Main View", masked)

    # Apply threshold
    frame = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    thresh = 80
    frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)[1]

    # Find edge
    blurred = cv2.GaussianBlur(frame, (13, 13), 0)
    edged = cv2.Canny(blurred, 30, 150)

    # perform find lane lines
    crop = findLaneLines(crop, edged)
    cv2.imshow("Output", crop)

    if cv2.waitKey(1) == 27:
        break

# clear everything once finished
video.release()
cv2.destroyAllWindows()
