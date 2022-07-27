import imutils
import numpy as np
from imutils.video import VideoStream
from PIL import Image
import cv2
import pytesseract
from pytesseract import Output

def displayImage(title, image, text, results):
    img = image.copy()
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:9]

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 100  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)[:5]
    print(lines)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)



    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
    cv2.imshow(title, img)
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def processimage(image):
    # First, we need to get grayscale of image
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Next, blur image
    blurred = cv2.GaussianBlur(gray, (5, 5,), 1)

    # Edge image
    edged = cv2.Canny(blurred, 75, 200)

    invert = cv2.bitwise_not(img)

    results = pytesseract.image_to_data(edged, output_type=Output.DICT)


    print(results["text"])
    displayImage("edged", invert, "Testing!", results)

def get_shape_center(frame):
    (height, width) = frame.shape[:2]
    return height / 2, width / 2

def draw_cube_on_webcam():
    vs = VideoStream(src=1).start()
    print("[INFO] Starting webcam")
    while True:
        # grab frame from webcam
        orig = vs.read()
        frame = imutils.resize(orig, width=1000)
        text = frame.shape
        center = get_shape_center(frame)
        cv2.putText(frame, 'Size' + ''.join(map(lambda x: str(x) + ' ', text)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, 'Center' + ''.join(map(lambda x: str(x) + ' ', center)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(frame, (int(center[1]), int(center[0])), (int(center[0] + 100), int(center[1] + 50)), (255, 0, 0), 2)
        cv2.circle(frame, (int(center[1]), int(center[0])), radius=50, color=(0, 0, 225), thickness=-1)
        cv2.imshow("OUT", frame)
        cv2.setWindowProperty("OUT", cv2.WND_PROP_TOPMOST, 1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    vs.stop()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    draw_cube_on_webcam()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
