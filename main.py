import imutils
import keyboard
import math
import numpy as np
from imutils.video import VideoStream
import cv2
import pytesseract
from PIL import ImageTk
from PIL import Image
from tkinter import Tk, Label

from visualization import VideoOCROutputBuilder


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta

def get_shape_center(frame):
    (height, width) = frame.shape[:2]
    return height / 2, width / 2


def draw_cube_on_webcam():
    vs = VideoStream(src=1).start()
    orig = vs.read()
    frame = imutils.resize(orig, width=1000)
    print("[INFO] Starting webcam")
    outputBuilder = None

    ws = Tk()
    ws.title('ScrabbleBuddy')
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(img)
    panel = Label(ws, image=img)
    panel.pack()

    ws.mainloop()

    current_frame = None
    was_space_pressed = False

    card = None

    while True:
        # grab frame from webcam
        orig = vs.read()
        frame = imutils.resize(orig, width=1000)
        orig_frame = frame.copy()

        # Show frame size and get center, assign white
        cy, cx = get_shape_center(frame)

        if outputBuilder is None:
            outputBuilder = VideoOCROutputBuilder(frame)

        # edge size is hypotenuse, a is opposit, b is adjacent
        edge_size = 200
        deg30 = 30 * (math.pi / 180)
        a = math.sin(deg30) * edge_size
        b = math.cos(deg30) * edge_size
        white = (255, 255, 255)
        adjust_center = 20

        # Create cube sides with polylines
        left_face = np.array([[cx - b, cy - a], [cx, cy - adjust_center], [cx, cy + (edge_size - adjust_center)],
                              [cx - b, cy + edge_size - a]], dtype=np.int32)
        right_face = np.array([[cx, cy - adjust_center], [cx + b, cy - a], [cx + b, cy + edge_size - a],
                               [cx, cy + (edge_size - adjust_center)]], dtype=np.int32)
        top_face = np.array(
            [[cx - b, cy - a], [cx, cy - a - (a - adjust_center)], [cx + b, cy - a], [cx, cy - adjust_center]],
            dtype=np.int32)

        # put faces of cube on frame
        cv2.polylines(frame, [left_face], True, white, thickness=3)
        cv2.polylines(frame, [right_face], True, white, thickness=3)
        cv2.polylines(frame, [top_face], True, white, thickness=3)

        if keyboard.is_pressed('space'):
            if not was_space_pressed:
                # SCAN

                # First set destination points for perspective transform
                t_dst = np.array([[0, 50], [50, 0], [100, 50], [50, 100]], dtype=np.float32)
                lr_dst = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

                # Copy src points to float array
                t_src = top_face.astype('float32')
                l_src = left_face.astype('float32')
                r_src = right_face.astype('float32')

                # Next get perspective transform matrices
                tM = cv2.getPerspectiveTransform(t_src, t_dst)
                lM = cv2.getPerspectiveTransform(l_src, lr_dst)
                rM = cv2.getPerspectiveTransform(r_src, lr_dst)

                t_warp = cv2.warpPerspective(orig_frame, tM, (100, 100))

                # These vars are the perspective changes for each side of cube
                l_warp = cv2.warpPerspective(orig_frame, lM, (100, 100))
                r_warp = cv2.warpPerspective(orig_frame, rM, (100, 100))
                t_mask = get_face(t_dst, t_warp)

                # l_warp = automatic_brightness_and_contrast(l_warp)
                # r_warp = automatic_brightness_and_contrast(r_warp)
                # t_mask = automatic_brightness_and_contrast(t_mask)

                # Go to grayscale
                # l_warp = cv2.cvtColor(l_warp, cv2.COLOR_BGR2GRAY)
                # r_warp = cv2.cvtColor(r_warp, cv2.COLOR_BGR2GRAY)
                # t_mask = cv2.cvtColor(t_mask, cv2.COLOR_BGR2GRAY)
                #
                # # Turn back into 3 channels
                # l_warp = cv2.cvtColor(l_warp, cv2.COLOR_GRAY2BGR)
                # r_warp = cv2.cvtColor(r_warp, cv2.COLOR_GRAY2BGR)
                # t_mask = cv2.cvtColor(t_mask, cv2.COLOR_GRAY2BGR)
                #
                # # Run through gaussian blur to smoothen
                # l_warp = cv2.GaussianBlur(l_warp, (5, 5), 0)
                # r_warp = cv2.GaussianBlur(r_warp, (5, 5), 0)
                # t_mask = cv2.GaussianBlur(t_mask, (5, 5), 0)
                #
                # # Sharpen image
                # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                # l_warp = cv2.filter2D(l_warp, -1, kernel)
                # r_warp = cv2.filter2D(r_warp, -1, kernel)
                # t_mask = cv2.filter2D(t_mask, -1, kernel)
                #
                # l_warp = cv2.cvtColor(l_warp, cv2.COLOR_BGR2RGB)
                # r_warp = cv2.cvtColor(r_warp, cv2.COLOR_BGR2RGB)
                # t_mask = cv2.cvtColor(t_mask, cv2.COLOR_BGR2RGB)

                l_result = pytesseract.image_to_string(l_warp)
                print("RESULTS: ", l_result)

                card = build_card(t_mask, r_warp, l_warp)

                was_space_pressed = True
        else:
            was_space_pressed = False
        # Show frame
        output = outputBuilder.build(frame, card, None)
        # cv2.imshow("OUT", output)
        # cv2.setWindowProperty("OUT", cv2.WND_PROP_TOPMOST, 1)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     break
        if panel is None:
            pan = Label(imag)

    vs.stop()
    # cv2.destroyAllWindows()

def get_face(face, frame):
    face = face.astype('int32')
    rect = cv2.boundingRect(face)
    x, y, w, h = rect
    cropped = frame[y:y + h, x:x + w].copy()
    top_face = face - face.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [top_face], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    return dst

def build_card(t, r, l):
    th, tw = t.shape[:2]
    lh, lw = l.shape[:2]
    rh, rw = r.shape[:2]

    card = np.zeros((max(th, lh, rh), tw + lw + rw, 3), dtype="uint8")
    card[0:th, 0:tw] = t
    card[0:lh, tw:tw + lw] = l
    card[0:rh, tw + lw:tw + lw + rw] = r

    return card




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    draw_cube_on_webcam()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
