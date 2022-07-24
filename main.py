import numpy as np
from PIL import Image
import cv2
import pytesseract


def processimage(image):
    originalImage = cv2.imread(image)

    grayImg = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    thresh, bwImage = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow('BW', bwImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pytesseract.image_to_string(bwImage)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import cv2
    import numpy as np
    img = cv2.imread('test.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cv2.imshow("Edged", blurred)
    print('out', pytesseract.image_to_string(blurred))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
