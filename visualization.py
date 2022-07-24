import numpy as np

class VideoOCROutputBuilder:
    def __init__(self, frame):
        self.maxW = frame.shape[1]
        self.maxH = frame.shape[0]

    def build(self, frame, card=None, ocr=None):
        # initialize card image dimensions w/ OCR image dimensions
        (frameH, frameW) = frame.shape[:2]
        (cardW, cardH) = (0, 0)
        (ocrW, ocrH) = (0, 0)

        if card is not None:
            (cardH, cardW) = card.shape[:2]

        if ocr is not None:
            (ocrH, ocrW) = ocr.shape[:2]

        # compute spacial dimensions of output frame
        outputW = max([frameW, cardW, ocrW])
        outputH = frameH + cardH + ocrH

        # update self spatial dimensions
        self.maxH = max(self.maxH, outputH)
        self.maxW = max(self.maxW, outputW)

        # allocate mem of output image using max dims
        output = np.zeros((self.maxH, self.maxW, 3), dtype="uint8")

        # set frame in output image
        output[0:frameH, 0:frameW] = frame

        if card is not None:
            output[frameH:frameH + cardH, 0:cardW] = card

        if ocr is not None:
            output[frameH + cardH: frameH + cardH + ocrH, 0:ocrW] = ocr

        return output