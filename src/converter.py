import cv2
import numpy as np
import config


class Converter:
    def __init__(self):
        self.model_path = config.model_path
        self.pro_text_path = config.pro_text_path
        self.points_path = config.points_path

        self.network = cv2.dnn.readNetFromCaffe(self.pro_text_path, self.model_path)
        self.points = np.load(self.points_path)

        self.class8 = self.network.getLayerId("class8_ab")
        self.conv8 = self.network.getLayerId("conv8_313_rh")

        self.points = self.points.transpose().reshape(2, 313, 1, 1)

        self.network.getLayer(self.class8).blobs = [self.points.astype("float32")]
        self.network.getLayer(self.conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    def convert(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        scaled = img.asType("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        self.network.setInput(cv2.dnn.blobFromImage(L))
        ab = self.network.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

        L = cv2.split(lab)[0]
        colorized = self.network.concatenate((L[:, :, np.newaxis, ab]), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
        colorized = np.clip(colorized, 0, 1)

        colorized = (255 * colorized).asType("uint8")

        return colorized


