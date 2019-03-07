import cv2
import numpy as np
import pandas as pd
import os
from model.model import Model
from config.config import *


def main():
    # Load model
    model_class = Model()
    if os.path.exists(MODEL_NAME):
        model_class.load_model(MODEL_NAME)

    else:
        print("Pls Run train")
        return

    if (IS_LOCAL):
        PATH = "../input/dataset/"
    else:
        PATH = "../input/"

    char_df = pd.read_csv(PATH + 'kmnist_classmap.csv', encoding='utf-8')

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.namedWindow("Canvas")
    global ix, iy, is_drawing
    is_drawing = False

    def paint_draw(event, x, y, flags, param):
        global ix, iy, is_drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_drawing == True:
                cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
                ix = x
                iy = y
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False
            cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
            ix = x
            iy = y
        return x, y

    cv2.setMouseCallback('Canvas', paint_draw)
    while (1):
        cv2.imshow('Canvas', 255 - image)
        key = cv2.waitKey(10)
        if key == ord(" "):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ys, xs = np.nonzero(image)
            min_y = np.min(ys)
            max_y = np.max(ys)
            min_x = np.min(xs)
            max_x = np.max(xs)
            image = image[min_y:max_y, min_x: max_x]

            image = cv2.resize(image, (28, 28))
            image = np.array(image, dtype=np.float32)[None, None, :, :]
            x_shaped_array = image.reshape(1, IMG_ROWS, IMG_COLS, 1)
            out_x = x_shaped_array / 255
            predicted_classes = model_class.model.predict_classes(out_x)
            target_names = ["Class {} ({}):".format(predicted_classes[0],
                                                    char_df[char_df['index'] == predicted_classes[0]]['char'].item())]
            print(target_names)
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            ix = -1
            iy = -1


if __name__ == '__main__':
    main()
