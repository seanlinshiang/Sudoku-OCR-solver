import cv2
import numpy as np
import sys
import imutils
from tensorflow.keras.models import load_model
from solver import *

INPUT_SIZE = 48


def main():
    args = sys.argv[1:]
    if len(args) == 0:
        return
    img = cv2.imread(args[0])
    # resized = cv2.resize(img, (1000, 1000))
    resized = img
    cv2.imshow("img", resized)

    board, location = find_board(resized)
    cv2.imshow("board", board)
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    rois = split_boxes(gray)
    rois = np.array(rois).reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)

    predicted_numbers = predict_numbers(rois)
    board_num = np.array(predicted_numbers).astype("uint8").reshape(9, 9)

    if solve(board_num):
        print_board(board_num)
    else:
        print("No solution")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_board(img):
    """Takes an image as input and finds a sudoku board inside of the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
    cv2.imshow("Contour", newimg)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None
    # Finds rectangular contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(img, location)

    return result, location


def get_perspective(img, location, height=900, width=900):
    """Takes an image and location of an interesting region.
    And return the only selected region with a perspective transformation"""
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result


def split_boxes(board):
    """Takes a sudoku board and split it into 81 cells.
    each cell contains an element of that board either given or an empty cell."""
    rows = np.vsplit(board, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = cv2.resize(box, (INPUT_SIZE, INPUT_SIZE)) / 255.0
            cv2.imshow("Splitted block", box)
            cv2.waitKey(50)
            boxes.append(box)
    return boxes


def predict_numbers(rois):
    classes = np.arange(0, 10)
    model = load_model("model-OCR.h5")
    # get prediction
    prediction = model.predict(rois)
    # print(prediction)
    predicted_numbers = []
    # get classes from prediction
    for i in prediction:
        index = np.argmax(i)
        predicted_number = classes[index]
        predicted_numbers.append(predicted_number)
    return predicted_numbers


if __name__ == "__main__":
    main()
