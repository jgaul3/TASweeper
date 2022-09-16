import math
import cv2
import numpy as np
import pyautogui

import pygetwindow


def scale_image(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


num_array_2 = np.stack(
    [scale_image(cv2.imread(f'templates/pixel_nums/{i}.png', cv2.COLOR_RGBA2GRAY), 2) for i in range(0, 10)],
    axis=0
)

num_array_4 = np.stack(
    [scale_image(cv2.imread(f'templates/pixel_nums/{i}.png', cv2.COLOR_RGBA2GRAY), 4) for i in range(0, 10)],
    axis=0
)


def get_num(in_array, scale):
    if scale == 2:
        num_array = num_array_2
    elif scale == 4:
        num_array = num_array_4
    else:
        raise Exception("invalid scale")

    return np.where(np.all(in_array == num_array, axis=(1, 2)))[0][0]


def main():
    # while True:
    # Capture screen
    # window_title = list(filter(lambda x: x.startswith("Google Chrome Mamono"), pygetwindow.getAllTitles()))[0]
    # 
    # x, y, window_width, window_height = pygetwindow.getWindowGeometry(window_title)

    screen = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGBA2GRAY)
    lv_target = cv2.imread('templates/LV.png', cv2.IMREAD_GRAYSCALE)
    res = cv2.matchTemplate(screen, lv_target, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    top_corner = max_loc[::-1]

    bottom_corner_target = np.ones((101, 101), np.uint8) * 255
    bottom_corner_target[:51, :51] = 0
    res = cv2.matchTemplate(screen, bottom_corner_target, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    bottom_corner = max_loc[1] + 50, max_loc[0] + 50

    screen_height = bottom_corner[0] - top_corner[0]
    screen_width = bottom_corner[1] - top_corner[1]

    def get_game_screen():
        region = (top_corner[1], top_corner[0], screen_width + 1, screen_height + 1)
        img = pyautogui.screenshot(region=region)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2GRAY)

    def get_level(curr_screen):
        lv_array = curr_screen[:lv_target.shape[0], lv_target.shape[1] + 4:lv_target.shape[1] + lv_target.shape[0] - 4]
        return get_num(lv_array, 4)

    screen = get_game_screen()

    print(get_level(screen))

    # https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python

    pass


# def test():
#     # Read the image using imread function
#     image = cv2.imread('templates/LV.png')
#     cv2.imshow('Original Image', image)
#
#     scale_percent = 400  # percent of original size
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     dim = (width, height)
#
#     resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#
#     orb = cv2.ORB_create()
#     kp1, des1 = orb.detectAndCompute(img1, None)
#     kp2, des2 = orb.detectAndCompute(img2, None)
#
#
#     cv2.imshow("Resized image", resized)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     pass


if __name__ == '__main__':
    main()
