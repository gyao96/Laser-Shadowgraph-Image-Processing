import cv2
import numpy as np
import matplotlib.pyplot as plt
from MyDIPUtils.counting_utils import *
import pdb
from MyDIPUtils.config import *

def count_circles(img, step_size=None, draw=True, check_cr=True, check_ol1=True, check_ol2=True, bw_flip=True, debug=False):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bw_flip:
        img = cv2.bitwise_not(img)
    img = np.pad(img, 5, mode='constant', constant_values=255)

    # shape in cv2.resize is an exception to general representation of image shape
    factor = target_size/max(img.shape[0], img.shape[1])
    scaled = cv2.resize(img, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    pad_num = (target_size-min(scaled.shape[0], scaled.shape[1]))//2
    if min(img.shape[0], img.shape[1]) == img.shape[0]:
        scaled = np.pad(scaled, [[pad_num, pad_num], [0, 0]], mode='constant', constant_values=255)
    else:
        scaled = np.pad(scaled, [[0, 0], [pad_num, pad_num]], mode='constant', constant_values=255)

    blur = cv2.GaussianBlur(scaled, (7, 7), 0)
    blur = cv2.medianBlur(blur, 5)

    _, bi = cv2.threshold(blur, 250, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erode = cv2.erode(bi, kernel)
    erode[erode == bi] = 255


    if step_size == None:
        step_size = [default_sample_step]
    output = []
    for sample_step in step_size:
        points_clk = direction_map(erode, True)
        points = list(map(lambda x: x[1], points_clk))
        points_sample = points[::sample_step]
        points_sample += [points_sample[-1]]
        if debug:
            imgshow(draw_points(points_sample, np.full_like(erode, 255, np.uint8)))
        final = find_feature_points2(points_sample)
        if len(final) < 3:
            return [1]
        if debug:
            test = erode.copy()
            for point in final:
                cv2.circle(test, point, 3, 0, cv2.FILLED)
            imgshow(test)

        para = {'edge': erode, 'points': final, 'img': bi}
        output.append(find_all_circles(check_close_radius=check_cr, check_overlapping_1=check_ol1, check_overlapping_2=check_ol2, **para))

    best_res = 0
    best = -1
    for out in output:
        tmp = check_ol3(out[0], bi)
        if tmp > best_res:
            best_res = tmp
            best = out

    if draw:
        plt.figure()
        plt.imshow(best[1], cmap='gray')
        plt.show()

    return best[0]


if __name__ == '__main__':
    img = cv2.imread('gray_217_1.png')
    count_circles(img, step_size=step_size)




