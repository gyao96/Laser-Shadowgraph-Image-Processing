import cv2
import numpy as np
import matplotlib.pyplot as plt
from MyDIPUtils.config import *


# check_ol1: circle[i] and circle[j] overlaps too much
# check_ol2: circle[i] lies too much outside original image
# check_ol3: compare the overlapping area between different stepsize

def direction_map(edge, clockwise):
    # edge_pad = np.pad(edge, 1, mode='constant', constant_values=255)
    # pdb.set_trace()
    edge_copy = edge.copy()
    flag = np.zeros_like(edge_copy)

    edge_y, edge_x = np.nonzero(edge_copy == 0)
    leftmost_x = np.min(edge_x)
    leftmost_y = np.max(edge_y[edge_x == leftmost_x])
    original_point = (leftmost_x, leftmost_y)

    points = []
    neigh = edge_copy[leftmost_y-1:leftmost_y+2, leftmost_x-1:leftmost_x+2]
    if not clockwise:
        direction = 0 if neigh[1, 2] == 0 else 7/4
        if direction == 0:
            next_point = (leftmost_x+1, leftmost_y)
        else:
            next_point = (leftmost_x+1, leftmost_y+1)
    else:
        direction = 0 if neigh[1, 2] == 0 else 1/4
        if direction == 0:
            next_point = (leftmost_x+1, leftmost_y)
        else:
            next_point = (leftmost_x+1, leftmost_y-1)

    points.append((direction, original_point))
    # flag[leftmost_y, leftmost_x] = 1
    while next_point != original_point:
        x, y = next_point
        neigh = edge_copy[y-1:y+2, x-1:x+2]
        flag_neigh = flag[y-1:y+2, x-1:x+2]
        this_point = next_point
        direction, next_point = find_next_direction(neigh, this_point, flag_neigh)
        points.append((direction, this_point))
        flag[this_point[1], this_point[0]] = 1
        # dir_map[y, x] = direction

    return points


def find_next_direction(neigh, this_point, flag_neigh):
    x, y = this_point
    neigh[flag_neigh==1] = 255
    # 4-neighbour is prior to 8-neighbour
    if neigh[0, 1] == 0:
        return 1/2, (x, y-1)
    if neigh[1, 2] == 0:
        return 0, (x+1, y)
    if neigh[2, 1] == 0:
        return 3/2, (x, y+1)
    if neigh[1, 0] == 0:
        return 1, (x-1, y)

    if neigh[0, 2] == 0:
        return 1/4, (x+1, y-1)
    if neigh[0, 0] == 0:
        return 3/4, (x-1, y-1)
    if neigh[2, 0] == 0:
        return 5/4, (x-1, y+1)
    if neigh[2, 2] == 0:
        return 7/4, (x+1, y+1)

def tangent_line(points, seq_num, img, draw=True):
    k = 0
    angle = 0
    for i in range(1, max_neigh):
        s0 = 0
        for j in range(i):  
            s0 += points[j+seq_num][0]
        angle += s0/i
        k += np.tan(s0*np.pi/(i))
    angle /= (max_neigh-1)
    k /= (max_neigh-1)
    x0, y0 = points[seq_num][1]
    y0 = img.shape[0] - y0
    b = y0-k*x0
    if draw:
        line_point(k, b, img)
    return k, angle, b

def points_sequence(points):
    # points should be passed directly from cv2.goodFeaturesToTrack
    # shape is (N, 1, 2)
    sequence = []
    points = np.squeeze(points)
    leftmost = np.argmin(points[:, 0])

    sequence.append(points[leftmost])

    for direction in ['lr', 'ur', 'ul', 'll']:
        next_point = find_next_anticlock(sequence[-1], points, direction)
        while np.any(next_point) is not None:
            sequence.append(next_point)
            next_point = find_next_anticlock(sequence[-1], points, direction)

    return sequence

def find_next_anticlock(point, points, direction):
    if direction not in ['lr', 'ur', 'ul', 'll']:
        raise ValueError('Unknown direction')

    x, y = point
    if direction == 'lr':
        target = points[points[:, 1] > y]
        if len(target) == 0:
            return None
        return target[np.argmin(target[:, 0])]

    if direction == 'ur':
        target = points[points[:, 0] > x]
        if len(target) == 0:
            return None
        return target[np.argmax(target[:, 1])]

    if direction == 'll':
        target = points[points[:, 0] < x]
        if len(target) == 0:
            return None
        return target[np.argmin(target[:, 1])]

    if direction == 'ul':
        target = points[points[:, 1] < y]
        if len(target) == 0:
            return None
        return target[np.argmax(target[:, 0])]

def find_line(point1, point2, img_size, pb):
    x1, y1 = point1
    x2, y2 = point2
    y1, y2 = img_size - y1, img_size - y2

    if pb == True:
        if np.abs(y1-y2) > l1_norm_threshold:
            k = -(x1-x2)/(y1-y2)
            b = (y1+y2)/2 - k*(x1+x2)/2
        else:
            k = None
            b = (x1+x2)/2
    else:
        if np.abs(x1-x2) > l1_norm_threshold:
            k = (y1-y2)/(x1-x2)
            b = y2 - k*x2
        else:
            k = None
            b = x1
    return k, b

def find_para_line(k, point, img_size):
    if k != None:
        return -k*point[0]+(img_size-point[1])
    else:
        return point[0]


def line_point(k, b, img):
    if k != None:
        if b > 0:
            point1 = (0, img.shape[0] - int(b))
        else:
            point1 = (int(-b/k), img.shape[0])

        if k*img.shape[0] + b > img.shape[0]:
            point2 = (int((img.shape[0] - b)/k), 0)
        else:
            point2 = (img.shape[0], int(img.shape[0] - (k*img.shape[0] + b)))
    else:
        point1 = (b, 0)
        point2 = (b, img.shape[0])

    cv2.line(img, point1, point2, 0)

    # return img

def line_gen_1(k, b, img_size):
    # img[i, j]: i->y, j->x
    if k != None:
        return lambda x, y: k*x-(img_size-y)+b
    else:
        return lambda x, y: x-b

def line_gen_2(k, b, img_size):
    # Warning: if k == None, cannot use this function
    assert k != None
    return lambda x: img_size-(k*x+b)

def distance(x1, y1, x2, y2, norm='l2'):
    if norm == 'l1':
        return min(np.abs(x1-x2), np.abs(y1-y2))
    else:
        return np.sqrt((x1-x2)**2+(y1-y2)**2)

def find_center_and_radius(point1, point2, points, img):
    # 1. find the side of the arc
    k0, b0 = find_line(point1, point2, img.shape[0], pb=False)
    line = line_gen_1(k0, b0, img.shape[0])
    for point in points:
        if not np.any(np.logical_or(point == point1, point == point2)):
            flag = np.sign(line(*point))
            break


    # 2. mask only the interested arc
    arc_ma = np.full_like(img, 255, dtype=np.uint8)
    arc_y, arc_x = np.nonzero(img != 255)
    for i in range(len(arc_x)):
        if flag != np.sign(line(arc_x[i], arc_y[i])):
            arc_ma[arc_y[i], arc_x[i]] = 0

    # 3. further mask only the area between 2 corner point
    k, b = find_line(point1, point2, img.shape[0], pb=True)
    b1, b2 = find_para_line(k, point1, img.shape[0]), find_para_line(k, point2, img.shape[0])
    line1, line2 = line_gen_1(k, b1, img.shape[0]), line_gen_1(k, b2, img.shape[0])
    sgn1, sgn2 = np.sign(line1(*point2)), np.sign(line2(*point1))
    arc_y, arc_x = np.nonzero(arc_ma != 255)
    for i in range(len(arc_x)):
        i_sgn1, i_sgn2 = np.sign(line1(arc_x[i], arc_y[i])), np.sign(line2(arc_x[i], arc_y[i]))
        if sgn1 != i_sgn1 or sgn2 != i_sgn2:
            arc_ma[arc_y[i], arc_x[i]] = 255
    # test = draw_points([tuple(point1), tuple(point2)], arc_ma)
    # line_point(k, b, test)
    # line_point(k0, b0, test)
    # imgshow(test)

    # plt.figure()
    # plt.imshow(arc_ma, cmap='gray')


    # 3.find center and radius
    arc_y, arc_x = np.nonzero(arc_ma == 0)
    len_arc = len(arc_y)
    if len_arc < 5:
        return None

    if k != None:
        lower_x = max((point1[0]+point2[0])//2-max_radius, 0)
        upper_x = min((point1[0]+point2[0])//2+max_radius, img.shape[0])

        line = line_gen_2(k, b, img.shape[0])
        dis_var = []
        dis = []
        for x in range(lower_x, upper_x):
            tmp_dis = []
            y = line(x)
            for i in range(len_arc):
                ay, ax = arc_y[i], arc_x[i]
                tmp_dis.append(distance(x, y, ax, ay))
            dis_var.append(np.var(tmp_dis))
            dis.append(np.mean(tmp_dis))
        cur = np.argmin(dis_var)
        center_x = lower_x + cur
        center_y = int(line(center_x))
        radius = dis[cur]
    else:
        lower_y = max((point1[1]+point2[1])//2-max_radius, 0)
        upper_y = min((point1[1]+point2[1])//2+max_radius, img.shape[0])
        x = b
        dis_var = []
        dis = []
        for y in range(lower_y, upper_y):
            tmp_dis = []
            for i in range(len_arc):
                ay, ax = arc_y[i], arc_x[i]
                tmp_dis.append(distance(x, y, ax, ay))
            dis_var.append(np.var(tmp_dis))
            dis.append(np.mean(tmp_dis))
        cur = np.argmin(dis_var)
        center_x = b
        center_y = lower_y + cur
        radius = dis[cur]

    return (int(center_x), int(center_y)), int(radius)

def check_close(circles):
    flags = [-1 for _ in range(len(circles))]
    count = 0
    for i in range(len(circles)):
        if flags[i] == -1:
            color = count
            count += 1
        else:
            color = flags[i]
        flags[i] = color
        for j in range(len(circles)):
            if j != i and distance(*circles[i][0], *circles[j][0]) < distance_threshold:
                flags[j] = color
    final = []
    for i in range(len(flags)):
        if flags[i] != -1:
            color = flags[i]
            flags[i] = -1
            tmp_center = [circles[i][0]]
            tmp_radius = [circles[i][1]]
            for j in range(i+1, len(flags)):
                if flags[j] == color:
                    tmp_center.append(circles[j][0])
                    tmp_radius.append(circles[j][1])
                    flags[j] = -1
            mean_center = np.mean(tmp_center, axis=0)
            mean_radius = np.mean(tmp_radius)
            final.append(((int(mean_center[0]), int(mean_center[1])), int(mean_radius)))
    return final

def overlapping(circle1, circle2, img_shape):
    tmp1 = np.full(img_shape, 255, dtype=np.uint8)
    tmp2 = np.full(img_shape, 255, dtype=np.uint8)
    cv2.circle(tmp1, circle1[0], circle1[1], 0, cv2.FILLED)
    cv2.circle(tmp2, circle2[0], circle2[1], 0, cv2.FILLED)
    ol = np.full(img_shape, 255, dtype=np.uint8)
    ol[np.logical_and(tmp1==0, tmp2==0)] = 0
    area1 = np.sum(tmp1==0)
    area2 = np.sum(tmp2==0)
    area_ol = np.sum(ol==0)
    return area_ol/area1, area_ol/area2

def check_ol1(circles, shape):
    final = []
    flags = [-1 for _ in range(len(circles))]
    for i in range(len(circles)):
        if flags[i] == -1:
            for j in range(i+1, len(circles)):
                if flags[j] == -1:
                    ol_i, ol_j = overlapping(circles[i], circles[j], shape)
                    if max(ol_i, ol_j) > overlapping1_threshold:
                        if max(ol_i, ol_j) == ol_i:
                            flags[i] = 0
                        else:
                            flags[j] = 0
            if flags[i] == -1:
                final.append(circles[i])
    return final

def check_ol2(circles, ori_img):
    final = []
    for circle in circles:
        tmp = np.full(ori_img.shape, 255, dtype=np.uint8)
        cv2.circle(tmp, circle[0], circle[1], 0, cv2.FILLED)
        ol = np.full(ori_img.shape, 255, dtype=np.uint8)
        ol[np.logical_and(tmp==0, ori_img==0)]=0
        if np.sum(ol==0)/np.sum(tmp==0) > overlapping2_threshold:
            final.append(circle)
    return final

def check_ol3(circles, ori_img):
    tmp = np.full(ori_img.shape, 255, dtype=np.uint8)
    for circle in circles:
        cv2.circle(tmp, circle[0], circle[1], 0, cv2.FILLED)
    intersec = np.full(ori_img.shape, 255, dtype=np.uint8)
    intersec[np.logical_and(tmp==0, ori_img==0)]=0
    ol = np.sum(intersec==0)
    tmp[intersec==0] = 255
    sub = np.sum(tmp==0)

    # return ol/sub or ol-sub?
    # problem...
    return ol-sub

def find_all_circles(check_close_radius=True, check_overlapping_1=True, check_overlapping_2=True, **kwargs):
    # points should be binded with edge
    if 'points' in kwargs.keys():
        points = kwargs['points']
        has_point = True
    else:
        has_point = False

    if 'edge' in kwargs.keys():
        edge = kwargs['edge']
        has_edge = True
    else:
        has_edge = False

    if 'img' in kwargs.keys():
        img = kwargs['img']
        has_img = True
    else:
        has_img = False

    flag_edge = has_point^has_edge
    if flag_edge:
        raise KeyError('Points and edge should be passed concurrently.')
    if has_img == False and has_point == False:
        raise KeyError('Either image or edge should be passed.')
    if has_img == False and check_overlapping_2 == True:
        raise KeyError('Checking overlapping 2 requries original image.')
    flag_edge = has_edge

    if not has_edge:
        img = cv2.GaussianBlur(img,(5,5),0)
        # TODO: integrate erosion 
        edge = cv2.Canny(img, 100, 200)
        edge = cv2.bitwise_not(edge)
        out = edge.copy()

        corners = cv2.goodFeaturesToTrack(img,10,0.1,10)
        white = np.zeros_like(img)
        for i in corners:
            x,y = i.ravel()
            cv2.circle(white,(x,y),3,255,-1)
        points = corners
    else:
        out = edge.copy()
    # sequence = points_sequence(points)
    points += [points[0]]
    sequence = np.array(points, dtype=np.int32)
    circles = []
    if len(sequence) < 3:
        left, right, up, down = find_4_points(edge)
        center = int((right+left)/2), int((down+up)/2)
        radius = int(min((right-left)/2, (down-up)/2))
        return [(center, radius)]

    for i in range(len(sequence) - 1):
        point = find_center_and_radius(sequence[i], sequence[i+1], sequence, edge)
        if point != None and point[1] < max_radius:
            circles.append(point)

    if check_overlapping_2 and has_img:
        circles = check_ol2(circles, img)

    if check_close_radius:
        circles = check_close(circles)

    if check_overlapping_1:
        # check overlapping 1 means check if one deteced circle is mostly inside another detected circle
        # TODO: take average or just choose the larger circle?
        circles = check_ol1(circles, edge.shape)

    for circle in circles:
        center, radius = circle
        cv2.circle(out,center,3,0,-1)
        cv2.circle(out,center,radius,0)

    return circles, out

def imgshow(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

def draw_points(points, img):
    copy = img.copy()
    for point in points:
        cv2.circle(copy, point, 3, 0, cv2.FILLED)
    return copy

def find_feature_points(points_sample):
    slope = []
    for i in range(len(points_sample)-1):
        x1, y1 = points_sample[i]
        x2, y2 = points_sample[i+1]
        theta = np.arctan((y1-y2)/(x1-x2)) if (x1-x2) != 0 else 0.5*np.pi
        print(theta, points_sample[i])
        slope.append(theta)
    interested = []
    for i in range(len(slope)-1):
        diff = np.abs(slope[i]-slope[i+1])
        if diff > np.pi/2:
            diff = np.pi - diff
        if diff > slope_lower:
            print(slope[i]-slope[i+1])
            # imgshow(test)
            interested.append(points_sample[i+1])
            # cv2.circle(erode, points_sample[i], 2, 0, cv2.FILLED)
    flags = [-1 for _ in range(len(interested))]
    count = 0
    for i in range(len(interested)):
        if flags[i] == -1:
            color = count
            count += 1
        else:
            color = flags[i]
        flags[i] = color
        for j in range(len(interested)):
            if j != i and distance(*interested[i], *interested[j]) < distance_threshold:
                flags[j] = color
    final = []
    for i in range(len(flags)):
        if flags[i] != -1:
            color = flags[i]
            flags[i] = -1
            tmp = [interested[i]]
            for j in range(i+1, len(flags)):
                if flags[j] == color:
                    tmp.append(interested[j])
                    flags[j] = -1
            mean = np.mean(tmp, axis=0)
            final.append((int(mean[0]), int(mean[1])))
    return final

def find_feature_points2(points_sample):
    slope = []
    slope_rev = []
    for i in range(len(points_sample)-1):
        x1, y1 = points_sample[i]
        x2, y2 = points_sample[i+1]
        x3, y3 = points_sample[i-1]
        theta = np.arctan((y1-y2)/(x1-x2)) if (x1-x2) != 0 else 0.5*np.pi
        theta_rev = np.arctan((y1-y3)/(x1-x3)) if (x1-x3) != 0 else 0.5*np.pi
        slope.append(theta)
        slope_rev.append(theta_rev)
    interested = []
    for i in range(len(slope)-1):
        diff = np.abs(slope[i]-slope_rev[i])
        if diff > slope_lower and diff < slope_upper:
            # imgshow(test)
            interested.append(points_sample[i])
            # cv2.circle(erode, points_sample[i], 2, 0, cv2.FILLED)
    flags = [-1 for _ in range(len(interested))]
    count = 0
    for i in range(len(interested)):
        if flags[i] == -1:
            color = count
            count += 1
        else:
            color = flags[i]
        flags[i] = color
        for j in range(len(interested)):
            if j != i and distance(*interested[i], *interested[j]) < distance_threshold:
                flags[j] = color
    final = []
    for i in range(len(flags)):
        if flags[i] != -1:
            color = flags[i]
            flags[i] = -1
            tmp = [interested[i]]
            for j in range(i+1, len(flags)):
                if flags[j] == color:
                    tmp.append(interested[j])
                    flags[j] = -1
            mean = np.mean(tmp, axis=0)
            final.append((int(mean[0]), int(mean[1])))
    return final

def find_4_points(img):
    x, y = np.nonzero(img==0)
    left, right = np.min(x), np.max(x)
    up, down = np.min(y), np.max(y)
    return left, right, up, down

def count_one(img):
    left, right, up, down = find_4_points(img)
    if np.abs((down-up)-(right-left)) > single_circle_thresh1:
        return None
    else:
        center = int((right+left)/2), int((down+up)/2)
        radius = int(min((right-left)/2, (down-up)/2))
        test = np.full_like(img, 255)
        imgshow(img)
        cv2.circle(test, center, radius, 0, cv2.FILLED)
        imgshow(test)
        ol = np.logical_and(img==0, test==0)
        area_ol = np.sum(ol)
        test[ol] = 255
        area_sub = np.sum(test==0)
        area_img = np.sum(img==0)
        if area_ol/area_img > single_circle_thresh2 and area_sub/area_img < single_circle_thresh3:
            return center, radius
        else:
            return None


if __name__ == '__main__':
    # img = cv2.imread('73.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find_all_circles(img)
    circle1 = ((50, 50), 40)
    circle2 = ((80, 80), 40)





