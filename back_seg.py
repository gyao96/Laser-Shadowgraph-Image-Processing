# coding:utf-8
import cv2
import numpy as np
from MyDIPUtils.hole import floodfill
#from MyDIPUtils.counting import count_circles
#from MyCvUtils import MyCvUtils

drawing = False # 鼠标左键按下时，该值为True，标记正在绘画
mode = True # True 画矩形，False 画圆
iy, ix = -1, -1 # 鼠标左键按下时的坐标
winflag = False
cord = ()

def backgroundseg(camera, window):
    # Read, Fraction and convert to gray scale
    ret, frame = camera.read()
    [height, width] = frame.shape[:2]
    frame = frame[window[0][0]:window[1][0], window[0][1]:window[1][1]]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    global i

    # Pre-processing: denoise
    gray_neg = cv2.bitwise_not(gray)
    gray_neg_med = cv2.medianBlur(gray_neg,11)
    gray_neg_med = cv2.medianBlur(gray_neg_med,11)
    gray_neg_med_blur = cv2.GaussianBlur(gray_neg_med,(5,5),0)
    # Calculate Foregound Mask; freground encoded by white; shadow encoded by gray
    fgmask = bs.apply(gray_neg_med_blur)
    
    # Pre-processing: hole filling with dynamic foregound segmentation
    hist = cv2.calcHist([fgmask],[0],None,[256],[30,256])
    num_pixel_shadow = sum(hist[100:120])
    num_pixel_object = sum(hist[240:255])
    shadow_object = num_pixel_shadow/(num_pixel_object+1)
    # filter out shadow if shadow out weigh object
    if shadow_object >= 0.15: # area grow
        fgmask_temp = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)[1]
        fgmask_temp = floodfill(fgmask_temp, FULL = True)
        fgmask_temp = cv2.dilate(fgmask_temp, cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)), iterations = 6)
        fgmask_temp = floodfill(fgmask_temp, FULL = True)
        fgmask_hole_filled = cv2.erode(fgmask_temp, cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)), iterations = 6)
    else:
        fgmask_temp = cv2.threshold(fgmask, 70, 255, cv2.THRESH_BINARY)[1]
        fgmask_hole_filled = floodfill(fgmask_temp, FULL = True)
        #fgmask_hole_filled = cv2.morphologyEx(fgmask_temp, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (9,9)))
    
    # Foregound Morphological Operation
    closed = cv2.morphologyEx(fgmask_hole_filled, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    dilated = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))) 
    mask = cv2.dilate(dilated, cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)), iterations = 3)
    
    # calculate backgound
    diff = cv2.add(cv2.cvtColor(np.bitwise_not(mask), cv2.COLOR_GRAY2BGR) , frame & cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    # mark contours of the foreground by rectangles
    image, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
    # result filtering
        if cv2.contourArea(c) > 50:
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)

    im_hole = mask.copy()
    image, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    listof_contours_roi = []
    listof_contours_full = []
    mask_refined = np.zeros(mask.shape)
    for index, c in enumerate(contours):
        if cv2.contourArea(c) > 50:
            (x,y,w,h) = cv2.boundingRect(c)
            roi = im_hole[y:y+h, x:x+w]
            im_hole_single = np.zeros(im_hole.shape)
            im_hole_single[y:y+h,x:x+w] = roi
            

            listof_contours_full.append(im_hole_single)
            
            #while len(np.nonzero(im_hole_single)[0]) >= 50:
            #    im_hole_single = cv2.erode(im_hole_single, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)), iterations = 1)
            #(seed_x,seed_y) = (np.nonzero(im_hole_single)[0][0], np.nonzero(im_hole_single)[1][0])                   

            gray_c = np.bitwise_not(gray[y:y+h,x:x+w]) & roi
            # Morph Gray obtain gray_c
            Mgray_c = cv2.threshold(gray_c, 200, 255, cv2.THRESH_BINARY)[1]
            Mgray_c = cv2.morphologyEx(Mgray_c, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
            Mgray_c = cv2.morphologyEx(Mgray_c, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))
            if len(np.nonzero(Mgray_c)[0]) == 0:
                pass
            else:
                image, contours_refined, hier_refined = cv2.findContours(Mgray_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for index_r, c in enumerate(contours_refined):
                    (x_r,y_r,w_r,h_r) = cv2.boundingRect(c)
                    x_refined = x + x_r
                    y_refined = y + y_r
                    roi_refined = Mgray_c[y_r:y_r+h_r, x_r:x_r+w_r]
                    mask_refined[y_refined : y_refined + h_r, x_refined : x_refined + w_r] = roi_refined 
                    cv2.rectangle(frame, (x_refined,y_refined), (x_refined+w_r, y_refined+h_r), (255, 255, 255), 2)
                    listof_contours_roi.append((x_refined,w_r,y_refined,h_r,roi_refined))
                    if i > 4:
                        roi_h, roi_w = roi_refined.shape[:2]
                        object_save = np.zeros((roi_h+16, roi_w+16), np.uint8)
                        object_save[8:8+roi_h,8:8+roi_w] = roi_refined
                        #circle_esti = count_circles(img, draw )
                        #cv2.imwrite('gray_'+str(i)+'_'+str(index+index_r)+'.png',roi_refined)
    cv2.imshow("mog", fgmask)
    cv2.imshow("thresh", mask_refined)
    cv2.imshow("Background", diff)
    #cv2.imwrite(str(i)+'diff.png',diff)
    cv2.imshow("detection", frame)
  
    i = i+1

def draw_circle(event, y, x, flags, param):
    global ix, iy, drawing, winflag, cord

    if event == cv2.EVENT_LBUTTONDOWN:
        # 鼠标左键按下事件
        drawing = True
        iy, ix = y, x

    elif event == cv2.EVENT_MOUSEMOVE:
        # 鼠标移动事件
        if drawing == True:
            cv2.rectangle(frame, (iy, ix), (y, x), (0, 255, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        # 鼠标左键松开事件
        drawing = False
        cv2.rectangle(frame, (iy,ix), (y, x), (0, 255, 0), -1)
        cord = (ix,iy),(x,y)
        winflag = True

if __name__ == '__main__':
    dir = 'C:/Users/sirei/Pictures/Bubble count_Deblur_Denoise_Video and Picture/'
    bs = cv2.createBackgroundSubtractorKNN(detectShadows = True)
    camera = cv2.VideoCapture(dir + "cavitation0.3.avi")

    ret, frame = camera.read()
    [height, width] = frame.shape[:2]
    i = 1
    window = ((50,50),(650,800))

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle) # 设置鼠标事件的回调函数

    while(True):
        cv2.imshow('image', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif winflag == True:
            break
    cv2.destroyWindow('image')
    while(True):
        backgroundseg(camera, cord)
        if (cv2.waitKey(30) & 0xFF) == 27:
            break
        elif (cv2.waitKey(30) & 0xFF) == ord('q'):
            break
        elif (cv2.waitKey(30) & 0xFF) == ord('s'):
            cv2.imwrite('debug_mog.png',fgmask)
            cv2.imwrite('debug_detection.png',frame)
            break  

    camera.release()
    cv2.destroyAllWindows()