import cv2
import numpy as np
import matplotlib.pyplot as plt
dir = 'C:/Users/sirei/Pictures/Bubble count_Deblur_Denoise_Video and Picture/'
'''
. @brief Creates KNN Background Subtractor
.
. @param history Length of the history.
. @param dist2Threshold Threshold on the squared distance between the pixel and the sample to decide
. whether a pixel is close to that sample. This parameter does not affect the background update.
. @param detectShadows If true, the algorithm will detect shadows and mark them. It decreases the
. speed a bit, so if you do not need this feature, set the parameter to false.
'''
bs = cv2.createBackgroundSubtractorKNN(history=20, dist2Threshold=3000, detectShadows = True)
camera = cv2.VideoCapture(dir + "cavitation0.3.avi")
ret, frame = camera.read()
i = 0
window = ((369, 365), (673, 737))
out_bubble_list = []
frame = frame[window[0][0]:window[1][0], window[0][1]:window[1][1]]
# cv2.imwrite('ori_'+str(i)+'.png',frame)
while(1):
    i = i+1
    ret, frame = camera.read()
    if ret == False:
        break
    frame = frame[window[0][0]:window[1][0], window[0][1]:window[1][1]]
    # cv2.imwrite('ori_'+str(i)+'.png',frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Grayscale Histograph Equalization
    gray = cv2.equalizeHist(gray)
    fgmask = bs.apply(gray)
    # Discard Invalid Detection
    hist = cv2.calcHist([fgmask],[0],None,[256],[30,256])
    num_pixel_shadow = sum(hist[100:130])
    num_pixel_object = sum(hist[240:255])
    shadow_frame = (num_pixel_shadow+num_pixel_object)/(fgmask.shape[0]*fgmask.shape[1])
    if shadow_frame > 0.5:
        print("Frame "+str(i)+" Too Much Noise")
        continue
    # Discard Scattered Mask
    scat = 0
    for rol in range(0,fgmask.shape[0]-12,11):
        for col in range(0,fgmask.shape[1]-12,11):
            temp = fgmask[rol:rol+5,col:col+5]
            scat = scat + np.var(temp)
    scat = scat/(fgmask.shape[0]*fgmask.shape[1]) #11, 0.093
    if scat > 1:
        print("Frame "+str(i)+" Scattered")
        continue
    # Distroy Small Noise
    th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1];
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    dilated = cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 2)
    # Discard if all black
    if len(np.nonzero(dilated)[0]) == 0:
        print("All Black")
        continue
    # Obtain ROUGH Mask
    mask = dilated.copy()
    _, contour, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Refine Mask Individually
    for index,c in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        bubble = gray[y:y+h,x:x+w]
        bubble_inv = np.bitwise_not(bubble)
        bubble_th = cv2.threshold(bubble_inv,180,255,cv2.THRESH_BINARY)[1]
        smooth = cv2.morphologyEx(bubble_th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        bubble_area = len(np.nonzero(smooth)[0])
        if bubble_area > 0:
            out_bubble_list.append(bubble_area)
        # cv2.imwrite('mask_'+str(i)+'_'+str(index)+'t.png',bubble)
    cv2.imwrite('detect_'+str(i)+'.png',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

t = range(1,len(out_bubble_list)+1)
plt.scatter(t, out_bubble_list, s=75, c='r', alpha=.5)
plt.ylim(max(np.min(out_bubble_list)-10,0),np.max(out_bubble_list)+10)
plt.xlim(0,max(t)+1)
plt.title("Total Number of Bubbles: "+str(len(out_bubble_list)))
plt.xlabel('# of Bubble')
plt.ylabel('Bubble Area')
plt.show()

camera.release()
cv2.destroyAllWindows()