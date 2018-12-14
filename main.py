# _*_ coding:utf8
import tkinter
from win32com.client import Dispatch
from tkinter import filedialog,messagebox
from tkinter import *
import sys
from sys import exit
import cv2
import numpy as np
import scipy.io
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt  
from MyDIPUtils.hole import floodfill
from MyDIPUtils.counting import *

drawing = False # Set to true when left click is on hole, marking drawing in progress
mode = True # Rectanglar Selection when true
iy, ix = -1, -1 # cord when left click interrupted
step_size = [3, 6, 9]
winflag = False
cord = ()
i = 1

class laser(object):
    def init(self,root):
        self.root = root
        self.create_menu(self.root)
        b = tkinter.Button(root, text='Big-Overlap', width=15, height = 2, command=self.open_files)
        b.pack()
        bn = tkinter.Button(root, text='Big-No-Overlap', width=15, height = 2, command=self.big_no_overlap)
        bn.pack()
        sb = tkinter.Button(root, text='Show', width=15, height = 2, command=self.show_rois)
        sb.pack()
        smb = tkinter.Button(root, text='Small-No-Overlap', width=15, height = 2, command=self.count_small)
        smb.pack()
        self.root.mainloop()
    
    def create_menu(self, root):
        self.root = root
        menubar = tkinter.Menu(self.root)
        filemenu = tkinter.Menu(menubar,tearoff=0)
        filemenu.add_command(label='Open Video File',command=self.show_rois,accelerator='Ctrl+O')
        filemenu.add_command(label='Exit',command=self.close,accelerator='Ctrl+X')
        root.bind_all("<Control-o>", lambda event:self.show_rois())
        root.bind_all("<Control-x>", lambda event:self.close())
        abort = tkinter.Menu(menubar,tearoff=0)
        abort.add_command(label='About',command=self.tkabort)
        menubar.add_cascade(label='File',menu=filemenu,)
        menubar.add_cascade(label='Help',menu=abort)
        root.config(menu=menubar)

    def open_files(self):
        global frame
        filenames = filedialog.askopenfilenames(initialdir = './',title='Select Video',filetypes=[('AVI','*.avi'),('MP4','*.mp4'),("JPG","*.jpg"),("WAV","*.wav")])
        # listboxs = self.contents(self.root)
        for name in filenames:
            if name:
                camera = cv2.VideoCapture(name)
                ret, frame = camera.read()
                out_bubble_list = []
                cv2.namedWindow('image')
                cv2.setMouseCallback('image', draw_circle) # Mouse click recall function

                while(True):
                    cv2.imshow('image', frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:
                        break
                    elif winflag == True:
                        break
                cv2.destroyWindow('image')
                # Establish Learning Basis
                for it in range(1,10):
                    ret, _ = backgroundseg(camera, cord, count = False, show = False, pre = True)
                    if ret == False:
                        break
                while(True):
                    ret, bubble_area = backgroundseg(camera, cord, count = True, show = False, pre = False)
                    if ret == False:
                        break
                    out_bubble_list.extend(bubble_area)
                    if (cv2.waitKey(30) & 0xFF) == 27:
                        break
                    elif (cv2.waitKey(30) & 0xFF) == ord('q'):
                        break
                camera.release()
                cv2.destroyAllWindows()
                # Plot the Result
                np.savetxt(name+'.txt', out_bubble_list)
                ShowHisto(out_bubble_list,x_scale='R')
                clear()
        return filenames

    def big_no_overlap(self):
        global frame
        filenames = filedialog.askopenfilenames(initialdir = './',title='Select Video',filetypes=[('AVI','*.avi'),('MP4','*.mp4'),("JPG","*.jpg"),("WAV","*.wav")])
        # listboxs = self.contents(self.root)
        for name in filenames:
            if name:
                camera = cv2.VideoCapture(name)
                ret, frame = camera.read()
                out_bubble_list = []
                cv2.namedWindow('image')
                cv2.setMouseCallback('image', draw_circle) # Mouse click recall function

                while(True):
                    cv2.imshow('image', frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:
                        break
                    elif winflag == True:
                        break
                cv2.destroyWindow('image')
                # Establish Learning Basis
                for it in range(1,10):
                    ret, _ = backgroundseg(camera, cord, count = False, show = False, pre = True)
                    if ret == False:
                        break
                while(True):
                    ret, bubble_area = backgroundseg(camera, cord, count = False, show = False, pre = False)
                    if ret == False:
                        break
                    out_bubble_list.extend(bubble_area)
                    if (cv2.waitKey(30) & 0xFF) == 27:
                        break
                    elif (cv2.waitKey(30) & 0xFF) == ord('q'):
                        break
                camera.release()
                cv2.destroyAllWindows()
                # Plot the Result
                np.savetxt(name+'.txt', out_bubble_list)
                ShowHisto(out_bubble_list)
                clear()
        return filenames

    def show_rois(self):
        global frame, i
        filenames = filedialog.askopenfilenames(initialdir = './',title='Select Video',filetypes=[('AVI','*.avi'),('MP4','*.mp4'),("JPG","*.jpg"),("WAV","*.wav")])
        # listboxs = self.contents(self.root)
        for name in filenames:
            if name:
                camera = cv2.VideoCapture(name)
                ret, frame = camera.read()

                cv2.namedWindow('image')
                cv2.setMouseCallback('image', draw_circle) # Mouse click recall function

                while(True):
                    cv2.imshow('image', frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:
                        break
                    elif winflag == True:
                        break
                cv2.destroyWindow('image')
                # Establish Learning Basis
                for it in range(1,10):
                    ret, _ = backgroundseg(camera, cord, count = False, show = False, pre = True)
                    if ret == False:
                        break
                # Reset and Start Showing
                camera = cv2.VideoCapture(name)
                while(True):
                    ret, _ = backgroundseg(camera, cord, count = False, show = True, pre = False)
                    if ret == False:
                        break
                    if (cv2.waitKey(30) & 0xFF) == 27:
                        break
                    elif (cv2.waitKey(30) & 0xFF) == ord('q'):
                        break
                camera.release()
                cv2.destroyAllWindows()
                clear()
        return filenames

    def count_small(self):
        global frame
        filenames = filedialog.askopenfilenames(initialdir = './',title='Select Video',filetypes=[('AVI','*.avi'),('MP4','*.mp4'),("JPG","*.jpg"),("WAV","*.wav")])
        # listboxs = self.contents(self.root)
        for name in filenames:
            if name:
                camera = cv2.VideoCapture(name)
                for passindec in range(10):
                    ret, frame = camera.read()
                out_bubble_list = []
                cv2.namedWindow('image')
                cv2.setMouseCallback('image', draw_circle) # Mouse click recall function

                while(True):
                    cv2.imshow('image', frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:
                        break
                    elif winflag == True:
                        break
                cv2.destroyWindow('image')
                while True:
                    ret, bubble_area = nooverlap(camera, cord)
                    out_bubble_list.extend(bubble_area)
                    if ret == False:
                        break
                    if (cv2.waitKey(30) & 0xFF) == 27:
                        return
                    elif (cv2.waitKey(30) & 0xFF) == ord('q'):
                        return 
                camera.release()
                cv2.destroyAllWindows()
                # Plot the Result
                np.savetxt(name+'.txt', out_bubble_list)
                ShowHisto(out_bubble_list)
                clear()
        return filenames
    #abort
    def tkabort(self):
        messagebox.showinfo('Help',
        'Big-Overlap: For big size bubbles with frequent overlap occurrence \n Big-No-Overlap: For big size bubbles with no ovelapping \n Show: For big bubbles, check the detection performance first \n Small-No-Overlap: For small bubbles, there expect to be no overlap')
    #Shut down Program
    def close(self):
        exit()

def draw_circle(event, y, x, flags, param):
    global ix, iy, drawing, winflag, cord

    if event == cv2.EVENT_LBUTTONDOWN:
        # Mouse Left Click
        drawing = True
        iy, ix = y, x

    elif event == cv2.EVENT_MOUSEMOVE:
        # Mouse Move
        if drawing == True:
            cv2.rectangle(frame, (iy, ix), (y, x), (0, 255, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        # Mouse Left Release
        drawing = False
        cv2.rectangle(frame, (iy,ix), (y, x), (0, 255, 0), -1)
        cord = (ix,iy),(x,y)
        print('Window:'+str(cord)+' lenY:'+str(abs(iy-y))+' lenX:'+str(abs(ix-x)))
        winflag = True

def backgroundseg(camera, window, count = False, show = True, pre = False):
    global bs, i
    # Read, Fraction and convert to gray scale
    ret, frame = camera.read()
    if ret == False:
        return False,[]
    frame = frame[window[0][0]:window[1][0], window[0][1]:window[1][1]]
    [height, width] = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pre-processing: denoise
    gray_neg = cv2.bitwise_not(gray)
    gray_neg_med = cv2.medianBlur(gray_neg,5)
    #gray_neg_med = cv2.medianBlur(gray_neg_med,11)
    gray_neg_med_blur = cv2.GaussianBlur(gray_neg_med,(5,5),0)
    # Calculate Foregound Mask; freground encoded by white; shadow encoded by gray
    fgmask = bs.apply(gray_neg_med)
    
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
    im_hole = mask.copy()
    image, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_refined = np.zeros(mask.shape)
    bubble_area = []  # This is the return list: Individual Bubbles Marked by its area
    num_circles = 0
    for index, c in enumerate(contours):
        # Result Filtering
        if cv2.contourArea(c) > 100:
            (x,y,w,h) = cv2.boundingRect(c)
            roi = im_hole[y:y+h, x:x+w]
            im_hole_single = np.zeros(im_hole.shape)
            im_hole_single[y:y+h,x:x+w] = roi             

            gray_c = np.bitwise_not(gray[y:y+h,x:x+w]) & roi
            # Morph Gray obtain gray_c
            Mgray_c = cv2.threshold(gray_c, 200, 255, cv2.THRESH_BINARY)[1]
            Mgray_c = cv2.morphologyEx(Mgray_c, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
            Mgray_c = cv2.morphologyEx(Mgray_c, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
            if len(np.nonzero(Mgray_c)[0]) == 0:
                pass
            else:
                image, contours_refined, hier_refined = cv2.findContours(Mgray_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for index_r, cc in enumerate(contours_refined):
                    if cv2.contourArea(cc) > 30:
                        (x_r,y_r,w_r,h_r) = cv2.boundingRect(cc)
                        x_refined = x + x_r
                        y_refined = y + y_r
                        roi_refined = Mgray_c[y_r:y_r+h_r, x_r:x_r+w_r]
                        mask_refined[y_refined : y_refined + h_r, x_refined : x_refined + w_r] = roi_refined 
                        cv2.rectangle(frame, (x_refined,y_refined), (x_refined+w_r, y_refined+h_r), (255, 255, 255), 2)
                        # Refined ROI Save
                        roi_h, roi_w = roi_refined.shape[:2]
                        object_save = np.zeros((roi_h+16, roi_w+16), np.uint8)
                        object_save[8:8+roi_h,8:8+roi_w] = roi_refined
                        if 100.0*(object_save.shape[0]*object_save.shape[1])/(height*width) > 10.0:
                            continue
                        # cv2.imshow("img", object_save)
                        if count == True:
                            if cv2.contourArea(cc) > 100:
                                if len(np.nonzero(object_save)[0]) == 0:
                                    continue
                                try:
                                    ret = count_circles(object_save, draw = False, step_size = step_size)
                                    scalar = max(object_save.shape[0],object_save.shape[1])/256
                                    num_circles = num_circles + len(ret)
                                    for cir in ret:
                                        if type(cir) == tuple:
                                            r = cir[1]*scalar
                                            bubble_area.extend([int(3.14*r*r)])
                                except:
                                    cv2.imwrite('debug_'+str(i)+'_'+str(index+index_r)+'.png',object_save)
                                    num_circles = num_circles + 1
                                    bubble_area.extend([cv2.contourArea(cc)])
                                    print('error_'+str(i))
                                    continue
                            else:
                                num_circles = num_circles + 1
                                bubble_area.extend([cv2.contourArea(cc)])
                        else:
                            num_circles = num_circles + 1
                            bubble_area.extend([cv2.contourArea(cc)])
    if pre == False:
        print('Frame '+str(i)+' has '+str(num_circles)+' bubbles')                                
        if show == True:
            cv2.imshow("mog", fgmask)
            cv2.imshow("thresh", mask_refined)
            cv2.imshow("Background", diff)
            cv2.imshow("detection", frame)
        i = i+1
    return True,bubble_area
 


def nooverlap(camera, window):
    global bss, i
    # Proceed Iteration
    i = i+1
    # Read, Fraction and convert to gray scale
    ret, frame = camera.read()
    if ret == False:
        return False, []
    frame = frame[window[0][0]:window[1][0], window[0][1]:window[1][1]]
    [height, width] = frame.shape[:2]
    # Covert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,None,fx=8, fy=8, interpolation = cv2.INTER_LINEAR)
    # Grayscale Histograph Equalization
    gray = cv2.equalizeHist(gray)
    fgmask = bss.apply(gray)
    # cv2.imwrite(str(i)+'frame.png',gray)
    # cv2.imwrite(str(i)+'gray.png',gray)
    # cv2.imwrite(str(i)+'fgmask.png',fgmask)
    # Discard Invalid Detection
    hist = cv2.calcHist([fgmask],[0],None,[256],[30,256])
    num_pixel_shadow = sum(hist[100:130])
    num_pixel_object = sum(hist[240:255])
    shadow_frame = (num_pixel_shadow+num_pixel_object)/(fgmask.shape[0]*fgmask.shape[1])
    if shadow_frame > 0.5:
        print("Frame "+str(i-1)+" Too Much Noise")
        return True, []
    # Discard Scattered Mask
    scat = 0
    for rol in range(0,fgmask.shape[0]-12,11):
        for col in range(0,fgmask.shape[1]-12,11):
            temp = fgmask[rol:rol+5,col:col+5]
            scat = scat + np.var(temp)
    scat = scat/(fgmask.shape[0]*fgmask.shape[1]) #11, 0.093
    if scat > 0.8:
        print("Frame "+str(i-1)+" Scattered "+str(scat))
        return True, []
    # Discard if too many contours
    th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1];
    _, contour, hier = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contour)>800:
        print("Frame "+str(i-1)+" Too many small contours with "+str(len(contour)))
        return True, []
    # Distroy Small Noise
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    dilated = cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 2)
    # Discard if all black
    if len(np.nonzero(dilated)[0]) == 0:
        print("Frame "+str(i-1)+" no bubble")
        return True, []
    # Obtain ROUGH Mask
    mask = dilated.copy()
    _, contour, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Refine Mask Individually
    bubble_area = []
    for index,c in enumerate(contour):
        flag = 0
        (x,y,w,h) = cv2.boundingRect(c)
        bubble = gray[y:y+h,x:x+w]
        bubble_inv = np.bitwise_not(bubble)
        bubble_th = cv2.threshold(bubble_inv,200,255,cv2.THRESH_BINARY)[1]
        smooth = cv2.morphologyEx(bubble_th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        smooth_inv = np.bitwise_not(smooth)
        # evaluate the "squareness" of the shape
        [sa, sb] = bubble.shape[:2]
        if sa > sb:
            sindex = sa/sb
        else:
            sindex = sb/sa
        effectiveindex = len(np.nonzero(smooth_inv)[0])/(len(np.nonzero(smooth)[0])+1)

        if len(np.nonzero(smooth)[0]) > 0 and len(np.nonzero(smooth_inv)[0]) > 0 and smooth.size < 600 and sindex < 1.5 and effectiveindex > 1.8:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
            cv2.imwrite(str(i)+str(index)+'bubble.png',bubble)
            bubble_area.extend([1.5/8*(sa+sb)/2])
            flag = 1
    ################  SAVE FOR DEBUG   #############
    cv2.imwrite(str(i)+'mask'+str(flag)+'.png',mask)
    cv2.imwrite(str(i)+'gray'+str(flag)+'.png',gray)
    ###################################################
    cv2.imshow('frame',frame)
    return True, bubble_area

def ShowHisto(x,x_scale = 'S'):
    import matplotlib.mlab as mlab  
    import matplotlib.pyplot as plt  
    # Remove Extreme Values
    x=np.array(x)
    x_mean = np.mean(x)
    dis_th = 3*np.mean(abs(x-x_mean))
    x = x[(x-x_mean)<dis_th]
    # A total of 50 bars
    num_bins = 50  
    # the histogram of the data  
    if x_scale != 'S':
        x = np.sqrt(x)/3.14
    n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.8)  
    if x_scale == 'S':
        plt.xlabel('Bubble Size')
    else:
        plt.xlabel('Bubble Radius')  
    plt.ylabel('Number of Bubbles') 
    plt.yticks() 
    plt.title(r'Total Number of Bubbles: '+str(len(x)))
    # Tweak spacing to prevent clipping of ylabel  
    plt.subplots_adjust(left=0.15)  
    plt.show() 

def clear():
    global i, cord, winflag, drawing, mode, iy, ix
    drawing = False # Set to true when left click is on hold, marking drawing in progress
    mode = True # Rectanglar Selection when true
    iy, ix = -1, -1 # cord when left click interrupted
    winflag = False
    cord = ()
    i = 1

if __name__=='__main__':
    bs = cv2.createBackgroundSubtractorKNN(history=10, dist2Threshold=400,detectShadows = True)
    bss = cv2.createBackgroundSubtractorKNN(history=30, dist2Threshold=8000, detectShadows = True)
    root =tkinter.Tk()
    root.geometry('200x200+100+100')
    root.resizable(width=True, height=True)
    root.title('Laser Shadow')
    root.iconbitmap('error')
    p = laser()
    p.init(root)
