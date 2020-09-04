#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:59:41 2019

2019/5/14 trying to display image and the selected ROIs

@author: chenghung
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
#import PIL
from skimage import io
#import h5py
#from scipy.signal import find_peaks
#from PIL import Image
#import math

# %matplotlib qt
# %matplotlib inline
# plt.close('all')

plt.close('all')
##############################################################
#'''
#Read IC3 of a singal scan from txt
#'''
#
#inputdir = '/Users/chenghung/desktop/20190422/ic_reading/'
#
#ic3_single = []    #creat an empty list to store ic3 reading for th image at each scan
#th = 9             #the order of image in raster scan (cell4 is 9; cell3 is 0)
#scan_list = np.arange(8862, 8920)
#
#for scan_ID in scan_list:    
#    ic3_file = inputdir + 'ic3_scan_' + str(scan_ID) + '.txt'
#    
#    '''
#    Read the file line by line, and put them into the different array
#    '''
#    scan_time = np.zeros(19)
#    ic3_reading = np.zeros(19)
#    i = 0
#    with open(ic3_file) as ic3_data:
#        for data in ic3_data:        #each 'data' is a line
#            data = data.strip('\n')  #removing the newline character
#            data = data.split(' ')   #using space character to seperate the string into a list of 3 strings
#            data = np.array(data)    #converting list to numpy array of strings
#            data = data.astype('float')  #converting numpy array strings into numpy array floats
#            
#            scan_time[i] = data[0]         #save the 1st column to scan_time
#            ic3_reading[i] = data[1]       #save the 2nd column to ic3_reading
#            i = i+1
#    
#    #print(scan_time)
#    #print(ic3_reading)
#    
#    ic3_single.append(ic3_reading[th])  #append the th in ic3_reading as ic3_single
#    ic3_data.close()
#    
##    #plot ic3 intensity versus time
##    plt.figure()
##    plt.plot(scan_time, ic3_reading, 'b-',)
##    plt.plot(scan_time, ic3_reading, 'ro',)
##    plt.xlabel('Scan time', fontsize=14)
##    plt.ylabel('ic3_reading', fontsize=14)
##    plt.title('Scan_' + str(scan_ID))
#
#'''
#Plot IC3 readings of all in situ scans
#'''
#ic3_single = np.array(ic3_single)       #converting list to numpy array of strings
#plt.figure()
#l1, = plt.plot(ic3_single, 'b-')
#l2, = plt.plot(ic3_single, 'ro')
#plt.xlabel('Scan sequence', fontsize=14)
#plt.ylabel('IC3 Reading', fontsize=14)
#plt.legend(handles=[l2], labels=['IC3'], fontsize=14,  loc='best')
#
##print (ic3_single)
##print(ic3_single.shape)


##############################################################

'''
Read in situ images from tiff stack
'''
insitu_file = 'cell3_img_00_aligned.tif'
insitu = io.imread(insitu_file)
#print(insitu.shape)


'''
Select the frames corresponding to voltage profile
'''
frame_number = np.array([0, 10, 18, 26, 28, 35, 39])
insitu_voltage = np.zeros ((len(frame_number), insitu.shape[1], insitu.shape[2]))
for i,j in zip(range(len(frame_number)), frame_number):
    insitu_voltage[i,:,:] = insitu[j,:,:]

insitu = insitu_voltage

'''
Display a single fram of image before -log if desired:
'''
#showframe = 0 #select which frame to show
#plt.figure('insitu before raw data, frame number = '+ str(showframe))
#plt.imshow(insitu[showframe], interpolation= None, cmap = plt.cm.autumn, vmin=0, vmax=0.04) #interpolation can be 'nearest' also. 
#cbar = plt.colorbar()
#cbar.set_label('X-ray Absorption', fontsize=14)

'''
Normalize insitu images by ic3_single and save it as tiff stack.
'''
#for k in range(0, insitu.shape[0]):
#    insitu[k,:,:] = insitu[k,:,:]/ic3_single[k]

#io.imsave('img_04_aligned_nor.tiff' , insitu)

'''
Plot intensity varaition along time
'''

log_insitu = -np.log(insitu)

'''
Set all non-finite values (infinity and NaN) to 0!!
'''
#log_insitu_0 = log_insitu[:]
#finite = np.isfinite(log_insitu_0)
#non_finite = np.invert(finite)
#log_insitu_0[non_finite] = 0    
#io.imsave('cell_03_aligned_log.tiff' , log_insitu_0)
#io.imsave('cell_03_aligned_02.tiff' , insitu)

'''
Display a single fram of image after -log if desired:
'''
showframe = 0 #select which frame to show
plt.figure('insitu negative natural log, frame number = '+ str(showframe))
#plt.title('Full-filed view of the observation window', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(np.arange(0, insitu.shape[1]+1, 500), fontsize=16)
plt.xticks(ticks=[], labels=[])
plt.yticks(ticks=[], labels=[])
plt.imshow(log_insitu[showframe], interpolation= None, cmap = "winter", vmin=2.4, vmax=5) #interpolation can be 'nearest' also.
plt.xlim(0, insitu.shape[2])
plt.ylim(insitu.shape[1], 0)
cbar = plt.colorbar()
cbar.set_label('X-ray Attenuation', fontsize=20, fontweight='bold')
cbar.ax.tick_params(labelsize=12) 

#Set frequency of ticks in colorbar
cbar.locator = ticker.MaxNLocator(nbins=6)
cbar.update_ticks()

#plt.show()

##Set all non-finite values (infinity and NaN) to 0!!
#log_insitu_0 = log_insitu[:]
#finite = np.isfinite(log_insitu_0)
#non_finite = np.invert(finite)
#log_insitu_0[non_finite] = 0    
#io.imsave('img_04_aligned_log.tiff' , log_insitu_0)
#io.imsave('img_04_aligned_02.tiff' , insitu)

#y = 1408  #Assign a y-pixel value
#
##Put x-pixel values with same y-pixel vs Scan time into a 2D array
#pixel_y = np.zeros((insitu.shape[0], insitu.shape[2]))
#for k in range(0, insitu.shape[0]):
#    pixel_y[k,:] = log_insitu[k,y,:]


'''
Normalize insitu images by each frame's average and save it as tiff stack.
'''
log_insitu_fov = np.zeros((insitu.shape[0], insitu.shape[1], insitu.shape[2]))
for k in range(insitu.shape[0]):
    a = np.isfinite(log_insitu[k,:,:])            #Pick up the finite values
    b = log_insitu[k, a]
    c = b < 999                                   #Set treshold value
    log_insitu_mask = b[c]
    log_insitu_fov[k,:,:] = log_insitu[k,:,:]/np.mean(log_insitu_mask)
    

##Set all non-finite values (infinity and NaN) to 0!!
#log_insitu_fov_0 = log_insitu_fov[:]
#finite = np.isfinite(log_insitu_fov_0)
#non_finite = np.invert(finite)
#log_insitu_fov_0[non_finite] = 0    
#io.imsave('img_04_aligned_log_avenor.tiff' , log_insitu_fov_0)


##############################################################

##line/ROI profile selection
#x_0 =  0 #1436
#x_1 =  1280 #2000 
#y_0 = 450
#y_1 = y_0 + 30 #1320
#plt.plot( [x_0, x_1], [y_0, y_0], 'r-')
#plt.plot( [x_0, x_1], [y_1, y_1], 'r-')
#plt.plot( [x_0, x_0], [y_0, y_1], 'r-')
#plt.plot( [x_1, x_1], [y_0, y_1], 'r-')

#ROI_i
#x_0 =  630
#x_1 =  880
#y_0 = 450
#y_1 = y_0 + 30
#plt.plot( [x_0, x_1], [y_0, y_0], 'r-')
#plt.plot( [x_0, x_1], [y_1, y_1], 'r-')
#plt.plot( [x_0, x_0], [y_0, y_1], 'r-')
#plt.plot( [x_1, x_1], [y_0, y_1], 'r-')
#
##ROI_ii
#x_0 =  290
#x_1 =  540
#y_0 = 620
#y_1 = y_0 + 30
#plt.plot( [x_0, x_1], [y_0, y_0], 'r-')
#plt.plot( [x_0, x_1], [y_1, y_1], 'r-')
#plt.plot( [x_0, x_0], [y_0, y_1], 'r-')
#plt.plot( [x_1, x_1], [y_0, y_1], 'r-')
#    
##ROI_iii
#x_0 =  510
#x_1 =  760
#y_0 = 415
#y_1 = y_0 + 30
#plt.plot( [x_0, x_1], [y_0, y_0], 'r-')
#plt.plot( [x_0, x_1], [y_1, y_1], 'r-')
#plt.plot( [x_0, x_0], [y_0, y_1], 'r-')
#plt.plot( [x_1, x_1], [y_0, y_1], 'r-')
#    
##ROI_iv
#x_0 =  550
#x_1 =  800
#y_0 = 500
#y_1 = y_0 + 30
#plt.plot( [x_0, x_1], [y_0, y_0], 'r-')
#plt.plot( [x_0, x_1], [y_1, y_1], 'r-')
#plt.plot( [x_0, x_0], [y_0, y_1], 'r-')
#plt.plot( [x_1, x_1], [y_0, y_1], 'r-')
#
#plt.show()
#plt.savefig('ROI_particle_evolution.png', dpi = 300)
plt.savefig('ROI_particle_FOV.png', dpi = 300)

x_roi = np.arange(x_0,x_1,1)
y_roi = np.arange(y_0,y_1,1)


pixel_y = np.zeros((insitu.shape[0], len(x_roi)))
for k in range(insitu.shape[0]):
    for i in range(len(x_roi)):
        #a = log_insitu[k, y_0:y_1, i+x_0]
        a = log_insitu_fov[k, y_0:y_1, i+x_0]
        b = np.isfinite(a)
        c = a[b]
        pixel_y[k, i] = np.mean(c)

pixel_x = np.arange(x_1-x_1, x_1-x_0)
pixel_size = 0.0325
width_x = pixel_x*pixel_size

#Creat a contour for making colorbar
plt.figure()
color_idx = np.linspace(0, 1, insitu.shape[0])
Z = [[0,0],[0,0]]
levels = color_idx*insitu.shape[0]
color_bar = plt.contourf(Z, np.rint(levels), cmap='winter') #plt.cm.autumn)
plt.close()


#Plot intensity variations along x-pixel under same y-pixel with 1-by-1 scan sequence
plt.figure( 'x_0 = ' + str(x_0) + ', x_1 = ' + str(x_1) + ', y_0 = ' + str(y_0) + ', y_1 = ' + str(y_1))
#plt.title('Full-filed view of the observation window', fontsize=14)
#plt.xlim(100, 500)
#plt.ylim(-11.5, -9.5)
x_label = 'X_Pixels: ' + str(x_0) + ' ~ ' + str(x_1)
for k , l in zip (range(0, insitu.shape[0]), color_idx):    
    plt.plot(width_x, pixel_y [k,:], label = k, color=plt.cm.winter(l))
    #plt.xlabel(x_label, fontsize=20, fontweight='bold')
    #plt.ylabel('Intensity', fontsize=20, fontweight='bold')
    #plt.xticks(fontsize=16)
    #plt.xticks(ticks=[], labels=[])
    #plt.yticks(fontsize=12)

ax = plt.gca()
ax.tick_params(direction='in', colors='red', length=4, width=2)
y_max = np.around(np.amax(pixel_y)+0.04, decimals = 2)
y_min = np.around(np.amin(pixel_y)-0.04, decimals = 2)
plt.yticks(np.arange(y_min, y_max, 0.1), fontsize=12, color='red', fontweight='bold')
x_max = np.around(np.ceil(np.amax(width_x)), decimals = 2)
x_min = x_1 - x_1
plt.xticks(np.arange(x_min, x_max, 2.5), fontsize=12, color='red', fontweight='bold')

spine_list = ['bottom', 'top', 'right', 'left']
spine_color = 'red'
spine_width = 2
for i in spine_list:
    ax.spines[i].set_color(spine_color)
    ax.spines[i].set_linewidth(spine_width)

cbar = plt.colorbar(color_bar)
cbar.set_label('Scan Sequence', fontsize=20, fontweight='bold')
cbar.ax.tick_params(labelsize=12)
imag_name = 'ROI_iv_02' + '(x_0 = ' + str(x_0) + ', x_1 = ' + str(x_1) + ', y_0 = ' + str(y_0) + ', y_1 = ' + str(y_1) + ').png'
#plt.savefig(imag_name, dpi = 300)

'''
Plot intensity variations by colormap  
'''
#plt.figure()
#plt.title('Intensity Variations', fontsize=14)
#plt.xlabel(x_label, fontsize=14)
#plt.ylabel('Scan Sequence', fontsize=14)
#y_inten = plt.pcolor(pixel_y, cmap = plt.cm.cool)
#plt.colorbar(y_inten)


##############################################################

##Normalized images by individual average
#thr_range = np.array([100])
##thr_range = np.arange(3.2, 4.1, 0.1)
#plt.figure()
#for thr in thr_range:
#    log_insitu_fov = np.zeros((insitu.shape[0], insitu.shape[1], insitu.shape[2]))
#    for k in range(insitu.shape[0]):
#        a = np.isfinite(log_insitu[k,:,:])            #Pick up the finite values
#        b = log_insitu[k, a]
#        c = b < thr                                   #Set treshold value
#        log_insitu_mask = b[c]
#        log_insitu_fov[k,:,:] = log_insitu[k,:,:]/np.mean(log_insitu_mask)
#        
#    #io.imsave('img_04_aligned_log_avenor.tiff' , log_insitu)
#    
#    '''
#    plt.figure()
#    #plt.title('Intensity Variations', fontsize=14)
#    #plt.xlabel(x_label, fontsize=14)
#    #plt.ylabel('Scan Sequence', fontsize=14)
#    c_map = plt.imshow(log_insitu[0,:,:], cmap = plt.cm.cool)
#    plt.colorbar(c_map)
#    '''
#    
#    #Calculate average intensity vs scan sequence
#    ave_int = np.zeros(insitu.shape[0])
#    for k in range(insitu.shape[0]):
#        a = np.isfinite(log_insitu_fov[k,:,:])
#        log_insitu_fin = log_insitu_fov[k, a]
#        ave_int[k] = np.mean(log_insitu_fin)
#    
#    
#    #plt.title('Threshold < '+str(thr), fontsize=14)
#    
#    thr_round = np.round(thr, decimals=2)
#    plt.xlabel('Scan Sequence', fontsize=14)
#    plt.ylabel('Normalized Intensity', fontsize=14)
#    plt.ylim(0.8, 1.2)
#    plt.plot(ave_int, label = thr_round)
#    #l2, = plt.plot(ave_int, 'ro')
#    #plt.legend(handles=[l1], fontsize=14,  loc='best')
#plt.legend()

##############################################################

#print(mask.shape)
#ref0 = []
#ref0 = ratio[0,:,:]
#ref1 = ratio[1,:,:]
#print(ref0.shape)


#i = 80
#while i < 119:
#    j= 37
#    while j < 67:
#        if sum_ratio[i][j] == 1:
#            ref0_exclude.append(ref0[i][j])
#            ref1_exclude.append(ref1[i][j])
#        j=j+1
#    i=i+1

#i=0  
#while i < sum_ratio.shape[0]:
#    j=0
#    while j < sum_ratio.shape[1]:
#        if sum_ratio[i][j] == 1:
#            ref0_exclude.append(ref0[i][j])
#            ref1_exclude.append(ref1[i][j])
#        j=j+1
#    i=i+1

#print (len(sum_1))

#ref0_average = np.average(ref0_exclude)
#ref0_std = np.std(ref0_exclude)
#ref1_average = np.average(ref1_exclude)
#ref1_std = np.std(ref1_exclude)
#
#print (ref0_average, ref0_std)
#print (ref1_average, ref1_std)




#h5 = h5py.File(h5_file, 'r')
#print(list(h5.keys()))
#eng_scan=np.array(h5['X_eng'])
#print(eng_scan)


#'''
#Plot spectrum from single pixel
#'''
#x_pixel = 540
#y_pixel = 680
#plt.figure()
#plt.plot(eng_scan, -np.log(im[:,x_pixel,y_pixel]), 'r')
#
#

#
#log_roi = -np.log(sum_roi)
#plt.figure()
#plt.plot(eng_scan, log_roi, 'b')
#
#
#






