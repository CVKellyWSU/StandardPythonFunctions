# -*- coding: utf-8 -*-
"""
Standard FCCS functions

Created on Mon Apr 16 14:32:31 2018
@author: CVKelly

This has been updated on Apr 21, 2021 to load TIF files from Solis

7/16/22 - Corr2D and higher-order crosscorrelation funcdtions
9/22/22 - to make plotting the fit spectra easier with continuous functions included
10/31/22 - added taulist function for quick assessment of number of tau


"""
import numpy as np
from scipy.optimize import curve_fit as fit
import matplotlib.pyplot as plt
from os import listdir, remove
from os.path import exists
#from PIL import Image
from skimage import io
import matplotlib

class dumb():
    pass

############################
# read files saved by LabVIEW and Solis
def load_meta_funct(file)  :
    file = file.replace('-RawData','')
    file = file.replace('.dat','')
    file = file.replace('-Corr','')
    file = file.replace('Fit','')
    file = file.replace('Para','')
    file = file.replace('-Meta','')
    file = file+'-Meta.dat'
    a = np.loadtxt(file)
    meta = dumb()
    meta.numframes = a[0].astype(np.uint32)
    meta.realexp = a[1]
    meta.realHz = a[2]
    meta.setexp = a[3]
    meta.sensorw = a[4].astype(np.uint16)
    meta.sensorh = a[5].astype(np.uint16)
    meta.binw = a[6].astype(np.uint16)
    meta.binh = a[7].astype(np.uint16)
    meta.width1 = a[8].astype(np.uint16)
    meta.height1 = a[9].astype(np.uint16) # width for line extraction
    meta.top1 = a[10].astype(np.uint16)
    meta.bot1 = a[11].astype(np.uint16) # center line for extraction
    meta.width2 = a[12].astype(np.uint16)
    meta.height2 = a[13].astype(np.uint16)
    meta.top2 = a[14].astype(np.uint16)
    meta.bot2 = a[15].astype(np.uint16)
    meta.numTau = a[16]
    try:
        meta.temp = a[18]
        meta.maxTau = a[17]
    except:
        meta.maxTau = 1
        meta.temp = a[17]
    return meta,a

def load_metaTIF_funct(file)  :
    file = file.replace('-RawData','')
    file = file.replace('.dat','')
    file = file.replace('-Corr','')
    file = file.replace('Fit','')
    file = file.replace('Para','')
    file = file.replace('-Meta','')
    file = file+'-Meta.dat'
    a = np.loadtxt(file)
    meta = dumb()
    meta.numframes = a[0].astype(np.uint32)
    meta.realexp = a[1]
    meta.realHz = a[2]
    meta.setexp = a[3]
    meta.sensorw = a[4].astype(np.uint16)
    meta.sensorh = a[5].astype(np.uint16)
    meta.binw = a[6].astype(np.uint16)
    meta.binh = a[7].astype(np.uint16)
    meta.width1 = a[8].astype(np.uint16)
    meta.height1 = a[9].astype(np.uint16) # width for line extraction
    meta.top1 = a[10].astype(np.uint16)
    meta.bot1 = a[11].astype(np.uint16) # center line for extraction
    meta.width2 = a[12].astype(np.uint16)
    meta.height2 = a[13].astype(np.uint16)
    meta.top2 = a[14].astype(np.uint16)
    meta.bot2 = a[15].astype(np.uint16)
    meta.numTau = a[16]
    try:
        meta.temp = a[18]
        meta.maxTau = a[17]
    except:
        meta.maxTau = 1
        meta.temp = a[17]
    return meta,a
    
def load_FRET_meta_funct(file)  :
    file = file.replace('-RawData','')
    file = file.replace('.dat','')
    file = file.replace('-Corr','')
    file = file.replace('Fit','')
    file = file.replace('Para','')
    file = file.replace('-Meta','')
    file = file+'-Meta.dat'
    a = np.loadtxt(file)
    meta = dumb()
    meta.numframes = a[0].astype(np.uint32)
    meta.realexp = a[1]
    meta.setexp = a[2]
    meta.sensorw = a[3].astype(np.uint16)
    meta.sensorh = a[4].astype(np.uint16)
    meta.binw = a[5].astype(np.uint16)
    meta.binh = a[6].astype(np.uint16)
    meta.bot1 = a[7].astype(np.uint16) # center line for extraction
    meta.height1 = a[8].astype(np.uint16) # width for line extraction
    meta.fret_im_height = a[9].astype(np.uint16)
    meta.fret_im_width = a[10].astype(np.uint16)
    meta.fret_im_step = a[11].astype(np.uint16)
    meta.sensor_temp = a[12].astype(np.uint16)
    return meta,a

def load_data_funct(file,meta):
    file = file.replace('-RawData','')
    file = file.replace('.dat','')
    file = file.replace('-Corr','')
    file = file.replace('Fit','')
    file = file.replace('Para','')
    file = file.replace('-Meta','')
    file = file+'-RawData.dat'
    f = open(file, "r")
    a = np.fromfile(f, dtype=np.uint16,count=-1,sep='')
    data = a.reshape((meta.numframes,meta.sensorw,meta.sensorh))
    data = data.swapaxes(0, 2)
    data = data.swapaxes(0, 1)
    f.close()
    return data

def load_TIF_funct(file):
    #file must be the full path to the 3D TIF file
#    im = Image.open(file)
    im = io.imread(file).astype(np.float64)
    n_frames = im.shape[0]
    hei = im.shape[1]
    wid = im.shape[2]
    data = im.transpose(1,2,0)
    return data,n_frames,wid,hei

###########################
# Manipulate TIF data
def get_lines_from_array(data,linenum=-1,width=4):
    s0 = data.shape[0]  # data height
    s1 = data.shape[1]  # data width
    s2 = data.shape[2]  # number of frames, i.e. time
    alldatalines = np.zeros((s1,s2)) ## for horizontal lines with proper prism orientation
    if linenum == -1:
        d1 = np.sum(data,axis=2)
        d1 = np.sum(d1,axis=1)
        args = np.argsort(d1)
        linenum = args[-2]
    if width/2 == np.round(width/2):  # even
        down = width/2-1
        up = width/2
    else:  # odd
        down = (width-1)/2
        up = (width-1)/2
    down = np.max(np.array([0,linenum-down]))
    up = np.min(np.array([s0-1,linenum+up+1])) ## for vertical lines with bad prism orientation
    down=down.astype(np.uint16)
    up=up.astype(np.uint16)
    for i in range(s2):
        line = data[down:up,:,i] ## for horizontal lines with proper prism orientation
        alldatalines[:,i] = np.sum(line,axis=0)
    return alldatalines

# %% get calibration peak if only 1 fluor is present
def get_calib_from_alldata_ave(alldatalines):
    # get standard fit funct from the time average line
    # used when only one color diffuser is present
    calib = np.mean(alldatalines,axis=1)
    calib = calib/sum(calib)
    return calib

def convert_and_save_3Ddata_as_calib(data,file_to_save,linenum=-1,width=4):
    # convert_data_to_calib(data,file_to_save)
    # data = array of (width, height, frame_num) if data.dim == 3
    # data = array of (width, height) if data.dim = 2
    #   use  data = load_TIF_funct(file) if necessary
    # linenum = center of good data
    # width = how many rows to ave togeter
    alldatalines = get_lines_from_array(data,linenum,width)
    spec = get_calib_from_alldata_ave(alldatalines)
    spec = spec  - np.min(spec)
    spec = spec/np.sum(spec)
    np.savetxt(file_to_save,spec)

    return spec

def TIF_to_calib_file(file,file_to_save='',linenum=-1,width=4):
    # file = full path to 3D .TIF file from Solis
    # file_to_save (optional) = new .npz file towrite the 1D matrix
    data,a,b,c = load_TIF_funct(file)
    if len(file_to_save) == 0:
        file_to_save = file[:file.find('.tif')]+'.dat'
    if exists(file_to_save):
        print('ERROR: FILE TO SAVE ALREADY EXISTS, line 232 in sff')
        print('   No file being saved. Process aborted.')
        return 'Error in sff.TIF_to_calib_file'
    spec = convert_and_save_3Ddata_as_calib(data,file_to_save,linenum,width)
    return spec

def resave_TIF_as_smaller_TIF(file,new_width=32):
    f = open('D:/Default FCS Data Folder/00000-FileWrite.txt')
    file_write = f.readline()
    f.close()
    data,a,b,c = load_TIF_funct(file)
    old_width = data.shape[1]
    cut = int((old_width-new_width)/2)
    if (cut < old_width/2) and (cut > 0):
        data_smaller = data[:,cut:-cut,:].astype(np.uint16)
    else:
        data_smaller = data.astype(np.uint16)
    if data_smaller.shape != data.shape:
        io.call_plugin('imsave',plugin='tifffile',file=file_write,data=data_smaller.transpose(2,0,1),imagej=True, metadata={'axes': 'XYT', 'fps': 10.0})
    return data_smaller # 'successfully saved TIF as NPY'

def resave_DATA_as_smaller_TIF(data,file_write,new_width=32,linenum=-1,new_height=4):
    # crop width to be new_width=32
    old_width = data.shape[1]
    cut = int((old_width-new_width)/2)
    if (cut < old_width/2) and (cut > 0):
        data_smaller = data[:,cut:-cut,:].astype(np.uint16)
    else:
        data_smaller = data.astype(np.uint16)
    # crop height to be new_height=4 around brightest line
    old_height = data.shape[0]
    if new_height < old_height:
        if linenum == -1:
            d1 = np.sum(data,axis=2)
            d1 = np.sum(d1,axis=1)
            args = np.argsort(d1)
            linenum = args[-1]
        if new_height/2 == np.round(new_height/2):  # even
            down = new_height/2-1
            up = new_height/2
        else:  # odd
            down = (new_height-1)/2
            up = (new_height-1)/2
        down = np.max(np.array([1,linenum-down]))
        up = np.min(np.array([old_height,linenum+up+1]))
        down=down.astype(np.uint16)
        up=up.astype(np.uint16)
        data_smaller2 = data_smaller[down:up,:,:]
    else:
        data_smaller2 = data_smaller

    if data_smaller2.shape != data.shape:
        print('   Data has been resized and resaved. New size:',data_smaller2.shape)

    if data_smaller.shape != data.shape:
        io.call_plugin('imsave',plugin='tifffile',file=file_write,data=data_smaller2.transpose(2,0,1),imagej=True, metadata={'axes': 'XYT', 'fps': 10.0})
    return data_smaller #, txt

####################################
# get inverse matrixes
# %% find the inverse matrices to use for fitting
def get_inv_mat1(calib):
    o = np.ones(calib.shape[0])
    o = o/sum(o)
    m = np.zeros(4)
    m[0] = sum(calib**2)
    m[1] = sum(o*calib)
    m[2] = m[1]
    m[3]= sum(o**2)
    M = np.matrix(((m[0],m[1]),(m[2],m[3])))
    Mi = np.linalg.inv(M)
    return Mi

def get_inv_mat2(calib1,calib2):
    calib1 = calib1/np.sum(calib1)
    calib2 = calib2/np.sum(calib2)
    o = np.ones(calib1.shape[0])
    o = o/np.sum(o)
    M = np.matrix(((np.sum(calib1**2)    ,np.sum(calib1*calib2),np.sum(calib1*o)),
                   (np.sum(calib1*calib2),np.sum(calib2**2)    ,np.sum(calib2*o)),
                   (np.sum(calib1*o)     ,np.sum(o*calib2)     ,np.sum(o**2) )))
    Mi = np.linalg.inv(M)
    return Mi
    
def get_inv_mat2_noBack(calib1,calib2):
    M = np.matrix(((np.sum(calib1**2)    ,np.sum(calib1*calib2)),
                   (np.sum(calib1*calib2),np.sum(calib2**2))))
    Mi = np.linalg.inv(M)
    return Mi

def get_inv_mat3(calib1,calib2,calib3):
    o = np.ones(calib1.shape[0])
    o = o/sum(o)
    M = np.matrix(((sum(calib1**2)    ,sum(calib1*calib2),sum(calib1*calib3),sum(calib1*o)),
                   (sum(calib1*calib2),sum(calib2**2)    ,sum(calib2*calib3),sum(calib2*o)),
                   (sum(calib1*calib3),sum(calib2*calib3),sum(calib3**2),    sum(calib3*o)),
                   (sum(calib1*o)     ,sum(o*calib2)     ,sum(o*calib3),     sum(o**2) )))
    Mi = np.linalg.inv(M)
    return Mi

def get_inv_mat3_noBack(calib1,calib2,calib3):
    M = np.matrix(((np.sum(calib1**2)    ,np.sum(calib1*calib2),np.sum(calib1*calib3)),
                   (np.sum(calib1*calib2),np.sum(calib2**2)    ,np.sum(calib2*calib3)),
                   (np.sum(calib1*calib3),np.sum(calib2*calib3),np.sum(calib3**2)) ))
    Mi = np.linalg.inv(M)
    return Mi

def get_inv_mat4(calib1,calib2,calib3,calib4):
    o = np.ones(calib1.shape[0])
    o = o/sum(o)
    M = np.matrix(((sum(calib1**2)    ,sum(calib1*calib2),sum(calib1*calib3),sum(calib1*calib4),sum(calib1*o)),
                   (sum(calib1*calib2),sum(calib2**2)    ,sum(calib2*calib3),sum(calib2*calib4),sum(calib2*o)),
                   (sum(calib1*calib3),sum(calib2*calib3),sum(calib3**2),    sum(calib3*calib4),sum(calib3*o)),
                   (sum(calib1*calib4),sum(calib2*calib4),sum(calib3*calib4),sum(calib4**2),    sum(calib4*o)),
                   (sum(calib1*o)     ,sum(o*calib2)     ,sum(o*calib3),     sum(o**calib4),    sum(o**2) )))
    Mi = np.linalg.inv(M)
    return Mi


def get_inv_mat4_noBack(calib1,calib2,calib3,calib4):
    M = np.matrix(((sum(calib1**2)    ,sum(calib1*calib2),sum(calib1*calib3),sum(calib1*calib4)),
                   (sum(calib1*calib2),sum(calib2**2)    ,sum(calib2*calib3),sum(calib2*calib4)),
                   (sum(calib1*calib3),sum(calib2*calib3),sum(calib3**2),    sum(calib3*calib4)),
                   (sum(calib1*calib4),sum(calib4*calib2),sum(calib4*calib3),sum(calib4**2)   )))
    Mi = np.linalg.inv(M)
    return Mi

def get_inv_matN_noBack(calib_list):
    if len(calib_list) == 2:
        Mi = get_inv_mat2_noBack(calib_list[0],calib_list[1])
    elif len(calib_list) == 3:
        Mi = get_inv_mat3_noBack(calib_list[0],calib_list[1],calib_list[2])
    elif len(calib_list) == 4:
        Mi = get_inv_mat4_noBack(calib_list[0],calib_list[1],calib_list[2],calib_list[3])
    else:
        Mi = -1
    return Mi
    
#######################################    
# fit lines with calibrations
# %% fit a single I vs X to find the contribution of each peak
def fit_one_time1(calib,Mi,data1):
    o = np.ones(calib.shape[0])
    o = o/sum(o)
    v1 = sum(calib*data1)
    v2 = sum(o*data1)
    V = np.matrix((v1,v2))
    V = V.transpose()
    inten = Mi*V
    return inten[0],inten[1]

def fit_one_time1_noback(data1):   # don't need this in a loop, just do the mean
    inten = np.mean(data1,axis=0)
    return inten[0],inten[1]

def fit_one_time2(calib1,calib2,Mi,data1):
    o = np.ones(calib1.shape[0])
    o = o/np.sum(o)
    v1 = np.sum(calib1*data1)
    v2 = np.sum(calib2*data1)
    v3 = np.sum(o*data1)
    V = np.matrix((v1,v2,v3))
    V = V.transpose()
    inten = Mi*V
    return inten[0][0,0],inten[1][0,0],inten[2][0,0]

def fit_one_time2_noback(calib1,calib2,Mi,data1):
    v1 = sum(calib1*data1)
    v2 = sum(calib2*data1)
    V = np.matrix((v1,v2))
    V = V.transpose()
    inten = Mi*V
    return inten[0],inten[1]

def fit_one_time3(calib1,calib2,calib3,Mi,data1):
    o = np.ones(calib1.shape[0])
    o = o/sum(o)
    v1 = sum(calib1*data1)
    v2 = sum(calib2*data1)
    v3 = sum(calib3*data1)
    v4 = sum(o*data1)
    V = np.matrix((v1,v2,v3,v4))
    V = V.transpose()
    inten = Mi*V
    return inten[0],inten[1],inten[2],inten[3]

def fit_one_time3_noBack(calib1,calib2,calib3,Mi,data1):
    v1 = np.sum(calib1*data1)
    v2 = np.sum(calib2*data1)
    v3 = np.sum(calib3*data1)
    V = np.matrix((v1,v2,v3))
    V = V.transpose()
    inten = Mi*V
    return inten[0][0,0],inten[1][0,0],inten[2][0,0]

def fit_one_time4(calib1,calib2,calib3,calib4,Mi,data1):
    o = np.ones(calib1.shape[0])
    o = o/sum(o)
    v1 = sum(calib1*data1)
    v2 = sum(calib2*data1)
    v3 = sum(calib3*data1)
    v4 = sum(calib4*data1)
    v5 = sum(o*data1)
    V = np.matrix((v1,v2,v3,v4,v5))
    V = V.transpose()
    inten = Mi*V
    return inten[0],inten[1],inten[2],inten[3],inten[4]

def fit_one_time4_noBack(calib1,calib2,calib3,calib4,Mi,data1):
    o = np.ones(calib1.shape[0])
    o = o/sum(o)
    v1 = sum(calib1*data1)
    v2 = sum(calib2*data1)
    v3 = sum(calib3*data1)
    v4 = sum(calib4*data1)
    V = np.matrix((v1,v2,v3,v4))
    V = V.transpose()
    inten = Mi*V
    return inten[0],inten[1],inten[2],inten[3]

# %% repeat the fitting for all times in
def fit_many_times1(calib,Mi,alldatalines):
    s1 = alldatalines.shape[1] # number of times
    inten0 = np.zeros(s1)
    inten1 = np.zeros(s1)
    for i in range(s1):
        data1 = alldatalines[:,i]
        inten0[i],inten1[i] = fit_one_time1(calib,Mi,data1)
    return inten0,inten1

def fit_many_times1_noback(alldatalines):
    inten0 = np.mean(alldatalines,axis=0)
    return inten0

def fit_many_times2(calib1,calib2,Mi,alldatalines):
    if len(alldatalines.shape)==2:
        s1 = alldatalines.shape[1] # number of times
        inten0 = np.zeros(s1)
        inten1 = np.zeros(s1)
        inten2 = np.zeros(s1)
        for i in range(s1):
            data1 = alldatalines[:,i]
            inten0[i],inten1[i],inten2[i] = fit_one_time2(calib1,calib2,Mi,data1)
    else:
        inten0,inten1,inten2 = fit_one_time2(calib1,calib2,Mi,alldatalines)
    return inten0,inten1,inten2

def fit_many_times2_noBack(calib1,calib2,Mi,alldatalines):
    if len(alldatalines.shape)==2:
        s1 = alldatalines.shape[1] # number of times
        inten0 = np.zeros(s1)
        inten1 = np.zeros(s1)
        for i in range(s1):
            data1 = alldatalines[:,i]
            inten0[i],inten1[i] = fit_one_time2_noBack(calib1,calib2,Mi,data1)
    else:
        inten0,inten1 = fit_one_time2_noBack(calib1,calib2,Mi,alldatalines)
    return inten0,inten1

def fit_many_times3(calib1,calib2,calib3,Mi,alldatalines):
    s1 = alldatalines.shape[1] # number of times
    inten0 = np.zeros(s1)
    inten1 = np.zeros(s1)
    inten2 = np.zeros(s1)
    inten3 = np.zeros(s1)
    for i in range(s1):
        data1 = alldatalines[:,i]
        inten0[i],inten1[i],inten2[i],inten3[i] = fit_one_time3(calib1,calib2,calib3,Mi,data1)
    return inten0,inten1,inten2,inten3

def fit_many_times3_noBack(calib1,calib2,calib3,Mi,alldatalines):
    if len(alldatalines.shape)>1:
        s1 = alldatalines.shape[1] # number of times
    else:
        s1 = 1
    inten0 = np.zeros(s1)
    inten1 = np.zeros(s1)
    inten2 = np.zeros(s1)
    for i in range(s1):
        if len(alldatalines.shape)>1:
            data1 = alldatalines[:,i]
        else:
            data1 = alldatalines.copy()
        inten0[i],inten1[i],inten2[i] = fit_one_time3_noBack(calib1,calib2,calib3,Mi,data1)
    return inten0,inten1,inten2
    
def fit_many_times4(calib1,calib2,calib3,calib4,Mi,alldatalines):
    if len(alldatalines.shape)>1:
        s1 = alldatalines.shape[1] # number of times
    else:
        s1 = 1
    inten0 = np.zeros(s1)
    inten1 = np.zeros(s1)
    inten2 = np.zeros(s1)
    inten3 = np.zeros(s1)
    inten4 = np.zeros(s1)
    for i in range(s1):
        if len(alldatalines.shape)>1:
            data1 = alldatalines[:,i]
        else:
            data1 = alldatalines.copy()
        inten0[i],inten1[i],inten2[i],inten3[i],inten4[i] = fit_one_time4(calib1,calib2,calib3,calib4,Mi,data1)
    return inten0,inten1,inten2,inten3,inten4

def fit_many_times4_noBack(calib1,calib2,calib3,calib4,Mi,alldatalines):
    if len(alldatalines.shape)>1:
        s1 = alldatalines.shape[1] # number of times
        inten0 = np.zeros(s1)
        inten1 = np.zeros(s1)
        inten2 = np.zeros(s1)
        inten3 = np.zeros(s1)
        for i in range(s1):
            data1 = alldatalines[:,i]
            inten0[i],inten1[i],inten2[i],inten3[i] = fit_one_time4_noBack(calib1,calib2,calib3,calib4,Mi,data1)
    else:
        inten0,inten1,inten2,inten3 = fit_one_time4_noBack(calib1,calib2,calib3,calib4,Mi,alldatalines)

    return inten0,inten1,inten2,inten3

def fit_many_timesN_noBack(calib_list,Mi,alldatalines):
    if len(calib_list) == 2:
        i0,i1 = fit_many_times2_noBack(calib_list[0],calib_list[1],Mi,alldatalines)
        i2 = 0
        i3 = 0
    elif len(calib_list) == 3:
        i0,i1,i2 = fit_many_times3_noBack(calib_list[0],calib_list[1],calib_list[2],Mi,alldatalines)
        i3 = 0
    elif len(calib_list) == 4:
        i0,i1,i2,i3 = fit_many_times4_noBack(calib_list[0],calib_list[1],calib_list[2],calib_list[3],Mi,alldatalines)
    else:
        inten_list = -1
    return i0,i1,i2,i3

###################################
# %% do correlation analysis
def get_tau_list(time_per_frame,max_tau,num_tau):
    tau_list  = np.logspace(np.log10(1),np.log10(max_tau/time_per_frame),num_tau)
    tau_list = tau_list.astype(np.uint64)
    tau_list = np.unique(tau_list)
    tau_list = tau_list.astype(np.uint32)
    return tau_list

def add_zero_to_tau_list(tau_list):
    tau_list = np.zeros(len(tau_list)+1)
    tau_list[1:] = tau_list
    tau_list = tau_list.astype(np.uint32)
    return tau_list
    
def calc_xcorr(inten0,inten1,dt):
    dt = dt.astype(np.uint32)
    G = np.zeros(dt.shape[0])
    for i in range(dt.shape[0]):
        G[i] = np.mean(inten0[:int(inten0.shape[0]-dt[i])]*inten1[dt[i]:])
    G = G/np.mean(inten0)/np.mean(inten1)-1
    return dt,G

def Corr2D(i1,i2,i3,tau_list):
    tau_list = tau_list.astype(np.uint32)
    G = np.zeros((len(tau_list),len(tau_list)))
    for i,tau1 in enumerate(tau_list):
        for j,tau2 in enumerate(tau_list):
            numtimes = int(len(i1)-np.max([tau1,tau2]))-1
            G[i,j] = np.mean(i1[:numtimes]*i2[tau1:int(numtimes+tau1)]*i3[tau2:int(numtimes+tau2)])
    G = G/np.mean(i1)/np.mean(i2)/np.mean(i3)-1 #normalize G
    return G,tau_list

def Corr3D(i1,i2,i3,i4,tau_list):
    G = np.zeros((len(tau_list),len(tau_list),len(tau_list)))
    for i,tau1 in enumerate(tau_list):
        for j,tau2 in enumerate(tau_list):
            for k,tau3 in enumerate(tau_list):
                numtimes = int(len(i1)-np.max([tau1,tau2,tau3]))-1
                G[i,j,k] = np.mean(i1[:numtimes]*i2[tau1:int(numtimes+tau1)]*i3[tau2:int(numtimes+tau2)]*i4[tau3:int(numtimes+tau3)])
    G = G/np.mean(i1)/np.mean(i2)/np.mean(i3)/np.mean(i4)-1 #normalize G
    return G,tau_list

############################
# fit correlation results
def fit_xcorr(G,dtlistsec,model):
    ## Model:
    #    '2d'     = Brownian 2D model
    #    '2danom' = Anomalous 2D model
    #    '3d'     = Brownian 3D model

    if (model.find('2danom') > -1 ):
        f = lambda x,td,a,alpha,c: a/(1+(x/td)**alpha)+c
        numparam = 3
        start = (0.005,1,0.9,0.0)
        bound = ((0.0,0.0,0.0,0.0),(1.0,np.inf,np.inf,np.inf))
    elif (model.find('2d') > -1 ):
        f = lambda x,td,a,c: a/(1+(x/td))+c
        numparam = 3
        start = (0.005,1,0)
        bound = ((0.0,-np.inf,-np.inf),(1.0,np.inf,np.inf))##sonali changed lower bound from 0.0 to -np.inf on 11/22/21

    elif (model.find('3dx') > -1):
        f = lambda x,td,td2,a,a2,r: a/(1+(x/td))/(1+r**-2*np.sqrt(x/td)) + a2/(1+(x/td2))/(1+r**-2*np.sqrt(x/td2))
        numparam = 3
        start = (0.005,0.01,0.1,0.5,0.5)
        bound = ((0.0,0.0,0.0,0.0,0.0),(1.0,np.inf,np.inf,np.inf,np.inf))
    #elif (model.find('3d') > -1):
        #f = lambda x,td,a,r: a/(1+(x/td))/(1+r**-2*np.sqrt(x/td))
        #numparam = 3
        #start = (0.005,1,1)
        #bound = ((0.0,-np.inf,0.01),(1.0,np.inf,np.inf))#sonali changed lower bound from 0.0 to -np.inf on 11/22/21
    elif (model.find('3d') > -1):
        f = lambda x,td,a,r: a/(1+(x/td))/np.sqrt(1+(r**-2)*(x/td))
        numparam = 3
        start = (0.005,1,1)
        bound = ((0.0,-np.inf,0.01),(1.0,np.inf,np.inf))

    else:
        print('no fitting function used')
        f = lambda x,a,b: a/x+b
        numparam = 2
        start = (1.0,1.0)
        bound = ((0.0,0.0),(np.inf,np.inf))

    fres,covf = fit(f,dtlistsec,G,p0=start,bounds=bound)
#    print(fres)
#    param = np.zeros(numparam)
#    for i in range(numparam):
#        param[i] = fres[i][0]
    return f, fres, covf, numparam  #, param

#########################################
# plot things
def get_colors(num_cols,cmap='jet'):
# num_cols = 4
    var = np.linspace(0,1,num_cols)
    cols = np.zeros((num_cols,4))
    for i in range(num_cols):
        if cmap == 'jet':
            cols[i,:] = matplotlib.cm.jet(var[i])
        elif cmap == 'hsv':
            cols[i,:] = matplotlib.cm.hsv(var[i])
        elif cmap == 'gray':
            cols[i,:] = matplotlib.cm.gray(var[i])
        elif cmap == 'rainbow':
            cols[i,:] = matplotlib.cm.rainbow(var[i])
        elif cmap == 'tab20b':
            cols[i,:] = matplotlib.cm.tab20b(var[i])
        elif cmap == 'nipy_spectral':
            cols[i,:] = matplotlib.cm.nipy_spectral(var[i])
        elif cmap == 'viridis':
            cols[i,:] = matplotlib.cm.viridis(var[i])
        else:
            cols[i,:] = matplotlib.cm.jet(var[i])
            print('   ', cmap,' not found. Jet used.')
    cols = cols.astype(np.float16)
    return(cols)
    
def plot_inten0(inten0,time_per_frame,actuallyplot):
    t = np.linspace(0,inten0.shape[0]*time_per_frame,inten0.shape[0])
    N = 20  # make a round number
#    smoothedI = np.convolve(inten0, np.ones((N,))/N, mode='valid')
#    offset = np.array([N/2],dtype=np.int16)
#    smoothedt = t[offset[0]:-(offset[0]-1)]
    if actuallyplot:
        plt.plot(t,inten0,label='intensity')
#        plt.plot(smoothedt,smoothedI,label='smoothed')
        plt.xlabel('Time (sec)')
        plt.ylabel('Intensity')
        plt.title('Single I vs. t')
        plt.xlim([0,np.max(t)])
        plt.legend()
        plt.show()
    return t, inten0  # , smoothedt, smoothedI

def plot_xcorr_and_fit(G,dtlistsec,f,fres,actuallyplot):
    xfit = np.logspace(np.log10(np.min(dtlistsec)),np.log10(np.max(dtlistsec)),1000)
    yfit = np.zeros(xfit.shape[0])
    for i in range(xfit.shape[0]):
        yfit[i] = f(xfit[i],*fres)
    if actuallyplot:
        plt.semilogx(dtlistsec,G,'o',label='data')
        plt.semilogx(xfit,yfit,'-',label='fit')
        plt.xlabel('tau (sec)')
        plt.ylabel('G(tau)')
#        plt.title('Correlation')
        plt.show()
    return dtlistsec, G, xfit, yfit

def plot_xcorr(G,dtlistsec,actuallyplot):
    xfit = np.logspace(np.log10(np.min(dtlistsec)),np.log10(np.max(dtlistsec)),1000)
    yfit = np.zeros(xfit.shape[0])
    if actuallyplot:
        plt.semilogx(dtlistsec,G,'o',label='data')
        plt.xlabel('tau (sec)')
        plt.ylabel('G(tau)')
#        plt.title('Autocorrelation')
        plt.show()

    return dtlistsec, G, xfit, yfit



########################
# assorted file manipulations
def make_file_string(num):
    z = '00000'
    if (type(num)==type(3)) | (type(num)==type(3.3)):
        num = np.array([num])
    out = []
    for i in range(len(num)):
        n = str(int(num[i]))
        zn = z[:(5-len(n))]
        out.append('\\'+zn+n)
    return out

# %% assorted functions for analysis
# def make_file_string(num):
    # z = '00000'
    # if type('sadf') == type(num):
        # num = int(num)
    # if not (type('sadf') == type(num)):
        # num = np.array([num])
    # out = []
    # for i in range(num.shape[0]):
        # n = str(int(num[i]))
        # zn = z[:(5-len(n))]
        # out.append(zn+n)
    # if len(out) == 1:
        # out = out[0]
    # return out
    
def get_filename_from_num(fold,num):
    fnumstr = num
    if type(num) != type('asdf'):
        fnumstr = make_file_string(num)
    files_in_fold = listdir(fold)
    for file in files_in_fold:
        if file.startswith(fnumstr) and file.endswith('RawData.dat'):
            file = file[:-12]
            break
    return file

def import_npz(npz_file,allow_pickle=False):
    # This doesn't work as part of a Python module
    Data = np.load(npz_file,allow_pickle=allow_pickle)
    for varName in Data:
        globals()[varName] = Data[varName]