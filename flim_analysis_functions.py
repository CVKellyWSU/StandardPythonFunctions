# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:31:31 2022

@author: KellyLab

assorted functions for analysis of FLIM data
"""
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# import pandas as pd
from os import listdir #, remove
# from os.path import isdir, getsize


def remove_extension_list(list1):
    for i,l in enumerate(list1):
        if l[-4] == '.':
            list1[i] = l[:-4]
    return list1

def get_Tnums(fold):
    files = listdir(fold)
    files_kept = []
    times = []
    for f in files:
        s2 = f.find('s_1.ptu')
        if s2 > 0:
            files_kept.append(f)
            s1 = f.find('_T')
            f2 = f[(s1+2):s2]
            times.append(float(f2))
    times = np.array(times)
    arg = np.argsort(times)
    times = times[arg]
    files = []
    for a in arg:
        files.append(files_kept[a])
    files = remove_extension_list(files)
    return files, times

def check_same_lists(list1,list2):
    same = 1
    if len(list1) == len(list2):
        for i1 in range(len(list1)):
            if list1[i1] != list2[i1]:
                same = 0
    return same

def align_string_lists(list1,list2):
    # make list2 have the same order as list1
    list1 = remove_extension_list(list1)
    list2 = remove_extension_list(list2)
    list2_fixed = []

    if len(list1) == len(list2):
        args = [-1]*len(list1)
        for i1,l1 in enumerate(list1):
            for i2, l2 in enumerate(list2):
                if l1 == l2:
                    args[i1] = i2
                    break
        args = np.array(args)
        changed = 0
        if np.min(args) > -1:
            for i2, l2 in enumerate(list2):
                list2_fixed.append(list2[args[i2]])
                if args[i2] != i2:
                    changed = 1
    else:
        changed = -1
    return list2_fixed,changed

def read_flim_image_ascii(file):
    # converts the ASCII text file export of SymPhoTime to a Python array
    f = open(file,'r')
    alllines = f.read()
    f.close()
    n1 = alllines.find('\n')
    seplines = []
    seplines.append(alllines[:n1])
    while (n1+1) < len(alllines):
        n2 = alllines[(n1+1):].find('\n')
        seplines.append(alllines[(n1+1):(n1+n2+1)])
        n1 += n2+1
    # find start of arrays
    x0list = []
    for i,s in enumerate(seplines):
        sw = s.find('(x0 | y0)')
        if sw>-1:
            x0list.append(i-1)
    # find the labels of each array
    labs = []
    for lnum in x0list:
        labs.append(seplines[lnum])
    # extract the actual array data
    array_all = []
    for lnum1 in x0list:
        array_now = []
        going = 1
        lnum2 = lnum1+3
        while going == 1 and lnum2 < len(seplines):
            lnow = seplines[lnum2]
            if lnow.find('\t')>-1:
                array_now.append(lnow.split('\t'))
                lnum2 += 1
            else:
                going = 0
        array_all.append(np.array(array_now).astype(float))
    array_all = np.array(array_all)
    return array_all, labs

def plot_arrayall_PQascii(array_all, labs):
    # array_all is a list of arrays extracted from the PicoQuant ASCII image export
    # array_all is created by "read_flim_image_ascii"
    fig = plt.figure(dpi=200,figsize=(5,len(array_all)*2))
    for i in range(len(array_all)):
        fig.add_subplot(len(array_all),1,i+1)
        plt.imshow(array_all[i])
        plt.colorbar()
        plt.clim([np.percentile(array_all[i],1),
                  np.percentile(array_all[i],99)])
        plt.title(labs[i])
    return fig

def load_all_PQ_ascii(foldASCII,file_part=-1):
    # combine all ASCII files from SymPhoTime into a single array
    # only load files that contain "file_part" if file_part != 1
    consider_files = listdir(foldASCII)
    ASCII_files = []
    file_partn = file_part
    for cf in consider_files:
        if file_part == -1:
            file_partn = cf
        if cf.find(file_partn) > -1 and cf.find('.txt') > -1:
            ASCII_files.append(cf)
    if len(ASCII_files) > 0:
        array_all_all = []
        for i,file in enumerate(ASCII_files[::-1]):
            array_all, labs = read_flim_image_ascii(foldASCII+file)
            if i == 0:
                array_all_all.append(array_all)
                sizematch = array_all.shape
            elif array_all.shape == sizematch:
                array_all_all.append(array_all)
        array_all_all = array_all_all[::-1]
        array_all_all = np.array(array_all_all)
    else:
        array_all_all = -1
        labs = -1
        print('ASCII folder not working')
    return array_all_all, labs

def make_mask(fold,index,thresh):
    # read all ASCII images from PQ in a folder
    # median over time for the proper index
    # select the proper index for intensity data
    array,  labs = load_all_PQ_ascii(fold)
    if labs == -1:
        print('FOLDER NOT WORKING!!')
        mask = [0]
    else:
        medianinten = np.median(array[:,index,:,:],axis=0)
        t2 = np.percentile(medianinten,thresh)
        mask = np.ones(medianinten.shape)
        mask[medianinten<t2] = 0
    return mask

def import_npz(npz_file,allow_pickle=False):
    # This doesn't work as part of a Python module
    Data = np.load(npz_file,allow_pickle=allow_pickle)
    for varName in Data:
        globals()[varName] = Data[varName]
