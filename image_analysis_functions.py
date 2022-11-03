# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 19:11:32 2021

@author: cvkelly

last updated:
2022 02 18

"""
import cv2
import matplotlib
import numpy as np
from skimage import io
import tifffile
from os import rename
from scipy.fft import fftn, ifftn, fftshift
from time import gmtime, strftime
import h5py

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


def save_tiff(data,file_write):
    if len(data.shape)==3:    # data should be in format (x,y,t)
        io.call_plugin('imsave',
                        plugin='tifffile',
                        file=file_write,
                        data=data.transpose(2,0,1),
                        imagej=True,
                        metadata={'axes': 'TYX', 'fps': 10.0})
    if len(data.shape)==4:    # data should be in format (x,y,c,t)
        io.call_plugin('imsave',
                       plugin='tifffile',
                       file=file_write,
                       data=data,
                       imagej=True,
                       metadata={'axes': 'XYCT', 'fps': 10.0})
    return 'save success'

def load_tif_stack(file):
    data = io.imread(file)
    # data = im.transpose(1,2,0)
    return data

def load_tif_stack_as_float(file):
    # I don't get why this code isn't working...

    data = io.imread(file).astype(np.float64)
    # if len(data.shape) == 3:
    #    data = data.transpose(1,2,0)
    return data

def load_MMtif_time(file,numtime=1,numcol=1):
    tmeta = tifffile.TiffFile(file)

    s = tmeta.ome_metadata
    d1 = 'DeltaT='
    d2 = 'DeltaTUnit'
    w1 = s.find(d1)
    w2 = s.find(d2)
    time = []
    cnt = 0
    while (w1 > 0 and cnt < 10**6):
        cnt += 1
        time.append(float(s[(w1+8):(w2-2)])) # originally in milliseconds
        s = s[(w2+5):]
        w1 = s.find(d1)
        w2 = s.find(d2)
    if not (numtime == 1 and numcol == 1):
        time = np.array(time).reshape((numtime,numcol))
    time = time/1000/60
    return time # in minutes

def stretch_image(data,frac_min=0.01,frac_max=0.01):
    data = data.astype(np.float64)
    dl = data.flatten()
    dl.sort()
    argmin = int(dl.shape[0]*frac_min)
    argmax = int(dl.shape[0]*frac_max)
    old_min = dl[argmin]
    old_max = dl[-argmax]-old_min

    data = data-old_min
    data = data/old_max
    data[data<0]=0
    data[data>1]=1
    data = data*old_max
    data = data+old_min

    return data,old_max,old_min

def smooth_image(a,size=3):
    kernel = np.ones((size,size),np.float32)
    kernel = kernel/np.sum(kernel)
    b = cv2.filter2D(a,-1,kernel)
    return b

#%% assorted image quantifications
def radial_average(c,maxr,numr):
    # assume the center of the image is the origin
    mid = int(c.shape[0]/2)
    a = 0
    if mid < c.shape[0]/2:
        a = 1
    x = np.arange(-mid,mid+a,1)

    X,Y = np.meshgrid(x,x)
    R = np.sqrt(X**2+Y**2)

    stepr = int(maxr/numr)
    stepr = np.max([stepr,1])

    r_edge = np.arange(0,maxr+stepr,stepr)
    g = np.zeros(len(r_edge)-1)
    for i in range(len(r_edge)-1):
        keep = np.all([R>r_edge[i],R<=r_edge[i+1]],axis=0)
        g[i] = np.mean(c[keep])

    r = (r_edge[1:]+r_edge[:-1])/2
    return r,g

def radial_average_orig(c,orig,numr=np.inf):

    # specify the origin
    maxr = np.min([c.shape[0]-orig[0],
                    c.shape[1]-orig[1],
                    orig[0],
                    orig[1]])

    x = np.arange(0,c.shape[0],1)-orig[1]
    y = np.arange(0,c.shape[1],1)-orig[0]

    X,Y = np.meshgrid(y,x)
    R = np.sqrt(X**2+Y**2)

    stepr = int(maxr/numr)
    stepr = np.max([stepr,1])

    r_edge = np.arange(0,maxr+stepr,stepr)
    g = np.zeros(len(r_edge)-1)
    for i in range(len(r_edge)-1):
        keep = np.all([R>r_edge[i],R<=r_edge[i+1]],axis=0)
        g[i] = np.mean(c[keep])

    r = (r_edge[1:]+r_edge[:-1])/2
    return r,g

def image_com(data):
    # data should be a 2D array
    background = np.mean([np.mean(data[:,:2]),
                          np.mean(data[:,-2:]),
                          np.mean(data[-2:,:]),
                          np.mean(data[:2,:])])
    data = data-background
    data[data<0] = 0
    xlist = np.arange(0,data.shape[0],1)
    ylist = np.arange(0,data.shape[1],1)
    com_y = np.sum(xlist*np.sum(data,axis=1))/np.sum(data)
    com_x = np.sum(ylist*np.sum(data,axis=0))/np.sum(data)
    return [com_x,com_y]


# %% do spatial correlations -- not verified
def spatial_autocorr_pre(image):
    b = fftn(image)
    b = np.abs(b)**2
    b = ifftn(b)
    b = np.real(fftshift(b))
    return b

def spatial_crosscorr_pre(image1,image2):
    b1 = fftn(image1)
    b2 = fftn(image2)
    b = b1 * np.conj(b2)
    b = ifftn(b)
    b = np.real(fftshift(b))
    return b

def spatial_corr_FFT(image1, image2, mask=0, maxr=100, numr=100):
    # uses FFT - unverified normalizations
    if np.all(mask==0):
        mask = np.ones(image1.shape)
    mask = mask/np.sum(mask)
    a = spatial_autocorr_pre(mask)
    b = spatial_crosscorr_pre(image1,image2)


    a[a<=0] = np.min(a[a>0])
    b[b<=0] = np.min(b[b>0])

    r1,g1 = radial_average(b,maxr,numr)
    r2,g2 = radial_average(a,maxr,numr)
    r = r1
    g = g1/g2

    return g,r

def spatial_corr_pre_manual(image1,image2,maxr = 50):
    dxlist = np.arange(-maxr,maxr+1,1)
    corr = np.zeros((len(dxlist),len(dxlist)))

    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    image1 = image1-np.mean(image1)
    image2 = image2-np.mean(image2)

    for i,dx in enumerate(dxlist):
        if dx > 0:
            a1 = image1[dx:,:]
            a2 = image2[:-dx,:]
        elif dx < 0:
            a2 = image2[-dx:,:]
            a1 = image1[:dx,:]
        else:
            a1 = image1.copy()
            a2 = image2.copy()
        for j,dy in enumerate(dxlist):
            if dy > 0:
                a3 = a1[:,dy:]
                a4 = a2[:,:-dy]
            elif dy < 0:
                a4 = a2[:,-dy:]
                a3 = a1[:,:dy]
            else:
                a3 = a1.copy()
                a4 = a2.copy()

            s1 = a3.shape[0]
            s2 = a3.shape[1]
            corr[i,j] = np.sum(a3 * a4) / s1 / s2

    # norm1 = np.mean(image1)
    # norm2 = np.mean(image2)
    # corr = corr # / norm1 / norm2
    return corr

def spatial_corr_manual(image1, image2, mask=0, maxr=100, numr=100):
    if np.all(mask==0):
        a = 1
    else:
        mask = mask/np.sum(mask)
        a = spatial_corr_pre_manual(mask,mask,maxr)

    if numr > maxr:
        numr = maxr

    b = spatial_corr_pre_manual(image1,image2,maxr)
    c = b/a
    r,g = radial_average(c,maxr,numr)

    return r,g

# %% rename files
def rename_file_number(file):
    z = '00000'
    sym = '-_ '
    loc = np.zeros(len(sym))
    for i,s in enumerate(sym):
        loc[i] = file.find(sym[i])
    loc[loc==-1] = np.inf
    minarg = np.argmin(loc)
    loc = loc[minarg]
    num = file[:loc]
    zn = z[:-len(num)]
    fnew = zn + file
    rename(file,fnew)
    return fnew

def print_time_str():
    t =  strftime("%Y-%m-%d %H:%M:%S", gmtime())
    return t

# %% for IMS analysis
def load_MMtif_time(file,numtime=1,numcol=1):
    tmeta = tifffile.TiffFile(file)

    s = tmeta.ome_metadata
    d1 = 'DeltaT='
    d2 = 'DeltaTUnit'
    w1 = s.find(d1)
    w2 = s.find(d2)
    time = []
    cnt = 0
    while (w1 > 0 and cnt < 10**6):
        cnt += 1
        time.append(float(s[(w1+8):(w2-2)]))
        s = s[(w2+5):]
        w1 = s.find(d1)
        w2 = s.find(d2)
    if not (numtime == 1 and numcol == 1):
        time = np.array(time).reshape((numtime,numcol))
    time = time/1000/60
    return time

def get_h5_file_info(h5_dataset):
    # Return the resolution levels, time points, channes, z levels, rows, cols, etc from a ims file
    # Pass in an opened f5 file
    # Get a list of all of the resolution options
    resolution_levels = list(h5_dataset)
    resolution_levels.sort(key = lambda x: int(x.split(' ')[-1]))

    # Get a list of the available time points
    time_points = list(h5_dataset[resolution_levels[0]])
    time_points.sort(key = lambda x: int(x.split(' ')[-1]))
    n_time_points = len(time_points)

    # Get a list of the channels
    channels = list(h5_dataset[resolution_levels[0]][time_points[0]])
    channels.sort(key = lambda x: int(x.split(' ')[-1]))
    n_channels = len(channels)

    # Get the number of z levels
    n_z_levels = h5_dataset[resolution_levels[0]][time_points[0]][channels[0]][
                   'Data'].shape[0]
    z_levels = list(range(n_z_levels))

    # Get the plane dimensions
    n_rows, n_cols = h5_dataset[resolution_levels[0]][time_points[0]][channels[0]][
                   'Data'].shape[1:]

    return resolution_levels, time_points, n_time_points, channels, n_channels, n_z_levels, z_levels, n_rows, n_cols

def get_timepoints(h5file):
    # get time each frame started in nanoseconds
    npnts = len(h5file['DataSetTimes']['Time'])
    timestart = np.zeros(npnts)
    for i in range(npnts):
        timestart[i] = h5file['DataSetTimes']['Time'][i][1]
    return timestart

def extract_h5_data(h5file):
    base_data = h5file['DataSet']
    file_info = get_h5_file_info(base_data)

    numtime = len(file_info[1])
    numcol = len(file_info[3])
    size1 = file_info[-2]
    size2 = file_info[-1]

    data = np.zeros((numtime,size1,size2,numcol)).astype(np.uint16)

    for i in range(numtime):
        for j in range(numcol):
            try:
                d = np.array(base_data[file_info[0][0]][file_info[1][i]][file_info[3][j]]['Data'][0]).astype(np.uint16)
                data[i,:,:,j] = d
            except:
                print('  time',i,j,' not loading')
                break
    data = remove_zeros(data)
    return data

def get_ims_data(file):
    if file[-4:] == '.ims':
        pass
    else:
        file = dset+'.ims'
    h5file = h5py.File(file,'r')
    data = extract_h5_data(h5file)
    times = get_timepoints(h5file) # in nanoseconds
    timestart = times[0]/10**9/60 # in min
    times = (times-times[0])/10**9/60 # in min
    h5file.close()
    return data,times

# %%
def bin_matrix(data,bx,by):
    # return a matrix of shape (n,m)
    n = data.shape[0]//bx
    m = data.shape[1]//by
    bs = data.shape[0]//n,data.shape[1]//m  # blocksize averaged over
    return np.reshape(np.array([np.sum(data[k1*bs[0]:(k1+1)*bs[0],k2*bs[1]:(k2+1)*bs[1]]) for k1 in range(n) for k2 in range(m)]),(n,m))

def bin_middle_two(data,bx,by):
    # remove empty rows and columns
    # assumes data is 4D array
    s = data.shape
    for i in range(s[0]):
        for j in range(s[3]):
            dtemp = bin_matrix(data[i,:,:,j],bx,by)
            if i == 0 and j == 0:
                newdata = np.zeros((s[0],dtemp.shape[0],dtemp.shape[1],s[3]))
            newdata[i,:,:,j] = dtemp
    return newdata

def remove_zeros(data):
    # assumes data is 4D
    # only consider middle two dimensions
    # remove empty rows and columns
    s = data.shape
    for i in range(s[1]):
        if np.sum(data[:,-1,:,:]) == 0:
            data = np.delete(data,data.shape[1]-1,1)
        else:
            break
    for i in range(s[2]):
        if np.sum(data[:,:,-1,:]) == 0:
            data = np.delete(data,data.shape[2]-1,2)
        else:
            break
    return data


def align_movie(data,r=15):
# calculate how much shifting is necessary to align the frames
# keep the full image for each dimension's shift
# data shape = num frames, x-dim, y-dim
# r = range of maximum shift

    shift = np.zeros((data.shape[0],2))
        
    rlist = np.arange(-(r-1),r,1)
    msd = np.zeros((len(rlist),2))
    dgoal = np.mean(data.astype(np.float32),axis=0)
    dgoal = dgoal[r:-r,r:-r]
    dgoal = dgoal - np.mean(dgoal)
    dgoal = dgoal/np.std(dgoal) 

    for frame in range(data.shape[0]):
        for ri,rnow in enumerate(rlist):
            d1 = data[frame,(r+rnow):-(r-rnow),r:-r]
            d2 = data[frame,r:-r,(r+rnow):-(r-rnow)]

            d1 = d1 - np.mean(d1)
            d1 = d1/np.std(d1)
            d2 = d2 - np.mean(d2)
            d2 = d2/np.std(d2)
            
            msd[ri,0] = np.mean((d1-dgoal)**2)
            msd[ri,1] = np.mean((d2-dgoal)**2)
            shift[frame,0] = rlist[np.argmin(msd[:,0])]
            shift[frame,1] = rlist[np.argmin(msd[:,1])]

    shift = shift.astype(np.int16)
    ms = int(np.max(np.abs(shift)))+1 # max shift used
    data2 = np.zeros(data[:,ms:-ms,ms:-ms].shape).astype(np.uint16)
    for i in range(data.shape[0]):
        data2[i,:,:] = data[i,(ms+shift[i,0]):(-ms+shift[i,0]),(ms+shift[i,1]):(-ms+shift[i,1])]

    return data2, shift