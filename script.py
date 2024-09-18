import numpy as np
import matplotlib.pyplot as plt
import os
import re
from skimage.filters import threshold_otsu, threshold_li
from skimage.restoration import denoise_wavelet
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
from numba import njit

# load envelope images
directory = 'testing'
files = [directory + "/" + f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.startswith('optimal')]
def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else float('inf')
sorted_files = sorted(files, key=extract_number)
Y_ = []
for n, i in enumerate(sorted_files):
    temp = rio.open(i)
    Y_.append(temp.read())
    temp.close()
    print(str(n+1)+"/"+str(len(sorted_files))+": "+sorted_files[n], end="\r")
Y_ = np.concatenate(Y_)

# load original images
directory = 'testing'
files = [directory + "/" + f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.startswith('raster')]
def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else float('inf')

sorted_files = sorted(files, key=extract_number)
X = []
for n, i in enumerate(sorted_files):
    temp = rio.open(i)
    X.append(temp.read())
    temp.close()
    print(str(n+1)+"/"+str(len(sorted_files))+": "+sorted_files[n], end="\r")
X = np.concatenate(X)

# de-standardize data
std = np.std(X)
mean = np.mean(X)
Y = (Y_ * std) + mean

# delete temp variables
del Y_
del mean
del std
del files
del sorted_files
del n
del i
del directory

@njit
def apply_along_axis_0(func1d, arr):
    """Like calling func1d(arr, axis=0)"""
    if arr.size == 0:
        raise RuntimeError("Must have arr.size > 0")
    ndim = arr.ndim
    if ndim == 0:
        raise RuntimeError("Must have ndim > 0")
    elif 1 == ndim:
        return func1d(arr)
    else:
        result_shape = arr.shape[1:]
        out = np.empty(result_shape, arr.dtype)
        _apply_along_axis_0(func1d, arr, out)
        return out


@njit
def _apply_along_axis_0(func1d, arr, out):
    """Like calling func1d(arr, axis=0, out=out). Require arr to be 2d or bigger."""
    ndim = arr.ndim
    if ndim < 2:
        raise RuntimeError("_apply_along_axis_0 requires 2d array or bigger")
    elif ndim == 2:  # 2-dimensional case
        for i in range(len(out)):
            out[i] = func1d(arr[:, i])
    else:  # higher dimensional case
        for i, out_slice in enumerate(out):
            _apply_along_axis_0(func1d, arr[:, i], out_slice)


@njit
def nb_mean_axis_0(arr):
    return apply_along_axis_0(np.mean, arr)


# ECS definition
@njit
def ECS(x, smooth_x=None):
    
    assert len(x.shape) == 3, "'x' is not three-dimensional"
    mean_image = nb_mean_axis_0(x)
    
    if smooth_x is not None:
        assert len(smooth_x.shape) == 3, "'smooth_x' is not three-dimensional"
        assert x.shape == smooth_x.shape, "'x' and 'smooth_x' are different shapes"
        cube = smooth_x.astype(np.float32)
    else:
        cube = x

    R = np.empty(mean_image.shape, np.float32)
    D = np.empty(cube.shape, np.float32)

    dims = mean_image.shape
    
    lin = dims[0]
    col = dims[1]
    
    for i in range(0, cube.shape[0]):
        D[i] = (cube[i] - mean_image)**2
    
    d = D.sum(axis=1).sum(axis=1).flatten()

    for i in range(lin):
        for j in range(col):
            R[i, j] = np.abs(np.corrcoef(d, D[:, i, j])[0][1])
    
    return R

def waveleted(x):
    xwav = denoise_wavelet(
            x, 
            wavelet="haar",
            wavelet_levels=2
    )
    return xwav

def apply_wavelet(x):
    xwav = np.ndarray(x.shape)
    t = xwav.shape[0]
    for i in range(t):
        xwav[i, :, :] = denoise_wavelet(
            x[i, :, :], 
            wavelet="haar", 
            wavelet_levels=2
        )
        print(str(i+1)+"/"+str(t), end="\r")
    return xwav

def segment_otsu(x):
    th = threshold_otsu(x)
    binary = x > th
    binary = binary.astype('uint8')
    return binary

def segment_li(x):
    th = threshold_li(x)
    binary = x > th
    binary = binary.astype('uint8')
    return binary

def segment_metrics(raster, change, nonchange):
    change = gpd.read_file(change)
    nonchange = gpd.read_file(nonchange)
    res = rio.open(raster)
    
    change_mask, _ = mask(res, change.geometry, crop=True, nodata=99999.0)
    nonchange_mask, _ = mask(res, nonchange.geometry, crop=True, nodata=99999.0)
    res.close()
    
    true_positive = (change_mask == 1).sum()
    false_negative = (change_mask == 0).sum()
    false_positive = (nonchange_mask == 1).sum()
    true_negative = (nonchange_mask == 0).sum()
    
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1 = (2 * precision * recall)/(precision + recall)
    accuracy = (true_positive + true_negative)/(true_negative + true_positive + false_negative + false_positive)
    
    return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}
    
Xwav = apply_wavelet(X)
Ywav = apply_wavelet(Y)
print("Wavelets done.")

ecs = ECS(X)
wecs = ECS(X, Xwav)
tecs = ECS(X, Y)
twecs = ECS(X, Ywav)
print("ECS done.")

# ou
'''
with rio.open('assets/ecs.tif') as src:
    ecs = src.read(1)
with rio.open('assets/wecs.tif') as src:
    wecs = src.read(1)
with rio.open('assets/tecs.tif') as src:
    tecs = src.read(1)
with rio.open('assets/twecs.tif') as src:
    twecs = src.read(1)
'''

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(ecs, cmap="gray")
ax[0, 1].imshow(wecs, cmap="gray")
ax[1, 0].imshow(tecs, cmap="gray")
ax[1, 1].imshow(twecs, cmap="gray")
ax[0, 0].set_title("ECS")
ax[0, 1].set_title("WECS")
ax[1, 0].set_title("TECS")
ax[1, 1].set_title("TWECS")
fig.tight_layout()
plt.show()

del X
del Y
del Xwav
del Ywav

with rio.open(
    'assets/ecs.tif',
    'w',
    driver='GTiff',
    height=ecs.shape[0],
    width=ecs.shape[1],
    count=1,
    dtype=ecs.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(ecs, 1)
    
with rio.open(
    'assets/wecs.tif',
    'w',
    driver='GTiff',
    height=wecs.shape[0],
    width=wecs.shape[1],
    count=1,
    dtype=wecs.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(wecs, 1)
    
with rio.open(
    'assets/tecs.tif',
    'w',
    driver='GTiff',
    height=tecs.shape[0],
    width=tecs.shape[1],
    count=1,
    dtype=tecs.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(tecs, 1)
    
with rio.open(
    'assets/twecs.tif',
    'w',
    driver='GTiff',
    height=twecs.shape[0],
    width=twecs.shape[1],
    count=1,
    dtype=twecs.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(twecs, 1)

bin_ecs = segment_li(ecs)
bin_wecs = segment_li(wecs)
bin_tecs = segment_li(tecs)
bin_twecs = segment_li(twecs)

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(bin_ecs, cmap="gray")
ax[0, 1].imshow(bin_wecs, cmap="gray")
ax[1, 0].imshow(bin_tecs, cmap="gray")
ax[1, 1].imshow(bin_twecs, cmap="gray")
ax[0, 0].set_title("ECS")
ax[0, 1].set_title("WECS")
ax[1, 0].set_title("TECS")
ax[1, 1].set_title("TWECS")
ax[0, 0].axis('off')
ax[0, 1].axis('off')
ax[1, 0].axis('off')
ax[1, 1].axis('off')
fig.tight_layout()
plt.show()

with rio.open(
    'assets/bin_ecs.tif',
    'w',
    driver='GTiff',
    height=bin_ecs.shape[0],
    width=bin_ecs.shape[1],
    count=1,
    dtype=bin_ecs.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(bin_ecs, 1)
    
with rio.open(
    'assets/bin_wecs.tif',
    'w',
    driver='GTiff',
    height=bin_wecs.shape[0],
    width=bin_wecs.shape[1],
    count=1,
    dtype=bin_wecs.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(bin_wecs, 1)
    
with rio.open(
    'assets/bin_tecs.tif',
    'w',
    driver='GTiff',
    height=bin_tecs.shape[0],
    width=bin_tecs.shape[1],
    count=1,
    dtype=bin_tecs.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(bin_tecs, 1)
    
with rio.open(
    'assets/bin_twecs.tif',
    'w',
    driver='GTiff',
    height=bin_twecs.shape[0],
    width=bin_twecs.shape[1],
    count=1,
    dtype=bin_twecs.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(bin_twecs, 1)
    
metric_ecs = segment_metrics('assets/bin_ecs.tif', "shp/Change.shp", "shp/NonChange.shp")
metric_tecs = segment_metrics('assets/bin_tecs.tif', "shp/Change.shp", "shp/NonChange.shp")
metric_wecs = segment_metrics('assets/bin_wecs.tif', "shp/Change.shp", "shp/NonChange.shp")
metric_twecs = segment_metrics('assets/bin_twecs.tif', "shp/Change.shp", "shp/NonChange.shp")

bin_tecsw = segment_li(waveleted(tecs))
bin_wecsw = segment_li(waveleted(wecs))

with rio.open(
    'assets/bin_tecsw.tif',
    'w',
    driver='GTiff',
    height=bin_tecsw.shape[0],
    width=bin_tecsw.shape[1],
    count=1,
    dtype=bin_tecsw.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(bin_tecsw, 1)
    
with rio.open(
    'assets/bin_wecsw.tif',
    'w',
    driver='GTiff',
    height=bin_wecsw.shape[0],
    width=bin_wecsw.shape[1],
    count=1,
    dtype=bin_wecsw.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(bin_wecsw, 1)

metric_wecsw = segment_metrics('assets/bin_wecsw.tif', "shp/Change.shp", "shp/NonChange.shp")
metric_tecsw = segment_metrics('assets/bin_tecsw.tif', "shp/Change.shp", "shp/NonChange.shp")

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(bin_wecs, cmap="gray")
ax[0, 1].imshow(bin_twecs, cmap="gray")
ax[1, 0].imshow(bin_tecsw, cmap="gray")
ax[1, 1].imshow(bin_wecsw, cmap="gray")
ax[0, 0].set_title("WECS")
ax[0, 1].set_title("TWECS")
ax[1, 0].set_title("TECSW")
ax[1, 1].set_title("WECSW")
fig.tight_layout()
plt.show()
