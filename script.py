import numpy as np
import matplotlib.pyplot as plt
import os
import re
from skimage.filters import threshold_otsu, threshold_li
from skimage.restoration import denoise_wavelet, denoise_tv_chambolle
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

def apply_tv(x):
    xtv = np.ndarray(x.shape)
    t = xtv.shape[0]
    for i in range(t):
        xtv[i, :, :] = denoise_tv_chambolle(
            x[i, :, :],
            weight=2
        )
        print(str(i+1)+"/"+str(t), end="\r")
    return xtv
        
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
    
    change_mask, _ = mask(res, change.geometry, crop=True, nodata=2)
    nonchange_mask, _ = mask(res, nonchange.geometry, crop=True, nodata=2)
    res.close()
    
    true_positive = (change_mask == 1).sum()
    false_negative = (change_mask == 0).sum()
    false_positive = (nonchange_mask == 1).sum()
    true_negative = (nonchange_mask == 0).sum()
    
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1 = (2 * precision * recall)/(precision + recall)
    accuracy = (true_positive + true_negative)/(true_negative + true_positive + false_negative + false_positive)
    
    return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}, change_mask, nonchange_mask
    
def save_with_rio(path, img, template):
    with rio.open(
         path,
         'w',
         driver='GTiff',
         height=img.shape[0],
         width=img.shape[1],
         count=1,
         dtype=img.dtype,
         crs='+proj=latlong',
         transform=template.transform
    ) as dst:
        dst.write(img, 1)
        
    return True

# total variation denoising
test = denoise_tv_chambolle(X[0], weight=2)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(X[0], cmap="gray")
ax[1].imshow(test, cmap="gray")
ax[0].set_title("Original")
ax[1].set_title("TV")
fig.tight_layout()
plt.show()

# wavelet denoising
test = denoise_wavelet(X[0], wavelet='haar', wavelet_levels=2)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(X[0], cmap="gray")
ax[1].imshow(test, cmap="gray")
ax[0].set_title("Original")
ax[1].set_title("Wavelet")
fig.tight_layout()
plt.show()

del test
del ax
del fig

Xwav = apply_wavelet(X)
Xtv = apply_tv(X)
#Ywav = apply_wavelet(Y)
print("Filtering done.")

ecs = ECS(X)
wecs = ECS(X, Xwav)
tvecs = ECS(X, Xtv)
specs = ECS(X, Y)
print("ECS done.")

# ou
'''
with rio.open('assets/ecs.tif') as src:
    ecs = src.read(1)
with rio.open('assets/wecs.tif') as src:
    wecs = src.read(1)
with rio.open('assets/tvecs.tif') as src:
    tvecs = src.read(1)
with rio.open('assets/specs.tif') as src:
    specs = src.read(1)
'''

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(ecs, cmap="gray")
ax[0, 1].imshow(wecs, cmap="gray")
ax[1, 0].imshow(tvecs, cmap="gray")
ax[1, 1].imshow(specs, cmap="gray")
ax[0, 0].set_title("ECS")
ax[0, 1].set_title("WECS")
ax[1, 0].set_title("TVECS")
ax[1, 1].set_title("SPECS")
fig.tight_layout()
plt.show()

del X
del Y
del Xwav
del Xtv

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
    'assets/tvecs.tif',
    'w',
    driver='GTiff',
    height=tvecs.shape[0],
    width=tvecs.shape[1],
    count=1,
    dtype=tvecs.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(tvecs, 1)
    
with rio.open(
    'assets/specs.tif',
    'w',
    driver='GTiff',
    height=specs.shape[0],
    width=specs.shape[1],
    count=1,
    dtype=specs.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(specs, 1)

bin_ecs = segment_otsu(ecs)
bin_wecs = segment_otsu(wecs)
bin_tvecs = segment_otsu(tvecs)
bin_specs = segment_otsu(specs)

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(bin_ecs, cmap="gray")
ax[0, 1].imshow(bin_wecs, cmap="gray")
ax[1, 0].imshow(bin_tvecs, cmap="gray")
ax[1, 1].imshow(bin_specs, cmap="gray")
ax[0, 0].set_title("ECS")
ax[0, 1].set_title("WECS")
ax[1, 0].set_title("TVECS")
ax[1, 1].set_title("SPECS")
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
    'assets/bin_tvecs.tif',
    'w',
    driver='GTiff',
    height=bin_tvecs.shape[0],
    width=bin_tvecs.shape[1],
    count=1,
    dtype=bin_tvecs.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(bin_tvecs, 1)
    
with rio.open(
    'assets/bin_specs.tif',
    'w',
    driver='GTiff',
    height=bin_specs.shape[0],
    width=bin_specs.shape[1],
    count=1,
    dtype=bin_specs.dtype,
    crs='+proj=latlong',
    transform=temp.transform
) as dst:
    dst.write(bin_specs, 1)
    
metric_ecs, ecs_change, ecs_nonchange = segment_metrics('assets/bin_ecs.tif', "shp/Change.shp", "shp/NonChange.shp")
metric_tvecs, tvecs_change, tvecs_nonchange = segment_metrics('assets/bin_tvecs.tif', "shp/Change.shp", "shp/NonChange.shp")
metric_wecs, wecs_change, wecs_nonchange = segment_metrics('assets/bin_wecs.tif', "shp/Change.shp", "shp/NonChange.shp")
metric_specs, specs_change, specs_nonchange = segment_metrics('assets/bin_specs.tif', "shp/Change.shp", "shp/NonChange.shp")
