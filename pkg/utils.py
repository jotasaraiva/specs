import rasterio as rio
import numpy as np
from skimage.filters import threshold_otsu, threshold_li
from skimage.restoration import denoise_wavelet, denoise_tv_chambolle
import geopandas as gpd
from numba import njit
from rasterio.mask import mask

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