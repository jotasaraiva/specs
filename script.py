import numpy as np
import matplotlib.pyplot as plt
import os
import re
from skimage.restoration import denoise_wavelet, denoise_tv_chambolle
import pywt
import rasterio as rio
from pkg import utils

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

# total variation denoising
test = X[0,:,:-3]
test_tv = denoise_tv_chambolle(test, weight=3)
test_wav = pywt.swt2(test, wavelet='haar', level=2, start_level=0)
fig, ax = plt.subplots(1, 3)
ax[0].imshow(test, cmap="gray")
ax[1].imshow(test_tv, cmap="gray")
ax[2].imshow(test_wav[0][0], cmap="gray")
ax[0].set_title("Original")
ax[1].set_title("TV")
ax[2].set_title("Wavelet")
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
fig.tight_layout()
plt.show()

del test;
del test_wav
del test_tv
del ax
del fig

Xwav = utils.apply_wavelet(X)
Xtv = utils.apply_tv(X)
Ytv = utils.apply_tv(Y)
print("Filtering done.")

ecs = utils.ECS(X)
wecs = utils.ECS(X, Xwav)
tvecs = utils.ECS(X, Xtv)
specs = utils.ECS(X, Y)
stvecs = utils.ECS(X, Ytv)
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

del X
del Y
del Xwav
del Xtv
del Ytv

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(stvecs, cmap="gray")
ax[0, 1].imshow(wecs, cmap="gray")
ax[1, 0].imshow(tvecs, cmap="gray")
ax[1, 1].imshow(specs, cmap="gray")
ax[0, 0].set_title("STVECS")
ax[0, 1].set_title("WECS")
ax[1, 0].set_title("TVECS")
ax[1, 1].set_title("SPECS")
ax[0, 0].axis('off')
ax[0, 1].axis('off')
ax[1, 0].axis('off')
ax[1, 1].axis('off')
fig.tight_layout()
plt.show()

utils.save_with_rio('assets/ecs.tif', ecs, temp)
utils.save_with_rio('assets/wecs.tif', wecs, temp)
utils.save_with_rio('assets/specs.tif', specs, temp)
utils.save_with_rio('assets/tvecs.tif', tvecs, temp)
utils.save_with_rio('assets/stvecs.tif', stvecs, temp)

bin_ecs = utils.segment_otsu(ecs)
bin_wecs = utils.segment_otsu(wecs)
bin_tvecs = utils.segment_otsu(tvecs)
bin_specs = utils.segment_otsu(specs)
bin_stvecs = utils.segment_otsu(stvecs)

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(bin_stvecs, cmap="gray")
ax[0, 1].imshow(bin_wecs, cmap="gray")
ax[1, 0].imshow(bin_tvecs, cmap="gray")
ax[1, 1].imshow(bin_specs, cmap="gray")
ax[0, 0].set_title("STVECS")
ax[0, 1].set_title("WECS")
ax[1, 0].set_title("TVECS")
ax[1, 1].set_title("SPECS")
ax[0, 0].axis('off')
ax[0, 1].axis('off')
ax[1, 0].axis('off')
ax[1, 1].axis('off')
fig.tight_layout()
plt.show()

utils.save_with_rio('assets/bin_ecs.tif', bin_ecs, temp)
utils.save_with_rio('assets/bin_wecs.tif', bin_wecs, temp)
utils.save_with_rio('assets/bin_specs.tif', bin_specs, temp)
utils.save_with_rio('assets/bin_tvecs.tif', bin_tvecs, temp)
utils.save_with_rio('assets/bin_stvecs.tif', bin_stvecs, temp)
    
metric_ecs, ecs_change, ecs_nonchange = utils.segment_metrics('assets/bin_ecs.tif', "shp/Change.shp", "shp/NonChange.shp")
metric_tvecs, tvecs_change, tvecs_nonchange = utils.segment_metrics('assets/bin_tvecs.tif', "shp/Change.shp", "shp/NonChange.shp")
metric_wecs, wecs_change, wecs_nonchange = utils.segment_metrics('assets/bin_wecs.tif', "shp/Change.shp", "shp/NonChange.shp")
metric_specs, specs_change, specs_nonchange = utils.segment_metrics('assets/bin_specs.tif', "shp/Change.shp", "shp/NonChange.shp")
metric_stvecs, stvecs_change, stvecs_nonchange = utils.segment_metrics('assets/bin_stvecs.tif', "shp/Change.shp", "shp/NonChange.shp")
