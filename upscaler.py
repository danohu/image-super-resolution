#%%
import os

from ISR.models import RDN
import numpy as np
import glob
from PIL import Image


def upscale(img):
    lr_img = np.array(img)
    rdn = RDN(weights='psnr-small')
    sr_img = rdn.predict(lr_img)
    hires = Image.fromarray(sr_img)
    return hires


def upscale_wrapper(fn):
    output_path = fn.replace('images', 'scaled_images')
    if os.path.exists(output_path):
        print('skip existing %s' % fn)
        return
    lores = Image.open(fn)
    hires = upscale(lores)
    hires.save(output_path)
    # images.append((lores, hires))
    print('done %s' %  fn)

for pattern in ['images/*jpg', 'images/*png']:
    for fn in glob.glob(pattern):
        upscale_wrapper(fn)
