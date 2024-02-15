import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import argparse

def Luminance_ldr(img, Exposure = 1):
    (B, G, R) = cv.split(img)
    return 179 * (0.2126*R + 0.7152*G + 0.0722*B) / Exposure


def scale_mask(original_matrix, new_min = 1, new_max = 2000):
    # Assuming your original matrix is named 'original_matrix'
    original_min = np.min(original_matrix)
    original_max = np.max(original_matrix)
    return (original_matrix - original_min) / (original_max - original_min) * (new_max - new_min) + new_min


def Resize(img, target_h, target_w):
    dim = (target_h, target_w)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

def Scale_Sun2HDR(orig_hdr, sun_mask):
    target_h, target_w = orig_hdr.shape[:2]
    sun_mask = Resize(sun_mask, target_h, target_h)
    sun_mask_h, sun_mask_w = sun_mask.shape[:2]
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    canvas_h, canvas_w = canvas.shape[:2]
    canvas[0:target_h, canvas_w//2 - sun_mask_w//2:canvas_w//2 + sun_mask_w//2] = sun_mask
    return canvas

def Scale_Sun_Lum(sun_mask, scale_ratio):
    sun_mask = sun_mask/255.
    sun_mask_scaled = sun_mask * scale_ratio
    return np.stack((sun_mask_scaled, sun_mask_scaled, sun_mask_scaled), axis=-1)

def Expand3Channel(binary_img):
    return np.repeat(binary_img[:, :, np.newaxis], 3, axis=-1)


def main():
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--sun_mask_path', type=str, default='../m_equ2fish_0700.png', help='Path to the sun mask image')
    parser.add_argument('--input_hdr', type=str, default='new+penn+bed+07091127.hdr', help='Input HDR file')
    parser.add_argument('--out_hdr', type=str, default='penn+bed+07091127_new_sun_0700.hdr', help='Output HDR file')
    parser.add_argument('--sun_brightness', type=float, default=1.6 * (10 ** 6), help='Sun brightness in cd/m2')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    """
    00: import the files
    """
    args = main()
    sun_mask_path = args.sun_mask_path
    input_hdr = args.input_hdr
    out_hdr = args.out_hdr
    sun_brightness = args.sun_brightness

    """
    01: adjust the hdr shape
    """
    orig_hdr = cv.imread(input_hdr, -1)
    orig_hdr = Resize(orig_hdr, 3600, 3600)
    h, w = orig_hdr.shape[:2]
    sun_mask = cv.imread(sun_mask_path, 0)
    sun_mask = Resize(sun_mask, h, w)


    """
    02: scale sun_mask to hdr
    """
    sun_mask = Scale_Sun2HDR(orig_hdr, sun_mask)
    no_sun_mask = ~sun_mask
    no_sun_mask[no_sun_mask != 0] = 255
    sun_mask_verse = Expand3Channel(no_sun_mask)/255.


    """
    03: adding sun brightness
    """
    lum_map = Luminance_ldr(orig_hdr, Exposure = 1)
    baseline_sun = np.max(lum_map)
    target_sun = sun_brightness


    """
    04: scale the sun region
    """
    scale_ratio = target_sun/baseline_sun
    sun_layer = Scale_Sun_Lum(sun_mask, scale_ratio)
    new_hdr_sun = sun_layer * sun_layer
    new_hdr_sun = np.float32(new_hdr_sun)


    """
    05: save HDR file
    """
    new_hdr_background = np.float32(orig_hdr * sun_mask_verse)
    new_hdr = new_hdr_sun + new_hdr_background
    cv.imwrite(out_hdr, new_hdr)