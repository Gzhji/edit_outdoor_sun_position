import numpy as np
import cv2
import math
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import os
import argparse


# Example usage:
xf = 0.5
yf = 0.5
pan = 0
#tkey = 0 # equidistant

# # %% circular fisheye projection transformation code:
# # % 0: Equisolid-angle
# # % 1: Equidistant
# # % 2: Orthographic
# # % 3: Stereographic
# trans_from = 1; #% source proj code
# trans_to = 1; #% target proj code


def xyzrotate(xyz, thetaXYZ):
    tX = np.radians(thetaXYZ[0])
    tY = np.radians(thetaXYZ[1])
    tZ = np.radians(thetaXYZ[2])

    Tm = np.array([[np.cos(tY)*np.cos(tZ), -np.cos(tY)*np.sin(tZ), np.sin(tY)],
                   [np.cos(tX)*np.sin(tZ) + np.cos(tZ)*np.sin(tX)*np.sin(tY),
                    np.cos(tX)*np.cos(tZ) - np.sin(tX)*np.sin(tY)*np.sin(tZ),
                    -np.cos(tY)*np.sin(tX)],
                   [np.sin(tX)*np.sin(tZ) - np.cos(tX)*np.cos(tZ)*np.sin(tY),
                    np.cos(tZ)*np.sin(tX) + np.cos(tX)*np.sin(tY)*np.sin(tZ),
                    np.cos(tX)*np.cos(tY)]])

    xyznew = np.dot(xyz, Tm)
    return xyznew


def fish2equ(xf, yf, roll, tilt, pan, fov, tkey):
    thetaS = np.rad2deg(np.arctan2(yf, xf))

    if tkey == 1:
        phiS = np.sqrt(yf ** 2 + xf ** 2) * fov / 2  # equidistant
    elif tkey == 0:
        phiS = 2 * np.rad2deg(
            np.arcsin(np.sqrt(yf ** 2 + xf ** 2) * np.sin(np.deg2rad(fov / 4))))  # equisolidangle proj
    elif tkey == 2:
        phiS = np.rad2deg(np.arcsin(np.sqrt(yf ** 2 + xf ** 2) * np.sin(np.deg2rad(fov / 2))))  # orthographic proj
    elif tkey == 3:
        phiS = 2 * np.rad2deg(np.arctan(np.sqrt(yf ** 2 + xf ** 2) * np.tan(np.deg2rad(fov / 4))))  # Stereographic proj

    sindphiS = np.sin(np.deg2rad(phiS))
    xs = sindphiS * np.cos(np.deg2rad(thetaS))
    ys = sindphiS * np.sin(np.deg2rad(thetaS))
    zs = np.cos(np.deg2rad(phiS))

    xyzsz = xs.shape
    xyz = xyzrotate(np.column_stack((xs.flatten(), ys.flatten(), zs.flatten())), [roll, tilt, pan])

    xs = np.reshape(xyz[:, 0], xyzsz)
    ys = np.reshape(xyz[:, 1], xyzsz)
    zs = np.reshape(xyz[:, 2], xyzsz)

    thetaE = np.rad2deg(np.arctan2(xs, zs))
    phiE = np.rad2deg(np.arctan2(ys, np.sqrt(xs ** 2 + zs ** 2)))

    xe = thetaE / 180
    ye = 2 * phiE / 180

    return xe, ye


def equ2fish(xe, ye, fov, roll, tilt, pan, tkey):
    thetaE = xe * 180
    phiE = ye * 90
    cosdphiE = np.cos(np.deg2rad(phiE))
    xs = cosdphiE * np.cos(np.deg2rad(thetaE))
    ys = cosdphiE * np.sin(np.deg2rad(thetaE))
    zs = np.sin(np.deg2rad(phiE))

    xyzsz = xs.shape
    xyz = xyzrotate(np.column_stack((xs.flatten(), ys.flatten(), zs.flatten())), [roll, tilt, pan])

    xs = np.reshape(xyz[:, 0], xyzsz)
    ys = np.reshape(xyz[:, 1], xyzsz)
    zs = np.reshape(xyz[:, 2], xyzsz)

    thetaF = np.rad2deg(np.arctan2(zs, ys))

    if tkey == 1:
        r = 2 * np.rad2deg(np.arctan2(np.sqrt(ys**2 + zs**2), xs)) / fov
    elif tkey == 0:
        r = np.sin(np.arctan2(np.sqrt(ys**2 + zs**2), xs) / 2) / np.sin(np.deg2rad(fov / 4))
    elif tkey == 2:
        r = np.sin(np.arctan2(np.sqrt(ys**2 + zs**2), xs)) / np.sin(np.deg2rad(fov / 2))
    elif tkey == 3:
        r = np.tan(np.arctan2(np.sqrt(ys**2 + zs**2), xs) / 2) / np.tan(np.deg2rad(fov / 4))

    xf = r * np.cos(np.deg2rad(thetaF))
    yf = r * np.sin(np.deg2rad(thetaF))

    return xf, yf


def imequ2fish(imgE, tkey, fov=180, roll=0, tilt=0, pan=0, w_ratio =1):
    he, we, ch = imgE.shape
    wf = round(we / w_ratio)
    hf = he

    xf, yf = np.meshgrid(np.linspace(1, wf, wf), np.linspace(1, hf, hf))
    xf = 2 * ((xf - 1) / (wf - 1) - 0.5)
    yf = 2 * ((yf - 1) / (hf - 1) - 0.5)
    idx = np.sqrt(xf**2 + yf**2) <= 1
    xf = xf[idx]
    yf = yf[idx]

    xe, ye = fish2equ(xf, yf, roll, tilt, pan, fov, tkey)

    Xe = np.round((xe + 1) / 2 * (we - 1)).astype(np.uint32)
    Ye = np.round((ye + 1) / 2 * (he - 1)).astype(np.uint32)
    Xf = np.round((xf + 1) / 2 * (wf - 1)).astype(np.uint32)
    Yf = np.round((yf + 1) / 2 * (hf - 1)).astype(np.uint32)

    If = np.zeros((hf * wf, ch), dtype='float32')
    Ie = imgE.reshape(-1, ch)

    idnf = np.ravel_multi_index((Yf - 1, Xf - 1), (hf, wf), mode='clip')
    idne = np.ravel_multi_index((Ye - 1, Xe - 1), (he, we), mode='clip')

    If[idnf, :] = Ie[idne, :]
    imgF = If.reshape(hf, wf, ch)

    return imgF

def imfish2equ(imgF, tkey, fov=180, roll=0, tilt=0, pan=0, w_ratio = 2):
    hf, wf, ch = imgF.shape
    we = wf * w_ratio
    he = hf

    xe, ye = np.meshgrid(np.arange(1, we+1), np.arange(1, he+1))
    xe = 2 * ((xe - 1) / (we - 1) - 0.5)
    ye = 2 * ((ye - 1) / (he - 1) - 0.5)

    xf, yf = equ2fish(xe, ye, fov, roll, tilt, pan, tkey)
    idx = np.sqrt(xf**2 + yf**2) <= 1

    xf = xf[idx]
    yf = yf[idx]
    xe = xe[idx]
    ye = ye[idx]

    Xe = np.round((xe + 1) / 2 * (we - 1) + 1).astype(np.uint32)
    Ye = np.round((ye + 1) / 2 * (he - 1) + 1).astype(np.uint32)
    Xf = np.round((xf + 1) / 2 * (wf - 1) + 1).astype(np.uint32)
    Yf = np.round((yf + 1) / 2 * (hf - 1) + 1).astype(np.uint32)

    Ie = imgF.reshape(-1, ch)
    If = np.zeros((he * we, ch), dtype='float32')

    idnf = np.ravel_multi_index((Yf-1, Xf-1), (hf, wf), mode='clip')
    idne = np.ravel_multi_index((Ye-1, Xe-1), (he, we), mode='clip')

    If[idne, :] = Ie[idnf, :]
    imgE = If.reshape(he, we, ch)
    return imgE


def Resiz(img, h, w):
    down_points = (w, h)
    return cv2.resize(img, down_points, interpolation=cv2.INTER_AREA)

def hdr2ldr(hdr):
    print('hdr, min max: ', np.min(hdr), np.max(hdr))
    ldr = hdr/np.max(hdr)
    print(np.max(ldr))
    # Color space conversion
    # 0-255 remapping for bit-depth conversion
    return ldr, np.max(hdr)

def Square2Equrec(img):
    h, w = img.shape[:2]
    rect_canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    equ_img = rect_canvas.copy()
    equ_img[0:h, int(w * 0.5): int(w * 1.5)] = img
    return equ_img

def Crop_Cube(img):
    cols, rows = img.shape[:2]
    midx = cols // 2
    midy = rows // 2

    scale_ratio = rows / 36
    targt_ridus = int(11.6 * scale_ratio) #the number 11.6 is only for cropping fisheye in Cannon 6D sensor dimension

    return img[midx - targt_ridus: midx + targt_ridus, midy - targt_ridus:midy + targt_ridus, :]


def main():
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--task', type=str, default='e2f_batch', choices=['e2f', 'f2e', 'f2e_batch', 'e2f_batch', 'e2f_global'],
                        help='Path to the sun mask image')
    parser.add_argument('--single_img_path', type=str, default='single_img_path', help='single fisheye image')
    parser.add_argument('--pano_img_path', type=str, default='single_img_path', help='single panoramic image')
    parser.add_argument('--input_folder', type=str, default='input_folder_path', help='input image set')
    parser.add_argument('--output_folder', type=str, default='output_folder_path', help='output image set')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = main()
    """
    00: import the files
    """
    if args.task == 'e2f':
        imgE = cv2.imread(args.single_img_path)
        h,w = imgE.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        imgE = imgE[0:h, int(0.25*w): int(0.75*w)]
        canvas[0:h, int(0.25*w): int(0.75*w)] = imgE
        img_r = imequ2fish(canvas, tkey = 0, fov=180, roll=0, tilt=0, pan=0, w_ratio = 2)
        cv2.imwrite('equ2fish.png', img_r)

    if args.task == 'f2e':
        img = cv2.imread(args.single_img_path, -1)
        img_e = imfish2equ(img, tkey = 0, fov=180, roll=0, tilt=0, pan=0, w_ratio = 2)
        h,w = img_e.shape[:2]
        img_e_c = img_e[0:h, int(0.25*w): int(0.75*w)]
        cv2.imwrite('fish2equ.png', img_e_c)


    if args.task == 'f2e_batch':
        ldr_set_path = args.input_folder
        new_img_path = args.output_folder
        filename_list = [a for a in os.listdir(ldr_set_path)]
        filename_list.sort()

        for filename in filename_list:
            img = cv2.imread(ldr_set_path + filename, -1)
            print('ldr_set_path + filename:', ldr_set_path + filename)
            img = Crop_Cube(img)
            img_e = imfish2equ(img, tkey = 1, fov=180, roll=0, tilt=0, pan=0, w_ratio = 2)

            h,w = img_e.shape[:2]
            img_e_c = img_e[0:h, int(0.25*w): int(0.75*w)]
            cv2.imwrite(new_img_path + filename, img_e_c)

    if args.task == 'e2f_batch':
        generated_path = args.input_folder
        generated_fisheye_path = args.output_folder
        filename_list = [a for a in os.listdir(generated_path) if a.endswith('.png')]
        filename_list.sort()

        for filename in filename_list:
            img = cv2.imread(generated_path + filename, -1)
            print('ldr_set_path + filename:', generated_path + filename)
            img_r = imequ2fish(img, tkey=0, fov=180, roll=0, tilt=0, pan=0, w_ratio=2)
            cv2.imwrite(generated_fisheye_path + filename, img_r)

    if args.task == 'e2f_global':
        img = cv2.imread(args.pano_img_path, -1)
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        img = Resiz(img, h = 512, w = 1024)
        img, scale_ratio = hdr2ldr(img)

        h,w = img.shape[:2]
        imgE = img.copy()
        rotate_ang = 45

        for i in range(8):
            ang = (i + 1) * rotate_ang
            dis = w * rotate_ang/360
            imgE = np.roll(imgE, - int(dis), axis=1)

            canvas = np.zeros((h, w, 3), dtype=np.float32)
            canvas[0:h, int(0.25*w): int(0.75*w)] = imgE[0:h, int(0.25*w): int(0.75*w)]
            img_r = imequ2fish(canvas, tkey=0, fov=180, roll=0, tilt=0, pan=0, w_ratio = 2)
            if ang == 360:
                ang = 0
            cv2.imwrite('fisheye_%d.png' % ang, img_r*scale_ratio)
