import cv2 as cv
import numpy as np
import math
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import os
import argparse


def Draw_Cir(img, coord, radius = 100, color = (0, 0, 255), thickness = 50):
    return cv.circle(img, coord, radius, color, thickness)

def Resize(img, target_w, target_h):
    dim = (target_w, target_h)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

def Spher2Equ(theta, phi, building_orient):
    theta = (building_orient - theta) * np.pi / 180
    phi = phi * np.pi / 180

    # get x, y, z in 3D coordinate based on theta and phi angles
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi)
    z = r * np.cos(phi) * np.cos(theta)

    # get theta and phi for the equirectangle representation
    x_theta = np.arctan(x/z)
    y_phi = np.arcsin(y/math.sqrt((x**2 + y**2 + z**2)))
    return x_theta, y_phi

def Square2Equrec(img):
    h, w = img.shape[:2]
    rect_canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    equ_img = rect_canvas.copy()
    equ_img[0:h, int(w * 0.5): int(w * 1.5)] = img
    return equ_img

def Equ2Square(img):
    h, w = img.shape[:2]
    # squ_canvas = np.zeros((h, h, 3), dtype=np.uint8)
    squ_img = img[0:h, int(w * 0.25): int(w * 0.75)]
    return squ_img

def Square2Equrec_BW(img):
    h, w = img.shape[:2]
    rect_canvas = np.zeros((h, w * 2), dtype=np.uint8)
    equ_img = rect_canvas.copy()
    equ_img[0:h, int(w * 0.5): int(w * 1.5)] = img
    return equ_img

def Resize(img, target_w, target_h):
    dim = (target_w, target_h)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

def Read_Sky_Patch(input_sky_area):
    seg_canvas = np.zeros((h, h))
    sky_region = cv.imread(input_sky_area, 0)
    sky_region = Get_Edge(sky_region, line_width=200)

    sky_h, sky_w = sky_region.shape[:2]
    seg_canvas[0:sky_h, 0:sky_w] = sky_region
    seg_layer = seg_canvas.copy()
    seg_layer[seg_layer != 0] = 255
    seg_layer = Square2Equrec_BW(seg_layer)
    return seg_layer

def Seg_building(input_sky_area):
    seg_canvas = np.zeros((h, h))
    sky_region = cv.imread(input_sky_area, 0)
    sky_h, sky_w = sky_region.shape[:2]

    seg_canvas[0:sky_h, 0:sky_w] = sky_region
    seg_layer = seg_canvas.copy()
    seg_layer[seg_layer != 0] = 255
    seg_layer = 255 - seg_layer
    seg_layer = Square2Equrec_BW(seg_layer)
    return seg_layer


def Get_Cont(img, line_width = 60):
    edged = cv.Canny(img, 30, 200)
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    return cv.drawContours(img, contours, -1, 0, line_width)

def Get_Edge(img, line_width):
    edged = cv.Canny(img, 1, 255)
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    img = cv.drawContours(img, contours, -1, (0, 255, 0), line_width)
    return img

def Transform_Equ_Binary(h_dis, w_dis, img):
    #move in horizontal direction
    img_hori = np.roll(img, int(w_dis), axis=1)
    #move in vertical direction
    img_vert = np.roll(img_hori, int(h_dis), axis=0)
    return img_vert

def Transform_Equ(h_dis, w_dis, img):
    b_channel, g_channel, r_channel = cv.split(img)

    #move in horizontal direction
    b_hori = np.roll(b_channel, int(w_dis), axis=1)
    g_hori = np.roll(g_channel, int(w_dis), axis=1)
    r_hori = np.roll(r_channel, int(w_dis), axis=1)

    #move in vertical direction
    b_vert = np.roll(b_hori, int(h_dis), axis=0)
    g_vert = np.roll(g_hori, int(h_dis), axis=0)
    r_vert = np.roll(r_hori, int(h_dis), axis=0)

    return cv.merge((b_vert, g_vert, r_vert))

def Find_Max(img_path):
    input_gray = cv.imread(img_path, 0)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(input_gray)
    return max_loc

def Adj_Build_Orient(img_path, theta, phi, building_orient):
    input_img = cv.imread(img_path, -1)
    h, w = input_img.shape[:2]
    # r = h // 2

    x_theta, y_phi = Spher2Equ(theta, phi, building_orient - 180)
    dx = w // 2 - x_theta / (math.pi) * w // 2
    # dy = h // 2 - y_phi / (math.pi / 2) * h // 2

    max_loc = Find_Max(img_path)
    dx_adj = dx - int(max_loc[0])
    transformed_input = Transform_Equ(0, dx_adj, input_img)
    return transformed_input

def Expand_3Channel(gray):
    return np.stack((gray, gray, gray), axis=-1)


def main():
    parser = argparse.ArgumentParser(description='Process file paths and image processing parameters.')

    # Paths
    parser.add_argument('--ldr_set_path', type=str, default='ldr_set/captured_equ/', help='Path to the LDR set directory')
    # parser.add_argument('--new_img_path', type=str, default='ldr_set/generated/', help='Path to save new images')
    # parser.add_argument('--lama_img_path', type=str, default='lama_new_sky/', help='Path for LAMA processed images')
    parser.add_argument('--input_sky_area', type=str, default='input/coda334_202307061254_sky_region.png', help='Input sky area image path')
    parser.add_argument('--input_sky_path', type=str, default='../sky_color/time/coda334_20230706_t=8/coda334_20230706_1057_ab_02_02_02_t=8.png', help='Input sky path')
    parser.add_argument('--targt_sky_path', type=str, default='../sky_color/time/coda334_20230706_t=8/coda334_20230706_0857_ab_02_02_02_t=8.png', help='Target sky path')

    # Parameters
    parser.add_argument('--Sky_Color', type=bool, default=True, help='Flag to adjust sky color')
    parser.add_argument('--Bld_Color', type=bool, default=True, help='Flag to adjust building color')
    parser.add_argument('--tone_alpha', type=float, default=0.8, help='alpha to control color edit')
    parser.add_argument('--current_theta', type=float, default=108.54, help='current sun theta')
    parser.add_argument('--current_phi', type=float, default=54.28, help='current sun phi')
    parser.add_argument('--target_theta', type=float, default=86.07, help='target sun theta')
    parser.add_argument('--target_phi', type=float, default=31.76, help='target sun phi')
    parser.add_argument('--adj_ang', type=float, default=0, help='additonal adjust ang for theta')

    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = main()



    # import the equirectangler image
    filename_list = [a for a in os.listdir(args.ldr_set_path)]
    filename_list.sort()
    for filename in filename_list:
        print('filename:', filename)
        rec_fisheye = cv.imread(args.ldr_set_path + filename, -1)
        rect_img = Square2Equrec(rec_fisheye)
        scene_context = rect_img.copy()

        #get img dimension
        h, w = rect_img.shape[:2]
        r = h//2

        """
        01: compute the current sun position on image 
        """
        building_orient = 167 + args.adj_ang
        x_theta, y_phi = Spher2Equ(args.current_theta, args.current_phi, building_orient)
        dx = w//2 - x_theta/(math.pi) * w//2
        dy = h//2 - y_phi/(math.pi/2) * h//2

        """
        02: get sky patch and context layer
        """
        #get lens mask due to light loss at the edge
        lens_mask = cv.imread('input/mask.png', 0)
        lens_mask = Square2Equrec_BW(lens_mask)
        lens_mask = Resize(lens_mask, w, h)

        #read segmentation layer to seg img into sky and non-sky regions
        seg_layer = Read_Sky_Patch(args.input_sky_area)
        building_layer = Seg_building(args.input_sky_area)
        cv.imwrite('01_sky_layer.png', seg_layer)
        cv.imwrite('01_building_layer.png', building_layer)

        seg_layer[lens_mask == 255] = 0
        sky_patch = scene_context*(Expand_3Channel(seg_layer)/255)
        build_patch = scene_context*(Expand_3Channel(building_layer)/255)

        """
        03: compute the target sun position on image 
        """
        #convert target theta and phi angled into x, y coordinates in the image
        x_theta_new, y_phi_new = Spher2Equ(args.target_theta, args.target_phi, building_orient)
        dx_new = w//2 - x_theta_new/(math.pi) * w//2
        dy_new = h//2 - y_phi_new/(math.pi/2) * h//2

        #get x, y distance on the image
        h_dis = dy_new - dy
        w_dis = dx_new - dx

        #outline the sun position on the image
        outlined = Draw_Cir(scene_context, (int(dx), int(dy)), radius=100, color=(0, 0, 255), thickness=50)
        outlined = Draw_Cir(scene_context, (int(dx_new), int(dy_new)), radius=100, color=(255, 0, 0), thickness=50)
        cv.imwrite('01_outlined_ori.png', outlined)


        """
        04: transfer sky appearance from full spectral sky image
        """
        if args.Sky_Color  == True:

            #adjust orientation to the actual room orientation
            rendered_sky = Adj_Build_Orient(args.input_sky_path, args.current_theta, args.current_phi, building_orient)
            rendered_sky = Resize(rendered_sky, w, h)

            # outline the sun in the rendered image
            outlined_rend = Draw_Cir(rendered_sky.copy(), (int(dx), int(dy)), radius=100, color=(0, 0, 255), thickness=50)
            outlined_rend = Draw_Cir(outlined_rend, (int(dx_new), int(dy_new)), radius=100, color=(255, 0, 0), thickness=50)
            cv.imwrite('01_outlined_rend_sky.png', outlined_rend)

            # add color change to the target sky
            sky_patch = sky_patch/(rendered_sky*args.tone_alpha)
            targt_sky = Adj_Build_Orient(args.targt_sky_path, args.target_theta, args.target_phi, building_orient)
            targt_sky = Resize(targt_sky, w, h)

            # outline the sun in the target image
            outlined_targ = Draw_Cir(targt_sky.copy(), (int(dx), int(dy)), radius=100, color=(0, 0, 255), thickness=50)
            outlined_targ = Draw_Cir(outlined_targ, (int(dx_new), int(dy_new)), radius=100, color=(255, 0, 0), thickness=50)
            cv.imwrite('01_outlined_targ_sky.png', outlined_targ)

            # apply alpha to control intensity
            new_sky_patch = Transform_Equ(h_dis, w_dis, sky_patch)
            transformed_sky = new_sky_patch*(targt_sky*args.tone_alpha)

        else:
            transformed_sky = Transform_Equ(h_dis, w_dis, sky_patch)

        """
        05: transfer ground appearance from full spectral sky image
        """
        if args.Bld_Color == True:

            # read the current ground color from the rendered image
            input_grd_color = cv.imread(args.input_sky_path, -1)
            input_grd_color = Resize(input_grd_color, w, h)
            input_grd_color = input_grd_color[h // 2:h, 0:w]
            input_grd_color = Resize(input_grd_color, w, h)

            # read the target ground color from the rendered image
            targt_grd_color = cv.imread(args.targt_sky_path, -1)
            targt_grd_color = Resize(targt_grd_color, w, h)
            targt_grd_color = targt_grd_color[h // 2:h, 0:w]
            targt_grd_color = Resize(targt_grd_color, w, h)

            build_patch = build_patch / input_grd_color
            build_patch = build_patch * targt_grd_color


        """
        06: save the new images
        """
        #save transflated image
        transformed_sky[building_layer == 255] = np.array([0, 0, 0])
        new_scene = transformed_sky + build_patch
        new_scene = Resize(new_scene, 1024, 512)
        cv.imwrite('new_' + filename[:-4] + '.png', new_scene)

        #save paired mask
        transformed_sky_mask = Transform_Equ(h_dis, w_dis, Expand_3Channel(seg_layer))
        transformed_sky_mask[building_layer == 255] = np.array([255, 255, 255])
        transformed_sky_mask = 255 - transformed_sky_mask
        transformed_sky_mask = Resize(transformed_sky_mask, 1024, 512)
        cv.imwrite('new_' + filename[:-4] + '_mask.png', transformed_sky_mask)