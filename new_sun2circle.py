import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import math
import argparse

def Resize(img, target_w, target_h):
    dim = (target_w, target_h)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

def gaussian_circle(center, radius, sigma):
    x, y = np.meshgrid(np.arange(center[0]-radius, center[0]+radius+1),
                       np.arange(center[1]-radius, center[1]+radius+1))
    d = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    circle = np.exp(-(d**2 / (2 * sigma**2)))
    return circle*255

def scale_mask(original_matrix, new_min = 1, new_max = 2000):
    original_min = np.min(original_matrix)
    original_max = np.max(original_matrix)
    return (original_matrix - original_min) / (original_max - original_min) * (new_max - new_min) + new_min

def Spher2Equ(theta, phi, building_orient):
    theta = (building_orient - theta) * np.pi / 180
    phi = phi * np.pi / 180

    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi)
    z = r * np.cos(phi) * np.cos(theta)

    x_theta = np.arctan(x/z)
    y_phi = np.arcsin(y/math.sqrt((x**2 + y**2 + z**2)))

    return x_theta, y_phi

def Square2Equrec_BW(img):
    h, w = img.shape[:2]
    rect_canvas = np.zeros((h, w * 2), dtype=np.uint8)
    equ_img = rect_canvas.copy()
    equ_img[0:h, int(w * 0.5): int(w * 1.5)] = img
    return equ_img

def Draw_Cir(img, coord, radius = 100, color = (0, 0, 255), thickness = 50):
    return cv.circle(img, coord, radius, color, thickness)

def Equ2Square(img):
    h, w = img.shape[:2]
    return img[0:h, int(w * 0.25): int(w * 0.75)]


def main():
    parser = argparse.ArgumentParser(description='Process canvas and sun circle parameters.')

    # Canvas dimensions
    parser.add_argument('--canvas_h', type=int, default=2048, help='Height of the base canvas')
    parser.add_argument('--canvas_w', type=int, default=None, help='Width of the base canvas. Defaults to twice the height.')

    # Export path
    parser.add_argument('--export_path', type=str, default='../inversed_fisheye2equirect/', help='Path for exporting the result')

    # Sun circle parameters
    parser.add_argument('--center_x', type=int, default=400, help='X-coordinate of the sun circle center')
    parser.add_argument('--center_y', type=int, default=400, help='Y-coordinate of the sun circle center')
    parser.add_argument('--radius', type=int, default=200, help='Radius of the sun circle')
    parser.add_argument('--sigma', type=int, default=25, help='Standard deviation for Gaussian function')

    # Additional parameters
    parser.add_argument('--adj_ang', type=float, default=-5, help='Adjustment angle')
    parser.add_argument('--theta', type=float, default=194.07, help='Theta value of the sun')
    parser.add_argument('--phi', type=float, default=40.31, help='Phi value of the sun')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = main()

    # Use the arguments
    canvas_h = args.canvas_h
    canvas_w = args.canvas_w if args.canvas_w is not None else canvas_h * 2
    export_path = args.export_path
    center = (args.center_x, args.center_y)
    radius = args.radius
    sigma = args.sigma


    """
    00: Generate the Gaussian circle
    """
    circle = gaussian_circle(center, radius, sigma)
    circle = Resize(circle, center[0], center[1])
    circle = cv.circle(circle, (center[0]//2, center[1]//2), radius//30, 255, -1)
    cv.imwrite('gaussian circle.png', circle)

    h = canvas_h
    w = canvas_w
    r = h // 2


    """
    01: sun position on the equi-rectanglar coordinate
    """
    adj_ang = args.adj_ang
    theta = args.theta
    phi = args.phi
    building_orient = 264 + adj_ang

    x_theta, y_phi = Spher2Equ(theta, phi, building_orient)
    dx = w // 2 - x_theta / (math.pi) * w // 2
    dy = h // 2 - y_phi / (math.pi / 2) * h // 2
    print('dx, dy:', dx, dy)


    """
    02: draw circle for inpainting
    """
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    rect_img_ = canvas.copy()
    outlined = Draw_Cir(rect_img_, (int(dx), int(dy)), radius=150, color=(255, 255, 255), thickness=-1)
    outlined = Equ2Square(outlined)
    cv.imwrite(export_path + 'sun_mask2.png', outlined)


    """
    03: save gaussian circle
    """
    canvas[int(dy)-center[0]//2:int(dy)+center[0]//2,
           int(dx)-center[1]//2:int(dx)+center[0]//2] = circle

    print('int(dx)-center[1]//2, int(dx)+center[0]//2:', int(dx)-center[1]//2, int(dx)+center[0]//2)
    cv.imwrite(export_path + 'sun_mask2canvas.png', canvas)
