import os
from IPython import embed
import torchio as tio
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import csv
# from preprocess import get_mhd_files
# import pyvista as pv
import torch
import random
import torch.nn.functional as F 
import torch.nn as nn
from scipy.ndimage import rotate
from time import time


def name2coord(mhd_name, root_dir='/public_bme/data/xiongjl/det'):
    # * 输入name，输出这个name所对应着的gt坐标信息
    xyz = []
    whd = []
    # csv_file_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
    csv_file_dir = os.path.join(root_dir, 'csv_file', 'AT_afterlungcrop_guanfang.csv')
    with open(csv_file_dir, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            
            if row[0] == mhd_name:
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                # radius = float(row[5])
                # result.append((x, y, z, radius))
                w = float(row[4]) 
                h = float(row[5]) 
                d = float(row[6]) 
                xyz.append((x, y, z))
                whd.append((w, h, d))
    # print(f'xyz : {xyz}, whd : {whd}  in name2coord func')
    return xyz, whd


def random_crop_3d(name, crop_size, p=0.8, augmentatoin=False):
    path = '/public_bme/data/xiongjl/det/nii_data_resample_seg_crop/{}_croplung.nii.gz'.format(name)
    image = tio.ScalarImage(path)
    image = image.data[0, :, :, :]

    # new_shape = (crop_size, crop_size, crop_size)
    origin_coords, origin_whd = name2coord(name)
    width, height, depth = image.shape[:]

    crop_width, crop_height, crop_depth = crop_size
    
    # pad the image if it's smaller than the desired crop size
    pad_width = max(0, crop_width - width)
    pad_height = max(0, crop_height - height)
    pad_depth = max(0, crop_depth - depth)
    if pad_height > 0 or pad_width > 0 or pad_depth > 0:
        image = np.pad(image, ((0, pad_width), (0, pad_height), (0, pad_depth)), mode='constant')
        width, height, depth = image.shape[:]

    if random.random() < p:
        # 80% chance to have one or some points in the cropped image
        point = random.choice(origin_coords)
        x, y, z = point
        # 考虑到要是结束的范围比开始的范围要小的话就去强行变化范围
        x_sta = int(max(0, x - crop_width + 1))
        x_stop = int(min(x + 1, width - crop_width))
        y_sta = int(max(0, y - crop_height + 1))
        y_stop = int(min(y + 1, height - crop_height))
        z_sta = int(max(0, z - crop_depth + 1))
        z_stop = int(min(z + 1, depth - crop_depth))
        if x_sta > x_stop:
            x_sta = x_stop - 10
        if y_sta > y_stop:
            y_sta = y_stop - 10
        if z_sta > z_stop:
            z_sta = z_stop - 10
        x1 = random.randint(x_sta, x_stop)
        x2 = x1 + crop_width
        y1 = random.randint(y_sta, y_stop)
        y2 = y1 + crop_height
        z1 = random.randint(z_sta, z_stop)
        z2 = z1 + crop_depth

    else:
        # 20% chance to randomly crop the image
        x1 = random.randint(0, width - crop_width)
        x2 = x1 + crop_width
        y1 = random.randint(0, height - crop_height)
        y2 = y1 + crop_height
        z1 = random.randint(0, depth - crop_depth)
        z2 = z1 + crop_depth
    
    cropped_image = image[x1:x2, y1:y2, z1:z2]

    cropped_points = [(x-x1,y-y1,z-z1) for (x,y,z) in origin_coords if x1 <= x < x2 and y1 <= y < y2 and z1 <= z < z2]

    if augmentatoin == True:
        if random.random() < 0.5:
            pass
        elif random.random() < 0.8:
            cropped_image, cropped_points, origin_whd = rotate_img(cropped_image, cropped_points, origin_whd, rotation_range=(-15, 15))
            cropped_points = [(x,y,z) for (x,y,z) in origin_coords if 0 <= x <= cropped_image.shape[0] and 0 <= y <= cropped_image.shape[1] and 0 <= z <= cropped_image.shape[2]]
        else:
            cropped_image = add_noise(cropped_image)

    #* bulid the other label
    mask = create_mask(cropped_points, crop_size, reduce=1) # 0.0s no save is so fast
    whd = create_whd(coordinates=cropped_points, whd=origin_whd, shape=crop_size, reduce=1)
    offset = create_offset(coordinates=cropped_points, shape=crop_size, reduce=1)
    # hmap = create_hmap(coordinates=cropped_points, shape=crop_size, reduce=1)
    # hmap = create_hmap_v2(coordinates=cropped_points, whds=origin_whd, shape=crop_size)
    # hmap = create_hmap_v3(coordinates=cropped_points, whds=origin_whd, shape=crop_size)
    # hmap = create_hmap_v4(coordinates=cropped_points, whds=origin_whd, shape=crop_size)
    # hmap = create_hmap_v5(coordinates=cropped_points, whds=origin_whd, shape=crop_size)
    hmap = create_hmap_v6(coordinates=cropped_points, whds=origin_whd, shape=crop_size)
    
    hmap = torch.from_numpy(hmap)
    offset = torch.from_numpy(offset)
    mask = torch.from_numpy(mask)
    whd = torch.from_numpy(whd)
    
    dict = {}
    dict['hmap'] = hmap
    dict['offset'] = offset
    dict['mask'] = mask
    dict['input'] = cropped_image
    dict['new_coords'] = cropped_points
    dict['name'] = name
    dict['origin_whd'] = origin_whd
    dict['origin_coords'] = origin_coords
    dict['whd'] = whd

    return dict




def create_mask(coordinates, shape, reduce=4, save=False, name=''):
    
    arr = np.zeros(tuple(np.array(shape) // reduce)) 
    for coord in coordinates:
        x, y, z = coord
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[int(x)][int(y)][int(z)] = 1
    if save:
        np.save('/public_bme/data/xiongjl/det//npy_data//{}_mask.npy'.format(name), arr)
    
    return arr


def create_whd(coordinates, whd, shape, reduce=4, save=False):
    
    arr = np.zeros(tuple(np.insert(np.array(shape) // reduce, 0, 3)))
    for i in range(len(coordinates)):
        x, y, z = coordinates[i]
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[0][int(x)][int(y)][int(z)] = whd[i][0]
        arr[1][int(x)][int(y)][int(z)] = whd[i][1]
        arr[2][int(x)][int(y)][int(z)] = whd[i][2]
    if save:
        np.save('array.npy', arr)
    
    return arr


def create_offset(coordinates, shape, reduce=4, save=False):
    arr = np.zeros(tuple(np.insert(np.array(shape) // reduce, 0, 3)))
    for coord in coordinates:
        x, y, z = coord
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[0][int(x)][int(y)][int(z)] = x - int(x)
        arr[1][int(x)][int(y)][int(z)] = y - int(y)
        arr[2][int(x)][int(y)][int(z)] = z - int(z)
    if save:
        np.save('array.npy', arr)
    return arr


# * load time is 0.09s
def create_hmap(coordinates, shape, reduce=4, save=None, hmap_dir=''): # 1.37s, if save :4.33s
    arr = np.zeros(tuple(np.array(shape) // reduce))
    for coord in coordinates:
        x, y, z = coord
        arr[int(x / reduce)][int(y / reduce)][int(z / reduce)] = 1
    # time_si = time()
    arr = gaussian_filter(arr, sigma=3)
    # print('time of si is {}'.format(time() - time_si))
    if arr.max() == arr.min():
        if save != None:
            np.save(hmap_dir, arr)
        return arr
    else:
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        if save != None:
            np.save(hmap_dir, arr)
        return arr


def create_gaussian_kernel(whd):
    size = int(np.mean(whd))
    if size % 2 == 1:
        size = 2 * size + 1
    else:
        size = 2 * size + 1
    kernel = np.zeros((size, size, size))
    center = tuple(s // 2 for s in (size, size, size))
    kernel[center] = 1
    gassian_kernel = gaussian_filter(kernel, sigma=size//6)
    arr_min = gassian_kernel.min()
    arr_max = gassian_kernel.max()
    normalized_arr = (gassian_kernel - arr_min) / (arr_max - arr_min)
    return normalized_arr


def create_gaussian_kernel_v3(whd):
    size = int(np.mean(whd))
    if size % 2 == 1:
        size = 2 * size + 1
    else:
        size = 2 * size + 1
    kernel = np.zeros((size, size, size))
    center = tuple(s // 2 for s in (size, size, size))
    kernel[center] = 1
    if size // 6 <= 3:
        sigma = 3
    else:
        sigma = size // 6
    gassian_kernel = gaussian_filter(kernel, sigma=sigma)
    arr_min = gassian_kernel.min()
    arr_max = gassian_kernel.max()
    normalized_arr = (gassian_kernel - arr_min) / (arr_max - arr_min)
    return normalized_arr


def create_gaussian_base(size, threshold):

    if size % 2 != 1:  # 如果size是偶数就变成奇数
        dis = size / 2.
        size = size + 1
    else:
        dis = (size + 1) / 2.
    if threshold == 0.5:
        sigma = np.sqrt(dis**2 / (2 * np.log(2)))
    elif threshold == 0.8:
        sigma = np.sqrt(dis**2 / (2 * (np.log(5) - np.log(4))))
    elif threshold == 0.3:
        sigma = np.sqrt(dis**2 / (2 * (np.log(10) - np.log(3))))
    else:
        print(f'when x = distance, the y wrong input, now the threshold is {threshold}')
    kernel = np.zeros((int(size * 2 + 1), int(size * 2 + 1), int(size * 2 + 1)))
    center = tuple(s // 2 for s in (int(size * 2 + 1), int(size * 2 + 1), int(size * 2 + 1)))
    kernel[center] = 1
    # if size // 6 <= 3:
    #     sigma = 3
    # else:
    #     sigma = size // 6
    gassian_kernel = gaussian_filter(kernel, sigma=sigma)
    arr_min = gassian_kernel.min()
    arr_max = gassian_kernel.max()
    normalized_arr = (gassian_kernel - arr_min) / (arr_max - arr_min) # 归一化到 0-1 之间

    return normalized_arr


def combine_gaussian_kernels(kernel_large, kernel_small):
    center_large = np.array(kernel_large.shape) // 2
    small_shape = np.array(kernel_small.shape[0]) // 2

    kernel_large[center_large[0] - small_shape:center_large[0] + small_shape+1, 
                 center_large[1] - small_shape:center_large[1] + small_shape+1, 
                 center_large[2] - small_shape:center_large[2] + small_shape+1, ] += kernel_small[:, :, :]

    arr_min = kernel_large.min()
    arr_max = kernel_large.max()
    normalized_arr = (kernel_large - arr_min) / (arr_max - arr_min) # 归一化到 0-1 之间

    return normalized_arr


def create_gaussian_kernel_v4(whd):
    size_max = int(np.max(whd))
    size_min = int(np.min(whd))

    array_large = create_gaussian_base(size_max, 0.5)
    array_small = create_gaussian_base(size_min, 0.5)
    combined_kernel = combine_gaussian_kernels(array_large, array_small)

    return combined_kernel


def create_gaussian_kernel_v5(whd):
    size_max = int(np.max(whd))
    size_min = int(np.min(whd))

    array_large = create_gaussian_base(size_max, 0.8)
    array_small = create_gaussian_base(size_min, 0.8)
    combined_kernel = combine_gaussian_kernels(array_large, array_small)

    return combined_kernel


def create_gaussian_kernel_v6(whd):
    size_max = int(np.max(whd))
    size_min = int(np.min(whd))

    array_large = create_gaussian_base(size_max, 0.3)
    array_small = create_gaussian_base(size_min, 0.3)
    combined_kernel = combine_gaussian_kernels(array_large, array_small)

    return combined_kernel


def place_kernel_on_image(kernel, image, position):
    x, y, z = position
    x_offset = kernel.shape[0] // 2
    y_offset = kernel.shape[1] // 2
    z_offset = kernel.shape[2] // 2
    x_start = max(0, x - x_offset)
    y_start = max(0, y - y_offset)
    z_start = max(0, z - z_offset)
    x_end = min(image.shape[0], x + x_offset + 1)
    y_end = min(image.shape[1], y + y_offset + 1)
    z_end = min(image.shape[2], z + z_offset + 1)

    # 创建一个与图像大小相同的计数数组
    count = np.zeros_like(image)
    
    # 在重叠位置增加计数
    count[x_start:x_end, y_start:y_end, z_start:z_end] += 1
    image[x_start:x_end, y_start:y_end, z_start:z_end] += kernel[x_start-x+x_offset:x_end-x+x_offset, y_start-y+y_offset:y_end-y+y_offset, z_start-z+z_offset:z_end-z+z_offset]
    # 在重叠位置取平均值
    image[x_start:x_end, y_start:y_end, z_start:z_end] /= count[x_start:x_end, y_start:y_end, z_start:z_end]
    
    return image


def create_hmap_v2(coordinates, whds, shape): # 1.37s, if save :4.33s
    arr = np.zeros(shape)
    for i in range(len(coordinates)):
        coord = [int(x) for x in coordinates[i]]
        whd = [int(x) for x in whds[i]]
        kernel = create_gaussian_kernel(whd)
        arr = place_kernel_on_image(kernel, image=arr, position=coord)

    return arr


def create_hmap_v3(coordinates, whds, shape): # 1.37s, if save :4.33s
    arr = np.zeros(shape)
    for i in range(len(coordinates)):
        coord = [int(x) for x in coordinates[i]]
        whd = [int(x) for x in whds[i]]
        kernel = create_gaussian_kernel_v3(whd)
        arr = place_kernel_on_image(kernel, image=arr, position=coord)

    return arr


def create_hmap_v4(coordinates, whds, shape):
    arr = np.zeros(shape)
    for i in range(len(coordinates)):
        coord = [int(x) for x in coordinates[i]]
        whd = [int(x) for x in whds[i]]
        kernel = create_gaussian_kernel_v4(whd)
        arr = place_kernel_on_image(kernel, image=arr, position=coord)

    return arr


def create_hmap_v5(coordinates, whds, shape):
    arr = np.zeros(shape)
    for i in range(len(coordinates)):
        coord = [int(x) for x in coordinates[i]]
        whd = [int(x) for x in whds[i]]
        kernel = create_gaussian_kernel_v5(whd)
        arr = place_kernel_on_image(kernel, image=arr, position=coord)

    return arr


def create_hmap_v6(coordinates, whds, shape):
    arr = np.zeros(shape)
    for i in range(len(coordinates)):
        coord = [int(x) for x in coordinates[i]]
        whd = [int(x) for x in whds[i]]
        kernel = create_gaussian_kernel_v6(whd)
        arr = place_kernel_on_image(kernel, image=arr, position=coord)

    return arr


def rotate_coords(coordss, angle, center):
    rotated_coordss = []
    for coords in coordss:
        # 将coords转换为NumPy数组
        coords = np.array(coords)
        
        # 计算旋转矩阵
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        
        # 将坐标点平移到旋转中心
        coords -= center
        
        # 旋转坐标点
        rotated_coords = np.dot(coords, R.T)
        
        # 将坐标点平移回原来的位置
        rotated_coords += center

        rotated_coords.tolist()
        rotated_coordss.append(rotated_coords)
    
    return rotated_coordss


def rotate_img(image, coords, whd, rotation_range=(-15, 15)):
    # 将coords和whd转换为NumPy数组
    coords = np.array(coords)
    whd = np.array(whd)
    # 计算旋转角度
    angle = np.random.uniform(rotation_range[0], rotation_range[1])
    # 旋转图像
    rotated_image = rotate(image, angle, axes=(1, 0), reshape=False, mode='constant')
    # 规范化数据
    rotated_image = np.clip(rotated_image, 0, 1)
    # 计算旋转矩阵
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    center = [i/2 for i in image.shape]
    # 计算旋转后的坐标
    rotated_coords = rotate_coords(coords, angle, center)
    # 计算旋转后的whd
    rotated_whd = np.dot(whd, R.T)
    
    return rotated_image, rotated_coords, rotated_whd


def add_noise(img, std=(0, 0.05)):
    if isinstance(img, torch.Tensor):
        image = tio.ScalarImage(tensor=img.unsqueeze(0), type=tio.INTENSITY)
    else:
        image = tio.ScalarImage(tensor=torch.tensor(img).unsqueeze(0), type=tio.INTENSITY)
    transform = tio.RandomNoise(std=std)
    noisy_image = transform(image)
    result = np.array(noisy_image.data.squeeze(0))
    result = np.clip(result, 0, 1)
    return result


def npy2nii(name, image_npy, root_dir='/public_bme/data/xiongjl/det', suffix='', resample=None, affine=''):
    # csv_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_athcoord.csv'
    csv_dir = os.path.join(root_dir, 'annotations_pathcoord.csv')
    df = pd.read_csv(csv_dir)
    df = df[df['seriesuid'] == name]
    
    mhd_path = str(df[['path']].values[0])[2:-2]
    image = tio.ScalarImage(mhd_path)
    if resample != None:
        if affine == '':
            print("affine isn't be given")
    else:
        affine = image.affine
    
    if isinstance(image_npy, np.ndarray):
        image_npy = torch.from_numpy(image_npy)
    if len(image_npy.shape) == 3:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy.unsqueeze(0), affine=affine)
        image_nii.save('./nii_temp/{}_{}.nii'.format(name, suffix))
    elif len(image_npy.shape) == 4:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy, affine=affine)
        image_nii.save('./nii_temp/{}_{}.nii'.format(name, suffix))
    else: 
        print('DIM ERROR : npy.dim != 3 or 4')

    # image_nii.save('./nii_temp/{}_image.nii'.format(name))
    # return print('save done')




if __name__ == '__main__':
    name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.129567032250534530765928856531'

