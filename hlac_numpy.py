# -*- coding: utf-8 -*-

import cv2
import numpy as np

def hlac(img, dim=3):
    """
    HLAC（高次局所自己相関関数）特徴量を計算する

    Parameters
    ----------
    img : numpy.ndarray
        HLACを計算したい2値画像
    dim : int, default 3
        局所領域のサイズ
        
    Returns
    -------
    res : list
        HLAC特徴量のリスト（25次元）
    
    Notes
    -----
    dim=3の場合のみ実装
    
    Examples
    --------
    >>> img = np.zeros((100,100)) # 100×100の画像を生成
    >>> img[:30] = 255
    >>> hlac_feature = hlac(img)
    >>> hlac_feature
    [2842, 2842, 2842, 2842, 2842, 
     2842, 2744, 2744, 2744, 2842, 
     2744, 2744, 2744, 2744, 2744, 
     2744, 2842, 2842, 2744, 2744, 
     2744, 2744, 2744, 2842, 2842]
    """
    
    masks = ['000010000','000011000','001010000','010010000','100010000',
             '000111000','001010100','010010010','100010001','001110000',
             '010010100','100010010','000110001','000011100','001010010',
             '010010001','100011000','010110000','100010100','000110010',
             '000010101','000011010','001010001','010011000','101010000']
    
    height, width = img.shape
    res = [0 for i in range(len(masks))]
    for i in range(height-2):
        for j in range(width-2):
            img_ = to1d(img[i:i+dim, j:j+dim])
            pos_mask = 0
            for mask in masks:
                c1 = 0
                c2 = 0
                pos_img = 0
                for char in mask:
                    if char == "1":
                        c1 += 1
                        if img_[pos_img] == 255:
#                            print("----")
                            c2 += 1
                    pos_img += 1
                if c1 == c2:
                    res[pos_mask] += 1
                pos_mask += 1
                    
    return res

def hlac_v2(img, dim=3):
    masks = ['000010000','000011000','001010000','010010000','100010000',
             '000111000','001010100','010010010','100010001','001110000',
             '010010100','100010010','000110001','000011100','001010010',
             '010010001','100011000','010110000','100010100','000110010',
             '000010101','000011010','001010001','010011000','101010000']
    height, width = img.shape
    
def patchify(img, patch_shape):
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

def to1d(img):
    """
    2dのリストを1dに変換して返す
    """
    
    img_ = []
    for i in img:
        img_.extend(i)
    return img_

def main():
    img = cv2.imread("./sample.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    hlacs = hlac(img)
    
def test():
    """
    テスト用の関数
    """
    
    # img1 = np.zeros((100,100))
    # img2 = np.zeros((100,100))
    
    # img1[:30] = 255
    # img2[70:] = 255
    
    # hlacs1 = hlac(img1)
    # hlacs2 = hlac(img2)
    
#    print(hlacs1)
#    print(hlacs2)
#    print([h1+h2 for h1,h2 in zip(hlacs1, hlacs2)])

    mask = np.array([[0,0,0],[0,1,0],[0,1,0]])
    
    img3 = np.zeros(25).reshape(5, 5)
    img3[1, 1] = 255
    img3[2, 1] = 255
    # print(img3)
    a = patchify(img3, (3, 3))
    print(a)
    a = a.reshape(9,3,3)
    print(a)

    res = np.logical_and(mask, a)
    print(res)
    # print(np.sum(res, axis=2))
    # print(np.sum(np.sum(res, axis=2), axis=1))
    logic = np.sum(np.sum(res, axis=2), axis=1)
    print(np.sum(logic==2))

    # hlacs3 = hlac(img3)
    
    # print(hlacs3)

if __name__ == "__main__":
    test()
