import numpy as np

def bilinear_resize(src,size=(0,0),ratio=0):
    h,w=src.shape[:2]
    if ratio!=0:
        dsth, dstw = h*ratio,w*ratio
    elif size!=(0,0):
        dsth, dstw = size[1],size[0]
    else:
        print("size and ratio error")
        raise ValueError
    src1 = src.copy()
    src1 = np.pad(src1,((0,1),(0,1),(0,0)),'constant')
    dst = np.zeros((dsth,dstw,3),dtype=np.uint8)
    for h_idx in range(dsth):
        for w_idx in range(dstw):
            src_h_idx = (h_idx+1)*h/dsth-1
            src_w_idx = (w_idx+1)*w/dstw-1
            src_h_idx_f = int(np.floor(src_h_idx))
            src_w_idx_f = int(np.floor(src_w_idx))

            if src_h_idx_f<0:
                src_h_idx_f=0
            if src_w_idx_f<0:
                src_w_idx_f=0
            h_delta = src_h_idx-src_h_idx_f
            w_delta = src_w_idx-src_w_idx_f
            for c in range(3):
                dst[h_idx,w_idx,c]=(1-h_delta)*(1-w_delta)*src1[src_h_idx_f,src_w_idx_f,c]+\
                    (1-h_delta)*(w_delta)*src1[src_h_idx_f,src_w_idx_f+1,c]+\
                        (h_delta)*(1-w_delta)*src1[src_h_idx_f+1,src_w_idx_f,c]+\
                            (h_delta)*(w_delta)*src1[src_h_idx_f+1,src_w_idx_f+1,c]
    
    return dst

import cv2

img = cv2.imread("/Users/shenyeqing/file/repo/DIPlearn/images/rocket.png")
img_resize = bilinear_resize(img,ratio=2)
cv2.imwrite("/Users/shenyeqing/file/repo/DIPlearn/images/rocket_biresize.png",img_resize)




