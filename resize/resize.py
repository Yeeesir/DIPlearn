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

def nearest_resize(src,size=(0,0),ratio=0):
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
            src_h_idx = int((h_idx+1)*h/dsth-1)
            src_w_idx = int((w_idx+1)*w/dstw-1)
            for c in range(3):
                dst[h_idx,w_idx,c]=src1[src_h_idx,src_w_idx,c]
    return dst

def bicubic_weight(p1=(0.0,0.0),p2=(0.0,0.0)):
    a=0.5
    w=0.0
    dist=np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))
    if dist>2:
        w=0.0
    elif dist>1:
        w=a*dist*dist*dist-5*a*dist*dist+8*a*dist-4*a
    else:
        w=(a+2)*dist*dist*dist-(a+3)*dist*dist+1
    return w


def bicubic_resize(src,size=(0,0),ratio=0):
    h,w=src.shape[:2]
    if ratio!=0:
        dsth, dstw = h*ratio,w*ratio
    elif size!=(0,0):
        dsth, dstw = size[1],size[0]
    else:
        print("size and ratio error")
        raise ValueError
    src1 = src.copy()
    src1 = np.pad(src1,((1,1),(1,1),(0,0)),'edge')
    dst = np.zeros((dsth,dstw,3),dtype=np.uint8)
    for h_idx in range(dsth):
        for w_idx in range(dstw):
            src_h_idx = (h_idx+1)*h/dsth-1
            src_w_idx = (w_idx+1)*w/dstw-1
            src_h_idx_f = int(np.floor(src_h_idx))
            src_w_idx_f = int(np.floor(src_w_idx))
            h_delta = src_h_idx-src_h_idx_f
            w_delta = src_w_idx-src_w_idx_f
            for c in range(3):
                dst[h_idx,w_idx,c]=src1[src_h_idx_f-1,src_w_idx_f-1,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f-1,src_w_idx_f-1))+\
                    src1[src_h_idx_f-1,src_w_idx_f,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f-1,src_w_idx_f))+\
                        src1[src_h_idx_f-1,src_w_idx_f+1,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f-1,src_w_idx_f+1))+\
                            src1[src_h_idx_f-1,src_w_idx_f+2,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f-1,src_w_idx_f+2))+\
                                src1[src_h_idx_f,src_w_idx_f-1,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f,src_w_idx_f-1))+\
                    src1[src_h_idx_f,src_w_idx_f,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f,src_w_idx_f))+\
                        src1[src_h_idx_f,src_w_idx_f+1,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f,src_w_idx_f+1))+\
                            src1[src_h_idx_f,src_w_idx_f+2,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f,src_w_idx_f+2))+\
                                src1[src_h_idx_f+1,src_w_idx_f-1,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f+1,src_w_idx_f-1))+\
                    src1[src_h_idx_f+1,src_w_idx_f,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f+1,src_w_idx_f))+\
                        src1[src_h_idx_f+1,src_w_idx_f+1,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f+1,src_w_idx_f+1))+\
                            src1[src_h_idx_f+1,src_w_idx_f+2,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f+1,src_w_idx_f+2))+\
                                src1[src_h_idx_f+2,src_w_idx_f-1,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f+2,src_w_idx_f-1))+\
                    src1[src_h_idx_f+2,src_w_idx_f,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f+2,src_w_idx_f))+\
                        src1[src_h_idx_f+2,src_w_idx_f+1,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f+2,src_w_idx_f+1))+\
                            src1[src_h_idx_f+2,src_w_idx_f+2,c]*bicubic_weight((src_h_idx,src_w_idx),(src_h_idx_f+2,src_w_idx_f+2))
                dst[h_idx,w_idx,c] = dst[h_idx,w_idx,c].clip(0,255)
    return dst

def bicubic(x):
    x = np.abs(x)
    if x<=1:
        return 1-2*(x**2)+(x**3)
    elif x<2:
        return 4-8*x+5*(x**2)-(x**3)
    else:
        return 0

def bicubic_resize_v2(src,size=(0,0),ratio=0):
    h,w=src.shape[:2]
    if ratio!=0:
        dsth, dstw = h*ratio,w*ratio
    elif size!=(0,0):
        dsth, dstw = size[1],size[0]
    else:
        print("size and ratio error")
        raise ValueError
    src1 = src.copy()
    src1 = np.pad(src1,((1,1),(1,1),(0,0)),'edge')
    dst = np.zeros((dsth,dstw,3),dtype=np.uint8)
    for h_idx in range(dsth):
        for w_idx in range(dstw):
            src_h_idx = (h_idx+1)*h/dsth-1
            src_w_idx = (w_idx+1)*w/dstw-1
            src_h_idx_f = int(np.floor(src_h_idx))
            src_w_idx_f = int(np.floor(src_w_idx))
            h_delta = src_h_idx-src_h_idx_f
            w_delta = src_w_idx-src_w_idx_f
            
            for c in range(3):
                tmp=0.0
                for ii in range(-1,3):
                    for jj in range(-1,3):
                        if src_h_idx_f+ii<0 or src_h_idx_f+ii>=h or src_w_idx_f+ii<0 or src_w_idx_f+ii>=w:
                            continue
                        tmp+=src1[src_h_idx_f+ii,src_w_idx_f+jj,c]*bicubic(ii-h_delta)*bicubic(jj-w_delta)
                if tmp>255:
                    tmp=255
                dst[h_idx,w_idx,c]=tmp
    return dst

import cv2

img = cv2.imread("/Users/shenyeqing/repo/DIPlearn/images/rocket.png")
img_resize = bicubic_resize_v2(img,ratio=2)
cv2.imwrite("/Users/shenyeqing/repo/DIPlearn/images/rocket_bicubicreisze_v2.png",img_resize)



