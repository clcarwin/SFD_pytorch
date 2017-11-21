# cython iou.pyx
# gcc -c -fPIC -I/usr/include/python2.7 -o iou.o iou.c
# gcc -shared -L/usr/lib/x86_64-linux-gnu -lpython2.7 iou.o -o iou.so

def IOU(float ax1,float ay1,float ax2,float ay2,float bx1,float by1,float bx2,float by2):
    cdef float sa,sb,x1,y1,x2,y2,w,h
    sa = abs((ax2-ax1)*(ay2-ay1))
    sb = abs((bx2-bx1)*(by2-by1))
    x1,y1 = max(ax1,bx1),max(ay1,by1)
    x2,y2 = min(ax2,bx2),min(ay2,by2)
    w  = x2 - x1
    h  = y2 - y1
    if w<0 or h<0: return 0.0
    else: return 1.0*w*h/(sa+sb-w*h)