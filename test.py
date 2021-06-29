import numpy as np
import cv2 as cv

output = './motion.avi'

cap = cv.VideoCapture('./data/video/jump9.mp4')

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
codec = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter(output, codec, fps, (width, height))

ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    ydiff = flow[...,0]
    xdiff = flow[...,1]
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    # print(flow[0][0][0], flow[0][0][1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)

    # for rowIdx, row in enumerate(mag):
    #     for colIdx, col in enumerate(row):
    #         if(col > 5):
    #             cv.line(next, (rowIdx, colIdx), (rowIdx + int(ydiff[rowIdx, colIdx]), colIdx + int(xdiff[rowIdx, colIdx])), (0, 255, 0), 1)

    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    out.write(bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next