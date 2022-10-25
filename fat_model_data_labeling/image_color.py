import glob
from turtle import bgcolor
import cv2
import numpy as np

final_path = './images_segm'
image_files = glob.glob(final_path+'/*')
for filename in image_files:
    
    img = cv2.imread(filename)

    bgr = img[:,:,0:3]
    mask = cv2.inRange(bgr,(0,0,0),(150, 255, 30))
    img2 = bgr.copy()

    img2[mask==255] = (0,0,0)
    print(filename)
    hsv_img = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    bound_lower = np.array([0, 50, 0])
    bound_upper = np.array([150, 255, 150])
    mask = cv2.inRange(hsv_img, bound_lower, bound_upper)
    kernel = np.ones((7,7),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    seg_img = cv2.bitwise_and(img, img, mask=mask)

    bgr = seg_img[:,:,0:3]
    mask = cv2.inRange(bgr,(0,0,0),(10, 10, 10))
    bgr_new = bgr.copy()

    bgr_new[mask!=255] = (255,0,0)
    filename = filename.replace('./images','./images_newcolor')
    print(filename)
    cv2.imwrite(filename,bgr_new)

'''
img = cv2.imread('./image.png',cv2.IMREAD_COLOR)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
bgr = img[:,:,0:3]
mask = cv2.inRange(bgr,(63,0,50),(66,0,150))
bgr_new = bgr.copy()

bgr_new[mask==255] = (255,0,0)

cv2.imwrite('./image_new.png',bgr_new)
'''
'''
img = cv2.imread('./49_left.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
bound_lower = np.array([50, 100, 50])
bound_upper = np.array([200, 255, 200])

mask = cv2.inRange(hsv_img, bound_lower, bound_upper)
kernel = np.ones((7,7),np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

seg_img = cv2.bitwise_and(img, img, mask=mask)


bgr = seg_img[:,:,0:3]
mask = cv2.inRange(bgr,(55,0,55),(205, 155, 205))
bgr_new = bgr.copy()

bgr_new[mask==255] = (255,0,0)


cv2.imwrite('./image_new.png',bgr_new)
'''


'''


img = cv2.imread('./59_right.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
bound_lower = np.array([0, 100, 0])
bound_upper = np.array([150, 255, 150])
mask = cv2.inRange(hsv_img, bound_lower, bound_upper)
kernel = np.ones((7,7),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
seg_img = cv2.bitwise_and(img, img, mask=mask)

bgr = seg_img[:,:,0:3]
mask = cv2.inRange(bgr,(0,0,0),(10, 10, 10))
bgr_new = bgr.copy()

bgr_new[mask!=255] = (255,0,0)
cv2.imwrite('./image_new.png',bgr_new)
'''