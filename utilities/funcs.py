import cv2 as cv
import numpy as np

def get_biggest_object (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 1
    return img2

def process_img (img):
    ret,temp_img = cv.threshold(img,60,1,cv.THRESH_TOZERO)
    
    kernel_cl1 = np.ones((19,19),np.uint8)
    temp_img = cv.morphologyEx(temp_img, cv.MORPH_DILATE, kernel_cl1)

    temp_mask = get_biggest_object(temp_img)
    temp_img = temp_img*temp_mask

    kernel_cl1 = np.ones((19,19),np.uint8)
    temp_img = cv.morphologyEx(temp_img, cv.MORPH_ERODE, kernel_cl1)

    kernel_op1 = np.ones((6,6),np.uint8)
    temp_img = cv.morphologyEx(temp_img, cv.MORPH_OPEN, kernel_op1)

    kernel_cl1 = np.ones((19,19),np.uint8)
    temp_img = cv.morphologyEx(temp_img, cv.MORPH_CLOSE, kernel_cl1)

    ret,final_mask = cv.threshold(temp_img,20,1,cv.THRESH_BINARY)


    return final_mask