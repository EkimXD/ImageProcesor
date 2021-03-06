from image_procesor import ImageProcesor as ip
import numpy as np
import cv2
from statistics import median
import os
import math as m
import cupy as cp

"""
Este código esta basado en el explicado en:
* https://medium.com/@enzoftware/how-to-build-amazing-images-filters-with-python-median-filter-sobel-filter-%EF%B8%8F-%EF%B8%8F-22aeb8e2f540
* http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/
"""

def is_even(number):
    return number % 2 == 0
def create_2d_filter(rows, columns):
    if (rows == 1 and columns == 1):
        raise "Filter to little, increase rows or columns"
    if is_even(rows) or is_even(columns):
        raise "Rows and columns must be odd numbers"
    return [0] * rows * columns
def create_cross_filter(length):
    return np.zeros((length, length))

def convolution_median_blur(image, kernel):
    # TODO
    return None

def convolution_sobel(image, kernel):
    kernel = cp.array(kernel)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_rows, image_columns = image.shape
    kernel_rows, kernel_columns = kernel.shape
    
    #output = np.zeros(image.shape)
    output = cp.zeros(image.shape)

    pad_height = int((kernel_rows - 1)/2)
    pad_width = int((kernel_columns - 1)/2)

    # Create a np.array with zeros on the borders
    #padded_image = np.zeros((image_rows + (2 * pad_height),(image_columns+(2 * pad_width))))
    padded_image = cp.zeros((image_rows + (2 * pad_height),(image_columns+(2 * pad_width))))
    # Putting values from the original image on the new array padded.
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = cp.array(image)
    #padded_image = cp.copy(padded_image)
    for row in range(image_rows):
        for col in range(image_columns):
            output[row, col] = cp.sum(kernel * padded_image[row:row + kernel_rows, col: col + kernel_columns])
    
    return output

def generate_arrays(rows,columns):
    if rows == 1:
        height_array = [rows]
    else:
        height_array = list(range(-1*int(rows/2),int(rows/2)+1))
    if columns == 1:
        width_array = [columns]
    else:
        width_array = list(range(-1*int(columns/2),int(columns/2)+1))
    return width_array, height_array

def sobel(image):
    width_array, height_array = generate_arrays(3, 3)
    vertical_filter = np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
    horizontal_filter = np.flip(vertical_filter.T, axis=0)

    new_image_x = convolution_sobel(image, vertical_filter)
    new_image_y = convolution_sobel(image, horizontal_filter)

    #magnitud_gradiente = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    magnitud_gradiente = cp.sqrt(cp.square(new_image_x) + cp.square(new_image_y))
    magnitud_gradiente *= 255 / magnitud_gradiente.max()
    return cp.asnumpy(magnitud_gradiente)

def old_sobel(image):
    width_array, height_array = generate_arrays(3, 3)
    padding = max(width_array + height_array)
    
    members = []
    #image = cv2.copyMakeBorder(image_unbordered, padding,padding,padding,padding,cv2.BORDER_REFLECT)
    image = cp.array(image)
    new_image = cp.copy(image)
    width, height, _ = new_image.shape
    for i in range(padding, width-padding):
        for j in range(padding, height-padding):
            #Inicializar Gx y Gy
            Gx = 0
            Gy = 0
            
            p = new_image[i-padding, j-padding]
            r = p[0]
            g = p[1]
            b = p[2]

            intensity = r+g+b

            Gx += -intensity
            Gy += -intensity

            p = new_image[i-padding, j]
            r = p[0]
            g = p[1]
            b = p[2]

            Gx += -2 * (r + g + b)

            p = new_image[i-padding, j+padding]
            r = p[0]
            g = p[1]
            b = p[2]
            Gx += -(r + g + b)
            Gy += (r + g + b)

            p = new_image[i, j-padding]
            r = p[0]
            g = p[1]
            b = p[2]

            Gy += -2 * (r + g + b)

            p = new_image[i, j+padding]
            r = p[0]
            g = p[1]
            b = p[2]

            Gy += 2 * (r + g + b)

            # right column
            p = new_image[i+padding, j-padding]
            r = p[0]
            g = p[1]
            b = p[2]

            Gx += (r + g + b)
            Gy += -(r + g + b)

            p = new_image[i+1, j]
            r = p[0]
            g = p[1]
            b = p[2]

            Gx += 2 * (r + g + b)

            p = new_image[i+1, j+1]
            r = p[0]
            g = p[1]
            b = p[2]

            Gx += (r + g + b)
            Gy += (r + g + b)

            # calculate the length of the gradient (Pythagorean theorem)
            length = m.sqrt((Gx * Gx) + (Gy * Gy))
            # normalise the length of gradient to the range 0 to 255
            length = length / 4328 * 255
            length = int(length)
            # draw the length in the edge image
            #newpixel = img.putpixel((length,length,length))
            image[i,j,0] = length
            image[i,j,1] = length
            image[i,j,2] = length
    return cp.asnumpy(image)

def median_blur(image_unbordered, rows, columns=None):
    if columns == None:
        columns = rows
    if (rows == 1 and columns == 1):
        raise "Filter to little, increase rows or columns"
    if is_even(rows) or is_even(columns):
        raise "Rows and columns must be odd numbers"
    width_array, height_array = generate_arrays(rows, columns)
    #print(width_array, height_array)
    padding = max(width_array + height_array)
    members = []
    #image = cv2.copyMakeBorder(image_unbordered, padding,padding,padding,padding,cv2.BORDER_REFLECT)
    image = cp.copy(image_unbordered)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width, height = image.shape
    # convolucion
    for i in range(padding, width-padding):
        for j in range(padding, height-padding):
            members = [image[i+alpha, j+beta] 
                        for alpha in width_array
                        for beta in height_array]
            image_unbordered[i-padding,j-padding] = median(members)
            #print(i,j,len(members))
    return image_unbordered
    """
    for i in range(0, image.shape[0]-1):
        for j in range(0, image.shape[1]-1):
            members[0] = image[i-1,j-1]
            members[1] = image[i-1,j]
            members[2] = image[i-1,j+1]
            members[3] = image[i,j-1]
            members[4] = image[i,j]
            members[5] = image[i,j+1]
            members[6] = image[i+1,j-1]
            members[7] = image[i+1,j]
            members[8] = image[i+1,j+1]
            members.sort()
            image_unbordered[i-1,j-1] = members[4]
    return image_unbordered
    """

"""
path="src/images/mona.png"
#path="src/prueba.png"
#path="src/images/izayoi.png"

image = cv2.imread(path,0)
cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imshow("Imagen",image)

print(image.shape)
cv2.imshow("Imagen 2",median_blur(image,3))
cv2.imshow("Imagen 3",cv2.medianBlur(image,3))
cv2.waitKey(0)
cv2.destroyAllWindows()    


"""
path="src/images/izayoi.png"
image = cv2.imread(path)
cv2.imshow("Imagen", image)
#cv2.imshow("Imagen 2",old_sobel(image))
cv2.imshow("Imagen 3",sobel(image))
cv2.waitKey(0)
cv2.destroyAllWindows() 
#"""
