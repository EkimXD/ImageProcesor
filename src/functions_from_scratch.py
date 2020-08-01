from image_procesor import ImageProcesor as ip
import numpy as np
import cv2
from statistics import median
import os
import math as m
from matplotlib import pyplot as plt

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
    kernel = np.array(kernel)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_rows, image_columns = image.shape
    kernel_rows, kernel_columns = kernel.shape

    #output = np.zeros(image.shape)
    output = np.zeros(image.shape)

    pad_height = int((kernel_rows - 1)/2)
    pad_width = int((kernel_columns - 1)/2)

    # Create a np.array with zeros on the borders
    #padded_image = np.zeros((image_rows + (2 * pad_height),(image_columns+(2 * pad_width))))
    padded_image = np.zeros(
        (image_rows + (2 * pad_height), (image_columns+(2 * pad_width))))
    # Putting values from the original image on the new array padded.
    padded_image[pad_height:padded_image.shape[0] - pad_height,
                 pad_width:padded_image.shape[1] - pad_width] = np.array(image)
    #padded_image = np.copy(padded_image)
    for row in range(image_rows):
        for col in range(image_columns):
            output[row, col] = np.sum(
                kernel * padded_image[row:row + kernel_rows, col: col + kernel_columns])

    return output


def generate_arrays(rows, columns):
    if rows == 1:
        height_array = [rows]
    else:
        height_array = list(range(-1*int(rows/2), int(rows/2)+1))
    if columns == 1:
        width_array = [columns]
    else:
        width_array = list(range(-1*int(columns/2), int(columns/2)+1))
    return width_array, height_array


def sobel(image):
    width_array, height_array = generate_arrays(3, 3)
    vertical_filter = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    horizontal_filter = np.flip(vertical_filter.T, axis=0)

    new_image_x = convolution_sobel(image, vertical_filter)
    new_image_y = convolution_sobel(image, horizontal_filter)

    #magnitud_gradiente = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    magnitud_gradiente = np.sqrt(
        np.square(new_image_x) + np.square(new_image_y))
    magnitud_gradiente *= 255 / magnitud_gradiente.max()
    return magnitud_gradiente


def old_sobel(image):
    width_array, height_array = generate_arrays(3, 3)
    padding = max(width_array + height_array)

    members = []
    #image = cv2.copyMakeBorder(image_unbordered, padding,padding,padding,padding,cv2.BORDER_REFLECT)
    image = np.array(image)
    new_image = np.copy(image)
    width, height, _ = new_image.shape
    for i in range(padding, width-padding):
        for j in range(padding, height-padding):
            # Inicializar Gx y Gy
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
            image[i, j, 0] = length
            image[i, j, 1] = length
            image[i, j, 2] = length
    return image


def cross_median_blur(image_unbordered, rows, columns=None):
    if columns == None:
        columns = rows
    if (rows == 1 and columns == 1):
        raise "Filter to little, increase rows or columns"
    if is_even(rows) or is_even(columns):
        raise "Rows and columns must be odd numbers"
    width_array, height_array = generate_arrays(rows, columns)
    del(width_array[1])
    #print(width_array, height_array)
    padding = max(width_array + height_array)
    members = []
    #image = cv2.copyMakeBorder(image_unbordered, padding,padding,padding,padding,cv2.BORDER_REFLECT)
    image = np.copy(image_unbordered)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width, height = image.shape
    # convolucion
    for i in range(padding, width-padding):
        for j in range(padding, height-padding):
            height_members = [image[i, j+beta] for beta in height_array]
            width_members = [image[i+alpha, j] for alpha in width_array]
            members = height_members + width_members
            image_unbordered[i-padding, j-padding] = median(members)
            # print(i,j,len(members))
    return image_unbordered

def equis_median_blur(image_unbordered, rows, columns=None):
    if columns == None:
        columns = rows
    if (rows == 1 and columns == 1):
        raise "Filter to little, increase rows or columns"
    if is_even(rows) or is_even(columns):
        raise "Rows and columns must be odd numbers"
    width_array, height_array = generate_arrays(rows, columns)
    del(width_array[1])
    del(height_array[1])
    #print(width_array, height_array)
    padding = max(width_array + height_array)
    members = []
    #image = cv2.copyMakeBorder(image_unbordered, padding,padding,padding,padding,cv2.BORDER_REFLECT)
    image = np.copy(image_unbordered)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width, height = image.shape
    # convolucion
    for i in range(padding, width-padding):
        for j in range(padding, height-padding):
            members = [image[i+alpha, j+beta]
                       for alpha in width_array
                       for beta in height_array]
            members.append(image[i,j])
            image_unbordered[i-padding, j-padding] = median(members)
            # print(i,j,len(members))
    return image_unbordered

def get_histogram(image, bins):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # find frequency of pixels in range 0-255 
    histr = cv2.calcHist(image,[0],None,[256],[0,256]) 
    # show the plotting graph of an image 
    plt.plot(histr) 
    plt.show() 

# This code was based on: https://github.com/hosnaa/Histograms_scratch
def make_histogram(image):
    if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(image2.size)
    img_arr = np.copy(image)
    flat = img_arr.flatten()

    # Take a flattened greyscale image and create a historgram from it
    image_range = 100 
    histogram = np.zeros(image_range, dtype=int)
    for i in range(flat.size):
        histogram[int(flat[i]*image_range/256)] += 1
    return histogram

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
    image = np.copy(image_unbordered)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width, height = image.shape
    # convolucion
    for i in range(padding, width-padding):
        for j in range(padding, height-padding):
            members = [image[i+alpha, j+beta]
                       for alpha in width_array
                       for beta in height_array]
            image_unbordered[i-padding, j-padding] = median(members)
            # print(i,j,len(members))
    return image_unbordered

# Este código fue tomado de:  https://github.com/PavanGJ/Image-Processing-Techniques
class FourierTransform() :
    def __init__(self) :
        self.f = None
        self.F = None
        self.magnitude = None
        self.phase = None
        self.M = None
        self.N = None
        self.image = None
    def setImage(self, image) :
        #image = np.asarray(image, dtype = float)
        if len(image.shape) == 2 :
            self.image = image
            try :
                self.dimensions = self.image.shape
            except :
                print("Internal Error! Image file not supported")
        else :
            print("Assignment Error. Given input is not an image")
    def forwardTransform(self) :
        try :
            M = self.image.shape[0]
            N = self.image.shape[1]
        except :
            print("Internal Error! Could not decompose the image shape")
            return
        x = np.arange(M, dtype = float)
        y = np.arange(N, dtype = float)
        u = x.reshape((M,1))
        v = y.reshape((N,1))
        exp_1 = pow(np.e, -2j*np.pi*u*x/M)
        exp_2 = pow(np.e, -2j*np.pi*v*y/N)
        self.F = np.dot(exp_2, np.dot(exp_1,self.image).transpose())/(M*N)
        return self.F

    def inverseTransform(self) :
        try :
            M = self.F.shape[0]
            N = self.F.shape[1]
        except :
            print("Internal Error! Could not decompose the image shape")
            return
        x = np.arange(M, dtype = float)
        y = np.arange(N, dtype = float)
        u = x.reshape((M,1))
        v = y.reshape((N,1))
        exp_1 = pow(np.e, 2j*np.pi*u*x/M)
        exp_2 = pow(np.e, 2j*np.pi*v*y/N)
        self.f = np.dot(exp_2, np.dot(exp_1,self.F).transpose())
        return self.f
    def shift(self, image) :
        try :
            M = image.shape[0]
            N = image.shape[1]
        except :
            print("Internal Error! Could not decompose the image shape")
            return
        m = int(M/2)
        n = int(N/2)
        temp = np.zeros((M,N))
        print(image.shape, m, n)
        temp[-m:,-n:] = np.abs(np.copy(image[:m,:n]))
        temp[-m:,:-n] = np.abs(np.copy(image[:m,n:]))
        temp[:-m,-n:] = np.abs(np.copy(image[m:,:n]))
        temp[:-m,:-n] = np.abs(np.copy(image[m:,n:]))
        return temp
    def error(self) :
        E = (self.image - self.f)**2
        M = E.shape[0]
        N = E.shape[1]
        I = np.ones((1,N))
        J = np.ones((M,1))
        print("Error: %s"% np.abs(np.dot(np.dot(I,E.transpose()),J)))

class Image() :
    def __init__(self) :
        self.image = None
        self.type = None
        self.dimensions = []
    def open(self,path,mode = cv2.IMREAD_GRAYSCALE) :
        try :
            self.image =  cv2.imread(path, mode)
        except Exception as e:
            print(e+",Error! Could not read the image from the path specified: %s"%path)
            return
        
        try :
            self.image = np.asarray(self.image, dtype = float)
            self.dimensions = self.image.shape
            self.type = path.split(".")[-1]
        except :
            print("Internal Error! Image file not supported")
    def set(self, image) :
        image = np.asarray(image, dtype = float)
        if len(image.shape) == 2 :
            self.image = image
            try :
                self.dimensions = self.image.shape
            except :
                print("Internal Error! Image file not supported")
        else :
            print("Assignment Error. Given input is not an image")
    def show(self, mode='Greys_r', name=None) :
        try :
            plt.imshow(self.image,cmap=mode)
        except :
            print("Image Could not be displayed")
            return
        if not name is None :
            plt.imsave(name,self.image,cmap=mode)
        plt.show()
def computeFourierTransforms(image) :
        image_old = np.copy(image)
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image_old, cv2.COLOR_BGR2GRAY)
        fourierTransform = FourierTransform()
        fourierTransform.setImage(image_gray)
        fimg = Image()
        fimg.set(np.log(np.abs(fourierTransform.shift(fourierTransform.forwardTransform()))**2))
        # fimg.show()
        return fimg.image
        """
        infimg = Image()
        infimg.set(np.abs(fourierTransform.inverseTransform()))
        # infimg.show()
        fourierTransform.error()
        width, height = infimg.image.shape
        for i in range(width):
            for j in range(height):
                image_old[i, j] = infimg.image[i, j]
        return image_old
        """