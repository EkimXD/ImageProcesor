import numpy
import cairo
import math
from math import pi
"""
data = numpy.zeros((20, 20, 4), dtype=numpy.uint8)
surface = cairo.ImageSurface.create_for_data(
    data, cairo.FORMAT_ARGB32, 20, 20)
cr = cairo.Context(surface)

# fill with solid white
cr.set_source_rgb(1.0, 1.0, 1.0)
cr.paint()

# draw red circle
cr.arc(10, 10, 8, 0, 2*math.pi)
cr.set_line_width(3)
cr.set_source_rgb(0.0, 0.0, 0.0)
cr.stroke()

# write output
#print (data[38:48, 38:48, 0])
surface.write_to_jpg("circle.png")
"""
"""

width = 20
height = 20
data = numpy.zeros((20, 20, 4), dtype=numpy.uint8)
surface = cairo.ImageSurface.create_for_data(
    data, cairo.FORMAT_ARGB32, 20, 20)
cr = cairo.Context(surface)
cr.set_source_rgb(1.0, 1.0, 1.0)
cr.rectangle(0, 0, width, height)
cr.fill()

# draw a rectangle
cr.set_source_rgb(1.0, 1.0, 1.0)
cr.rectangle(10, 10, width - 20, height - 20)
cr.fill()

# set up a transform so that (0,0) to (1,1)
# maps to (20, 20) to (width - 40, height - 40)
cr.translate(20, 20)
cr.scale((width - 40) / 1.0, (height - 40) / 1.0)

# draw lines
cr.set_line_width(0.2)
cr.set_source_rgb(0.0, 0.0, 0.0)
cr.move_to(1 / 3.0, 1 / 3.0)
cr.rel_line_to(0, 1 / 6.0)
cr.move_to(2 / 3.0, 1 / 3.0)
cr.rel_line_to(0, 1 / 6.0)
cr.stroke()

# and a circle
cr.set_source_rgb(0.0, 0.0, 0.0)
radius = 1
cr.arc(0.5, 0.5, 0.5, 0, 2 * pi)
cr.stroke()
cr.arc(0.5, 0.5, 0.33, pi / 3, 2 * pi / 3)
cr.stroke()
surface.write_to_png("smile.png")


import cv2
image = cv2.imread("circle.png")
print(type(image))
for i in range(0,image.shape[0]-1):
    print(image[i])
"""
import cv2
from matplotlib import pyplot as plt
image = cv2.imread('ex.png',0) 
#if len(image.shape) == 3:
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#x=image[:,:,0]
histr = cv2.calcHist([image],[0],None,[256],[0,256]) 
plt.title("HIstogramm for given Image")
plt.xlabel("Value")
plt.ylabel("pixels Frequency")
plt.plot(histr) 
plt.show()