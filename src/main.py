from image_procesor import ImageProcesor
import numpy as np

ip = ImageProcesor()
image = ip.read_image('images/izayoi.png')
ip.show_image(image)
image = ip.crop_percentage(image, .2) #corta imagen a un 20%



ip.show_images([image,
                ip.blur_filter(image),
                ip.get_R(image),
                ip.get_G(image),
                ip.get_B(image)])
