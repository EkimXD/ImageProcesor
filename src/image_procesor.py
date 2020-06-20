import cv2
import numpy as np


class ImageProcesor:

    def get_R(self, image):
        img = image.copy()
        img[:, :, 0] = 0
        img[:, :, 1] = 0
        return img

    def get_G(self, image):
        img = image.copy()
        img[:, :, 0] = 0
        img[:, :, 2] = 0
        return img

    def get_B(self, image):
        img = image.copy()
        img[:, :, 1] = 0
        img[:, :, 2] = 0
        return img

    def get_only_R(self, image):
        return image[:, :, 2]

    def get_only_G(self, image):
        return image[:, :, 1]

    def get_only_B(self, image):
        return image[:, :, 0]

    def read_image(self, directory):
        return cv2.imread(directory)

    def crop_percentage(self, image, percentage):
        x, y, z = image.shape
        xi = x / 2
        yi = y / 2
        x = int(x * percentage)
        y = int(y * percentage)
        xi = int(xi - x / 2)
        yi = int(yi - y / 2)
        return self.crop_coordinates(image, xi, yi, x, y)

    def crop_coordinates(self, image, xi, yi, x, y):
        return image[xi:xi + x + 1, yi:yi + y + 1].copy()

    def show_image(self, image):
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    def show_images(self, images):  # solo acepta arreglos de imagenes del mismo tamano
        cv2.imshow("Image", np.hstack(images))
        cv2.waitKey(0)

    def change_image_character(self, image, savedata=False, characters=['#', '&', '$', '%', '/', 'i', ';', ':', '.',
                                                                        ' ']):  # solo del primer canal por que si xD
        parts = 255 / len(characters)
        imgaux = []
        for x in range(len(image)):
            for y in range(len(image[0])):
                imgaux.append(characters[int((image[x, y, 0] - 1) / parts)])
        imgaux = np.array(imgaux).reshape((len(image), len(image[0])))
        if savedata:
            np.savetxt('data.txt', imgaux, fmt='%s', delimiter='   ')
        return imgaux

    def blur_filter(self, image, pblur=1):  # no hacer caso, es basura :v
        imgaux = image.copy()
        for z in range(len(imgaux[0, 0])):
            for x in range(1, len(imgaux) - 1):
                for y in range(1, len(imgaux[0]) - 1):
                    media = np.mean(self.crop_coordinates(image[:, :, z], x - 1, y - 1, x + 1, y + 1))
                    imgaux[x, y, z] = int((imgaux[x, y, z] + media) / 2)
        return imgaux
