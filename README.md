#### Importar clase

---
~~~
from image_procesor import ImageProcesor
~~~

#### Mostrar imagen

---

~~~
ip = ImageProcesor()
image = ip.read_image('imagePath/')
ip.show_image(image) # muestra una sola imagen
ip.show_images([image]) #muestra un conjunto de imagenes
~~~

![imagen 1](src/images/Captura%20de%20pantalla_2020-06-20_08-48-56.png)

#### Transformar una imagenes a caracteres

---

~~~
ip = ImageProcesor()
image = ip.read_image('imagePath/')
text=ip.change_image_character(image)
~~~

Para guardar en un archivo:

~~~
ip = ImageProcesor()
image = ip.read_image('imagePath/')
text=ip.change_image_character(image,savedata=True)
~~~

![imagen 1](src/images/Captura%20de%20pantalla_2020-06-18_21-33-39.png)

#### Desenfocar imagen

---

~~~
ip = ImageProcesor()
image = ip.blur_filter(image)
~~~


![imagen 1](src/images/Captura%20de%20pantalla_2020-06-19_09-53-08.png)

#### Algunas otras cosas :D

---

~~~
ip = ImageProcesor()
image = ip.read_image('images/izayoi.png')
image = ip.crop_percentage(image, .2) #corta imagen a un 20%

ip.show_images([image,
                ip.blur_filter(image), 
                ip.get_R(image), #coloca los canales G y B en 0
                ip.get_G(image), #coloca los canales R y B en 0
                ip.get_B(image)]) #coloca los canales R y G en 0
~~~

![imagen 1](src/images/Captura%20de%20pantalla_2020-06-20_08-33-56.png)

#### Usar UI provista por el desarrollador

---

Unicamente es necesario ejecutar la clase main que se encuentra en la carpeta src
~~~
python src/main.py
~~~

![imagen 1](src/images/Captura%20UI.png)

Cosas a considerar:
- Los valores en los recuadros estan mapeados en 11 niveles, del 0 al 10, es decir de 0 a 255 respectivamente.
- Si la imagen que se carga es demaciado grande solamente se muestra una porcion de la imagen, en este caso solo la parte superior derecha.
- El boton "Show image" abre una ventana en la que se puede dibujar, pero solo en comor negro, ademas para regresar a la anterior ventana se tiene que presionar enter.


![imagen 1](src/images/Captura.png)

