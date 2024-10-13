import funciones as func    
from PIL import Image
import os

# Path de la imagen a editar
path_image = 'C:\\Users\\javir\\Documents\\Python\\VA\\Imagenes\\image2.png'

# Definir un kernel
kernel = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]
kernel = [[x / 16 for x in row] for row in kernel]  # Normalizado para desenfoque

# Valor de sigma
sigma = 1.0

# Kernel Guassiano unidimensional
kernelG = func.gaussKernel1D(sigma)
print("Kernel Gaussiano (sigma = {}):".format(sigma))
print(kernelG)

# Tamaño del fitro
filterSize = 3

# Elemento estructurante 
SE = [[0, 1, 0],
      [1, 1, 1],
      [0, 1, 0]]  

# Cargar la imagen
inImage = Image.open(path_image).convert('L')  # Convertir a escala de grises

# Alteración del rango dinámico
altered_image = func.adjustIntensity(inImage, inRange=[0,255], outRange=[0, 1])
# Ecualización de histograma
ecualized_image = func.equalizeIntensity(inImage, 2048)
# Filtrado espacial mediante convolución
filtered_image = func.filterImage(inImage, kernel)
# Suavizado Gaussiano bidimensional
gaussian_image = func.gaussianFilter(inImage, sigma)
# Filtro de medianas bidimensional
median_image = func.medianFilter(inImage, filterSize)
# Operador morfológico de erosión
eroded_image = func.erode(inImage, SE)
# Operador morfológioc de dilatación
dilated_image = func.dilate(inImage, SE)
# Operador morfológico de apertura
opened_image = func.opening(inImage, SE)
# Operador morfológico de cierre
closed_image = func.closing(inImage, SE)
# Llenado de operadores morfológicos
filled_image = func.fill(inImage, SE)
# Gradiente de una imagen
gx, gy = func.gradientImage(inImage, operator='Sobel')

# Lista con los arrays procesadas
imagenes = [altered_image, ecualized_image, filtered_image, gaussian_image, median_image, eroded_image, dilated_image, opened_image, closed_image, filled_image]

# Convertir a una imagen y guardar
for i, img in enumerate(imagenes):

    final_image = Image.fromarray(img) 
    
    final_image.save(os.path.splitext(path_image)[0] + '_' + str(i+1) + '.jpg')