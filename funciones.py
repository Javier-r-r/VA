import numpy as np
from scipy.ndimage import convolve

def adjustIntensity(inImage, inRange=[], outRange=[0, 1]):

    # Convertir la imagen a un arreglo numpy
    inImage = np.array(inImage)
    
    # Si no se especifica inRange, usar el mínimo y máximo de la imagen
    if not inRange:
        imin = np.min(inImage)
        imax = np.max(inImage)
    else:
        imin, imax = inRange
    
    omin, omax = outRange
    
    # Normalizar la imagen en el rango de salida
    outImage = (inImage - imin) / (imax - imin) * (omax - omin) + omin
    
    # Asegurarse de que los valores de salida estén dentro de los límites
    outImage = np.clip(outImage, omin, omax)

    adjusted_image_pil = np.uint8(outImage * 255)  # Escalar a 0-255

    return adjusted_image_pil
    
def equalizeIntensity(inImage, nBins):

    # Convertir la imagen a un arreglo numpy y normalizar
    img_array = np.asarray(inImage) / 255.0  # Normalizar entre 0 y 1
    
    # Inicializar el histograma
    hist = np.zeros(nBins)
    
    # Calcular el histograma manualmente
    for pixel in img_array.flatten():
        # Calcular el índice del bin correspondiente
        bin_index = int(pixel * nBins)
        if bin_index == nBins:  # Para manejar el caso donde pixel es 1.0
            bin_index = nBins - 1
        hist[bin_index] += 1

    # Calcular la función de distribución acumulativa (CDF)
    cdf = np.zeros(nBins)
    cdf[0] = hist[0]
    for i in range(1, nBins):
        cdf[i] = cdf[i - 1] + hist[i]
    
    # Normalizar la CDF
    cdf_normalized = cdf / cdf[-1]  # Normalizar entre 0 y 1

    # Mapear la intensidad de entrada a la nueva intensidad utilizando la CDF normalizada
    outImage = np.interp(img_array, np.linspace(0, 1, nBins), cdf_normalized)

    # Normalizar de nuevo a [0, 255]
    return (outImage * 255).astype(np.uint8)  # Convertir de nuevo a [0, 255]

def filterImage(inImage, kernel):

    # Convertir la imagen a un array numpy
    inImage = np.array(inImage)
    
    # Obtener las dimensiones de la imagen y el kernel
    image_height, image_width = inImage.shape
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])

    # Calcular el padding necesario
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Añadir padding a la imagen
    padded_image = np.pad(inImage, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Crear una matriz para la imagen de salida
    outImage = np.zeros_like(inImage)

    # Realizar la convolución
    for i in range(image_height):
        for j in range(image_width):
            # Extraer la región de interés
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Realizar la convolución
            outImage[i, j] = np.sum(region * kernel)

    # Normalizar la imagen de salida para que esté en el rango [0, 255]
    outImage = np.clip(outImage, 0, 255)  # Asegurarse de que los valores están en el rango correcto
    return outImage.astype(np.uint8)

def gaussKernel1D(sigma):
    # Calcular el tamaño del kernel
    N = int(2 * np.ceil(3 * sigma) + 1)
    # Calcular el centro del kernel
    center = N // 2
    
    # Crear el kernel gaussiano
    kernel = np.zeros(N)
    for x in range(N):
        # Fórmula de la función gaussiana
        kernel[x] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    
    # Normalizar el kernel
    kernel /= np.sum(kernel)
    
    return kernel

def gaussianFilter(inImage, sigma):
    
    # Calcular el kernel unidimensional
    kernel_1d = gaussKernel1D(sigma)
    
    # Crear el kernel gaussiano bidimensional mediante la convolución del kernel 1D
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    # Convertir la imagen a un arreglo numpy
    img_array = np.asarray(inImage)

    # Convolucionar la imagen con el kernel 2D
    outImage = np.zeros_like(img_array)

    # Aplicar la convolución
    pad_width = kernel_2d.shape[0] // 2
    padded_image = np.pad(img_array, pad_width, mode='edge')

    for i in range(outImage.shape[0]):
        for j in range(outImage.shape[1]):
            outImage[i, j] = np.sum(kernel_2d * padded_image[i:i + kernel_2d.shape[0], j:j + kernel_2d.shape[1]])

    return outImage.astype(np.uint8)  # Asegurarse de que la salida sea de tipo uint8

def medianFilter(inImage, filterSize):
    # Asegurarse de que el tamaño del filtro sea impar
    if filterSize % 2 == 0:
        raise ValueError("filterSize debe ser un número impar.")
    
    # Convertir la imagen a un arreglo numpy
    img_array = np.asarray(inImage)
    outImage = np.zeros_like(img_array)

    # Calcular el radio del filtro
    pad_width = filterSize // 2
    
    # Asegurarse de que la imagen esté en el formato correcto
    padded_image = np.pad(img_array, pad_width, mode='edge')

    # Aplicar el filtro de mediana
    for i in range(outImage.shape[0]):
        for j in range(outImage.shape[1]):
            # Obtener la ventana de tamaño filterSize x filterSize
            window = padded_image[i:i + filterSize, j:j + filterSize]
            # Calcular la mediana y asignarla a la imagen de salida
            outImage[i, j] = np.median(window)

    return outImage.astype(np.uint8)  # Asegurarse de que la salida sea de tipo uint8

def erode(inImage, SE, center=[]):

    np.array(SE)
    P, Q = SE.shape

    if not center:
        center = [P // 2, Q // 2]
    
    img_array = np.asarray(inImage)
    outImage = np.zeros_like(img_array)

    pad_width = (P // 2, Q // 2)
    padded_image = np.pad(img_array, pad_width, mode='constant', constant_values=0)

    for i in range(outImage.shape[0]):
        for j in range(outImage.shape[1]):
            region = padded_image[i:i + P, j:j + Q]
            if np.all(region[SE == 1] == 1):  # Verifica si todos los píxeles en SE son 1
                outImage[i, j] = 1

    return outImage.astype(np.uint8)

def dilate(inImage, SE, center=[]):
    
    np.array(SE)
    P, Q = SE.shape
    if not center:
        center = [P // 2, Q // 2]
    
    img_array = np.asarray(inImage)
    outImage = np.zeros_like(img_array)

    pad_width = (P // 2, Q // 2)
    padded_image = np.pad(img_array, pad_width, mode='constant', constant_values=0)

    for i in range(outImage.shape[0]):
        for j in range(outImage.shape[1]):
            region = padded_image[i:i + P, j:j + Q]
            if np.any(region[SE == 1] == 1):  # Verifica si alguno de los píxeles en SE es 1
                outImage[i, j] = 1

    return outImage.astype(np.uint8)

def opening(inImage, SE, center=[]):
    
    eroded = erode(inImage, SE, center)
    opened_image = dilate(eroded, SE, center)
    return opened_image.astype(np.uint8)

def closing(inImage, SE, center=[]):

    dilated = dilate(inImage, SE, center)
    closed_image = erode(dilated, SE, center)
    return closed_image.astype(np.uint8)

def fill(inImage, seeds, SE=[], center=[]):
    
    # Si no se especifica SE, usamos conectividad 4 (una cruz 3x3)
    if not SE:
        SE = [[0, 1, 0],
              [1, 1, 1],
              [0, 1, 0]]  # Conectividad 4
    SE = np.array(SE)  # Convertir SE a numpy array para operaciones eficientes
    P, Q = SE.shape
    if not center:
        center = [P // 2, Q // 2]

    # Convertir la imagen a un array numpy
    img_array = np.array(inImage)
    filled_image = np.zeros_like(img_array)

    # Colocar los puntos semilla en la imagen de salida
    for seed in seeds:
        filled_image[seed[0], seed[1]] = 1  # Marcar los puntos semilla como parte de la región

    # Padding manual para aplicar el SE en los bordes
    pad_width = (P // 2, Q // 2)
    padded_image = np.pad(filled_image, pad_width, mode='constant', constant_values=0)

    # Iterar hasta que la región no crezca más
    while True:
        prev_image = filled_image.copy()

        # Aplicar dilatación a la imagen actual (usando el SE)
        for i in range(filled_image.shape[0]):
            for j in range(filled_image.shape[1]):
                region = padded_image[i:i + P, j:j + Q]
                if np.any(region[SE == 1] == 1) and img_array[i, j] == 1:  # Asegurarse de no salir del objeto
                    filled_image[i, j] = 1

        # Actualizar la imagen con padding
        padded_image = np.pad(filled_image, pad_width, mode='constant', constant_values=0)

        # Si no hay cambios, detener el proceso
        if np.array_equal(filled_image, prev_image):
            break

    return filled_image.astype(np.uint8)

def gradientImage(inImage, operator='Sobel'):
    
    # Definición de los kernels de cada operador
    if operator == 'Roberts':
        kernel_gx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_gy = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    elif operator == 'CentralDiff':  # Diferencias Centrales ([-1, 0, 1])
        kernel_gx = np.array([[-1, 0, 1]], dtype=np.float32)
        kernel_gy = np.transpose(kernel_gx)
    elif operator == 'Prewitt':
        kernel_gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_gy = np.transpose(kernel_gx)
    elif operator == 'Sobel':
        kernel_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_gy = np.transpose(kernel_gx)
    else:
        raise ValueError(f"Operador no válido: {operator}")

    # Aplicar la convolución usando scipy.ndimage.convolve
    gx = convolve(inImage, kernel_gx)
    gy = convolve(inImage, kernel_gy)

    return gx, gy