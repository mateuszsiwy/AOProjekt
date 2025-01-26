import cv2 as cv
import numpy as np
import math


SIZE = 600


def distance_transform(binary_img):
    """
    Implementacja transformaty odległościowej (L2 - Euklidesowa) według:
        https://people.cmm.minesparis.psl.eu/users/marcoteg/cv/publi_pdf/MM_refs/1986_Borgefors_distance.pdf
        dostęp: 25.01.2025
    Parameters:
        binary_img (numpy.ndarray): Obraz binarny (0 - tło, 1 - obiekt).

    Returns:
        numpy.ndarray: Mapa odległości.
    """
    
    h, w = binary_img.shape
    inf = h + w  # Maksymalna możliwa odległość (duża wartość)
    
    # Inicjalizacja mapy odległości: tło = 0, obiekt = inf
    dist = np.where(binary_img == 0, 0, inf).astype(np.float32)

    # **Forward pass** (góry dół, lewa prawa)
    for y in range(h):
        for x in range(w):
            if dist[y, x] > 0:  # Pomijamy tło (już 0)
                min_dist = dist[y, x]
                # Sprawdzamy sąsiadów z lewej strony i z góry
                if x > 0:
                    min_dist = min(min_dist, dist[y, x - 1] + 1)
                if y > 0:
                    min_dist = min(min_dist, dist[y - 1, x] + 1)
                if x > 0 and y > 0:
                    min_dist = min(min_dist, dist[y - 1, x - 1] + np.sqrt(2))
                if x < w - 1 and y > 0:
                    min_dist = min(min_dist, dist[y - 1, x + 1] + np.sqrt(2))
                dist[y, x] = min_dist

    # **Backward pass** (dół góra, prawa lewa)
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if dist[y, x] > 0:
                min_dist = dist[y, x]
                # Sprawdzamy sąsiadów z prawej strony i z dołu
                if x < w - 1:
                    min_dist = min(min_dist, dist[y, x + 1] + 1)
                if y < h - 1:
                    min_dist = min(min_dist, dist[y + 1, x] + 1)
                if x > 0 and y < h - 1:
                    min_dist = min(min_dist, dist[y + 1, x - 1] + np.sqrt(2))
                if x < w - 1 and y < h - 1:
                    min_dist = min(min_dist, dist[y + 1, x + 1] + np.sqrt(2))
                dist[y, x] = min_dist

    return dist

def watershed_segmentation(img,IS_INVERTED_FLAG=False):

    copy_img = img.copy()
    assert img is not None, "file could not be read, check with os.path.exists()"
    # zmiana na odcienie szarości
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # binaryzacja
    if(IS_INVERTED_FLAG):
        ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    else:
        ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #pozbywanie się szumu
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

    #określanie co na obrazie jest na pewno tłem
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    #użycie transformaty odległościowej aby zobaczyć jaka jest odległość 1 od najbliższego 0
    dist_transform = distance_transform(opening)

    #określanie co na obrazie jest na pewno obiektem 
    ret, sure_fg = cv.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)  
    # cv.imshow("ll",dist_transform)
    sure_fg = np.uint8(sure_fg)

    #określanie części nie określonej, czy tam gdzie coś może jest tłem a może obiektem
    unknown = cv.subtract(sure_bg,sure_fg)

    #zaznaczanie każdego obiektu z tych pewnych i podpisanie tła
    ret, markers = cv.connectedComponents(sure_fg) 
    markers = markers+1
    markers[unknown==255] = 0
    
    #główna procedura watershed używająca jako maski wcześniej podpisane markery
    height, width = markers.shape[:2]
    for y in range(height):
        for x in range(width):
            if markers[y, x] == 0:  # Tylko obszar nieokreślony
                # Określanie najbliższego "sąsiada"
                neighbors = set()
                for ny in range(max(0, y-1), min(height, y+2)):
                    for nx in range(max(0, x-1), min(width, x+2)):
                        if markers[ny, nx] > 0:  # Jeśli sąsiad to tło lub obiekt
                            neighbors.add(markers[ny, nx])
                
                if len(neighbors) == 1:
                    markers[y, x] = neighbors.pop()  # Jeśli jeden sąsiad, przypisujemy go
                else:
                    markers[y, x] = 0 

    copy_img[markers == -1] = [255,0,0]

    #wizualizacja segmentów
    markers_norm = cv.normalize(markers, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    colored_markers = cv.applyColorMap(markers_norm, cv.COLORMAP_JET)

    #tło zamieniane na biały a zawartość na czarny
    colored_markers[markers==1] = [255,255,255]
    colored_markers[markers!=1] = [0,0,0]
   
    # cv.imshow("Segmented Regions", colored_markers)

    return colored_markers

def resize_with_padding(image, target_size):
    """
    Skalowanie obrazu do zadanego rozmiaru:
    - Jeśli obraz jest za duży → skalujemy go proporcjonalnie w dół.
    - Jeśli obraz jest za mały → dodajemy białe tło do zadanych wymiarów.

    :param image: Wejściowy obraz (np. cv.imread())
    :param target_size: Docelowy rozmiar w postaci (szerokość, wysokość)
    :return: Przeskalowany obraz z ewentualnymi białymi ramkami
    """
    target_w =target_size
    target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)

    # Przeskalowanie obrazu
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)

    # Tworzymy biały obraz o docelowym rozmiarze
     # Sprawdzamy liczbę kanałów
    if len(image.shape) == 2:  # Obraz 1-kanałowy (np. po Canny)
        result = np.ones((target_h, target_w), dtype=np.uint8) * 255  # Białe tło (1-kanałowe)
    else:
        result = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255  # Białe tło (3-kanałowe)


    # Obliczamy, gdzie wkleić przeskalowany obraz (wyśrodkowanie)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Wklejamy obraz w środek
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return result

# def crop_image(image, border=10):
#     """
#     Obcina obraz o określoną liczbę pikseli z każdej krawędzi.
    
#     :param image: Wejściowy obraz (np. cv.imread())
#     :param border: Liczba pikseli do obcięcia (domyślnie 10px)
#     :return: Przycięty obraz
#     """
#     h, w = image.shape[:2]
#     if h <= 2 * border or w <= 2 * border:
#         raise ValueError("Obraz jest za mały do przycięcia.")
    
#     cropped_image = image[border:h-border, border:w-border]
#     return cropped_image

def invert_mask(image):
    image = 255 - image
    return image

# def delete_border(image):
#     image[0] = [255,255,255]
#     for i in range(1, image.shape[0]-1):
#          image[i,0] = [255,255,255]
#          image[i, image.shape[1]-1] = [255,255,255]
#     image[ image.shape[0]-1] = [255,255,255]
#     return image

def mean_border_pixel_value(image, border_thickness=4):
    """
    Oblicza średnią wartość piksela dla ramki obrazu o zadanej grubości.

    Parameters:
        image (numpy.ndarray): Obraz w skali szarości.
        border_thickness (int): Grubość ramki (domyślnie 4).

    Returns:
        float: Średnia wartość piksela w ramce.
    """
 

    top = image[:border_thickness, :]      
    bottom = image[-border_thickness:, :]  
    left = image[border_thickness:-border_thickness, :border_thickness]  
    right = image[border_thickness:-border_thickness, -border_thickness:]  

    border_pixels = np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])

    mean_value = np.mean(border_pixels)

    return mean_value

def generate_gaussian_kernel(length, weight):
    Kernel = np.zeros((length, length), dtype=np.float64)
    sumTotal = 0.0
    
    kernelRadius = length // 2
    calculatedEuler = 1.0 / (2.0 * math.pi * (weight ** 2))
    
    for filterY in range(-kernelRadius, kernelRadius + 1):
        for filterX in range(-kernelRadius, kernelRadius + 1):
            distance = ((filterX ** 2) + (filterY ** 2)) / (2 * (weight ** 2))
            Kernel[filterY + kernelRadius, filterX + kernelRadius] = calculatedEuler * math.exp(-distance)
            sumTotal += Kernel[filterY + kernelRadius, filterX + kernelRadius]
    
    Kernel /= sumTotal
    
    return Kernel

def gaussian_filter(image,kernel_size,sigma):
    gaussian_kernel = generate_gaussian_kernel(kernel_size, sigma)
    blurred_image = cv.filter2D(image, -1, gaussian_kernel)
    return blurred_image

# def blur_edges(image, kernel_size=15, sigma=5):
#     """
#     Aplikuje efekt rozmycia tylko na krawędziach obrazu.
    
#     :param image: Wejściowy obraz (np. cv.imread())
#     :param kernel_size: Rozmiar jądra Gaussa
#     :param sigma: Odchylenie standardowe dla filtra Gaussa
#     :return: Obraz z rozmytymi krawędziami
#     """
#     blurred = gaussian_filter(image, kernel_size, sigma)
#     mask = np.zeros_like(image, dtype=np.uint8)
#     border = kernel_size // 2
    
#     mask[:border, :] = 255  # Górna krawędź
#     mask[-border:, :] = 255  # Dolna krawędź
#     mask[:, :border] = 255  # Lewa krawędź
#     mask[:, -border:] = 255  # Prawa krawędź
    
#     result = np.where(mask == 255, blurred, image)
#     return result

def grad_x_calculation(image):
    sobelX_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    grad_x = cv.filter2D(image, cv.CV_64F, sobelX_kernel)
    return grad_x

def grad_y_calculation(image):
    sobelY_kernel = np.array([
        [1,  2,  1],
        [0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float64)
    
    grad_y = cv.filter2D(image, cv.CV_64F, sobelY_kernel)
    return grad_y

# def normalize_image(image):
#     min_val, max_val = np.min(image), np.max(image)
#     if max_val > min_val:
#         image = (image - min_val) / (max_val - min_val) * 255
#     return image.astype(np.uint8)

def calculate_orientation(grad_x,grad_y):
    orientation = np.arctan2(grad_y, grad_x)  
    orientation = np.rad2deg(orientation) 
    orientation = (orientation + 180) % 180 
    return orientation

def calculate_magnitude(grad_x,grad_y):
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    min_val = np.min(magnitude)
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = ((magnitude - min_val) / (max_val - min_val)) * 255
    else:
        magnitude = np.zeros_like(magnitude)
    return magnitude

def non_maximum_suppression(magnitude, orientation):
    H, W = magnitude.shape
    suppressed = np.zeros((H, W), dtype=np.float32)

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            angle = orientation[y, x]

            q, r = 255, 255
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = magnitude[y, x + 1]
                r = magnitude[y, x - 1]
            elif (22.5 <= angle < 67.5):
                q = magnitude[y - 1, x + 1]
                r = magnitude[y + 1, x - 1]
            elif (67.5 <= angle < 112.5):
                q = magnitude[y - 1, x]
                r = magnitude[y + 1, x]
            elif (112.5 <= angle < 157.5):
                q = magnitude[y - 1, x - 1]
                r = magnitude[y + 1, x + 1]

            if magnitude[y, x] >= q and magnitude[y, x] >= r:
                suppressed[y, x] = magnitude[y, x]
            else:
                suppressed[y, x] = 0

    return suppressed

def double_threshold(image, low_thresh, high_thresh):
    strong = 255
    weak = 60
    result = np.zeros_like(image, dtype=np.uint8)

    strong_y, strong_x = np.where(image >= high_thresh)
    weak_y, weak_x = np.where((image >= low_thresh) & (image < high_thresh))

    result[strong_y, strong_x] = strong
    result[weak_y, weak_x] = weak

    return result

def hysteresis(image):
    H, W = image.shape
    strong = 255
    weak = 60
    #sprawdzanie 8 sąsiadów danego piksela
    copy = image.copy()
    for y in range(1, H-1):
        for x in range(1, W-1):
        
            if image[y, x] == weak:
                if np.any(image[y-1:y+2, x-1:x+2] == strong):
                    copy[y, x] = strong  # zamiana na silną krawędź
                else:
                    copy[y, x] = 0  
    image = copy
    return image

def canny(image,low_thresh,high_thresh):
    image = gaussian_filter(image,7,5)
    # copy = image.copy()
    grad_x = grad_x_calculation(image)
    grad_y = grad_y_calculation(image)
    # grad_x = normalize_image(grad_x)
    # grad_y = normalize_image(grad_y)
    # cv.imshow("s",grad_y)
    magnitude = calculate_magnitude(grad_x,grad_y)
    orientation = calculate_orientation(grad_x,grad_y)
    # # print(orientation)
    image = non_maximum_suppression(magnitude,orientation)
    image = double_threshold(image,low_thresh,high_thresh)
    image = hysteresis(image)
    return image

def edge_detection_mask(image,IS_INVERTED_FLAG=False):
    print("Started edge detection")
    low = 20
    high = 60
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    image = canny(image,low,high)
    image = invert_mask(image)

    image = resize_with_padding(image,SIZE)
    image[image<230] = 0    
    image[image>=230] = 255

    kernel = np.ones((7, 7), np.uint8) 
    image = cv.erode(image, kernel, iterations=2)

    if(IS_INVERTED_FLAG):
        image = invert_mask(image)

    return image

def watershed_mask(image,is_inverted_before,is_inverted_after):
    print("Started watershed")
    image = watershed_segmentation(image,is_inverted_before)
    mean = mean_border_pixel_value(image)
    if(mean > 127):
        image = resize_with_padding(image,SIZE)
    else:
        image = invert_mask(image)
        image = resize_with_padding(image,SIZE)
        image = invert_mask(image)
    
    image[image<230] = 0    
    image[image>=230] = 255
    
    kernel = np.ones((3, 3), np.uint8) 
    if(is_inverted_before):
        image = cv.erode(image, kernel, iterations=2)
    else:
        image = cv.dilate(image, kernel, iterations=2)

    if(is_inverted_after):
        image= invert_mask(image)

    return image
if __name__ == "__main__":
    is_inverted_flag =False
    is_inverted_flag = True
    image = cv.imread("src/koszulka2.jpg")

    # image = watershed_mask(image,is_inverted_flag,False) # czy przed zaczeciem odwracamy maske, a druga flaga to czy rezultat odwracamy
    # image = invert_mask(image)
    # image = edge_detection_mask(image,is_inverted_flag) # ta flaga tylko odpowiada za rezultat

    cv.imshow("Segmented Res",image)

    cv.waitKey(0)
    cv.destroyAllWindows()