import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import unsharp_mask
import os

def get_frame(filename, index):
    counter = 0
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if counter == index:
                video.release()
                return frame
            counter += 1
        else:
            break
    video.release()
    return None

def filtro(frame, min_intensity_top, min_intensity_bottom, max_intensity, show_image=False):
    # Convert to grayscale
    im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply unsharp mask
    unsharp_response = unsharp_mask(im_gray)
    
    # Normalize the response to range [0, 1]
    unsharp_response_normalized = cv2.normalize(unsharp_response, None, 0, 1, cv2.NORM_MINMAX)
    
    # Get the image height
    height = unsharp_response_normalized.shape[0]
    
    # Divide the image into top and bottom halves
    top_half = unsharp_response_normalized[:height // 2, :]
    bottom_half = unsharp_response_normalized[height // 2:, :]
    
    # Apply the threshold to each half
    _, thresholded_top = cv2.threshold(top_half * 255, min_intensity_top * 255, max_intensity * 255, cv2.THRESH_BINARY)
    _, thresholded_bottom = cv2.threshold(bottom_half * 255, min_intensity_bottom * 255, max_intensity * 255, cv2.THRESH_BINARY)
    
    # Combine the two halves back into one image
    thresholded_image = np.vstack((thresholded_top, thresholded_bottom))
    
    # Convert to uint8 for contour detection
    thresholded_image = thresholded_image.astype(np.uint8)
    
    if show_image:
        plt.figure(figsize=(10, 5))
        plt.imshow(thresholded_image, cmap='gray')
        plt.title('Unsharp Mask Response with Top and Bottom Thresholds')
        plt.axis('off')
        plt.show()
    
    return thresholded_image

def identify_and_mark_blue_dots(image, radius, center_x, center_y,min_intensity_top, min_intensity_botton, max_intensities, show_image=False):
    # Procesar la imagen para obtener el canal azul y binarizarla por cuadrantes
    thresholded_image = filtro(image,min_intensity_top, min_intensity_botton, max_intensities, show_image)
    
    # Encontrar contornos en la imagen binarizada
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blue_dots = []
    output_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)

    # Procesar cada contorno encontrado
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            distance = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
            # Verificar si el punto está dentro del radio especificado
            if distance <= radius:
                blue_dots.append((cx, cy))
                cv2.circle(output_image, (cx, cy), 10, (0, 255, 0), -1)  # Marcado en verde

    if show_image:
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title('Puntos Azules Marcados')
        plt.show()

    return blue_dots

def calculate_dynamic_center(blue_dots, initial_center, search_radius=20):
    # Inicializar el centro dinámico
    center_x, center_y = initial_center
    if not blue_dots:
        return center_x, center_y  # Retorna el centro actual si no hay puntos azules

    # Buscar el punto azul más cercano al centro actual
    closest_dot = min(blue_dots, key=lambda dot: np.sqrt((dot[0] - center_x) ** 2 + (dot[1] - center_y) ** 2))
    
    # Verificar si el punto azul más cercano está dentro del radio de búsqueda
    distance = np.sqrt((closest_dot[0] - center_x) ** 2 + (closest_dot[1] - center_y) ** 2)
    if distance <= search_radius:
        return closest_dot  # Retorna el punto más cercano como el nuevo centro dinámico
    
    return center_x, center_y  # Retorna el centro actual si no se encuentra un punto cercano

def calculate_displacement(frames, radius, min_intensity_top,min_intensity_botton, max_intensities, initial_center):
    displacement_data = []
    blue_dots_data = []
    dynamic_centers = []
    
    # Inicializar el centro dinámico con el centro inicial proporcionado
    current_center = initial_center
    
    for i in range(len(frames) - 1):
        # Procesar los frames para obtener los puntos azules, utilizando el centro dinámico actual
        blue_dots1 = identify_and_mark_blue_dots(frames[i], radius, current_center[0], current_center[1], min_intensity_top,min_intensity_botton, max_intensities)
        # Calcular el centro dinámico para el frame actual
        current_center = calculate_dynamic_center(blue_dots1, current_center)

        # Usar el nuevo centro dinámico para el siguiente frame
        blue_dots2 = identify_and_mark_blue_dots(frames[i + 1], radius, current_center[0], current_center[1], min_intensity_top,min_intensity_botton, max_intensities)
        
        # Calcular el centro dinámico para el siguiente frame
        new_center = calculate_dynamic_center(blue_dots2, current_center)
        
        # Actualizar el centro dinámico para el siguiente frame
        dynamic_centers.append(current_center)
        current_center = new_center
        
        displacements = []
        
        for dot1 in blue_dots1:
            # Encuentra el punto azul más cercano en el segundo frame
            closest_dot2 = min(blue_dots2, key=lambda dot2: np.sqrt((dot2[0] - dot1[0]) ** 2 + (dot2[1] - dot1[1]) ** 2))
            dx = closest_dot2[0] - dot1[0]
            dy = closest_dot2[1] - dot1[1]
            displacements.append((dx, dy))
        
        displacement_data.append(displacements)
        blue_dots_data.append(blue_dots1)
    
    return displacement_data, blue_dots_data, dynamic_centers


def calculate_velocities(displacement_data, dt):
    velocities = []
    
    for displacements in displacement_data:
        velocity = [(dx / dt, dy / dt) for dx, dy in displacements]
        velocities.append(velocity)
    
    return velocities

def calculate_radial_velocities(velocities, blue_dots_data, dynamic_centers):
    radial_velocities = []
    
    for i in range(len(velocities)):
        radial_velocity = []
        center_x, center_y = dynamic_centers[i]
        for j, (vx, vy) in enumerate(velocities[i]):
            x, y = blue_dots_data[i][j]
            rx, ry = x - center_x, y - center_y
            r = np.sqrt(rx**2 + ry**2)
            if r == 0:
                vr = 0
                vtheta = 0
            else:
                vr = (vx * rx + vy * ry) / r
                vtheta = (vx * ry - vy * rx) / r
            radial_velocity.append((vr, vtheta))
        radial_velocities.append(radial_velocity)
    
    return radial_velocities

def average_tangential_velocity_by_radius(radial_velocities, blue_dots_data, center_x, center_y, epsilon=5):
    radius_bins = {}
    
    for i in range(len(radial_velocities)):
        for j, (_, vtheta) in enumerate(radial_velocities[i]):
            x, y = blue_dots_data[i][j]  # Usa solo x e y
            r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            
            # Encuentra el bin adecuado para el radio
            bin_key = round(r / epsilon) * epsilon
            
            if bin_key not in radius_bins:
                radius_bins[bin_key] = []
            
            radius_bins[bin_key].append(vtheta)
    
    # Calcular la velocidad tangencial promedio para cada bin de radio
    avg_tangential_velocities = {}
    for r_bin, vtheta_list in radius_bins.items():
        avg_tangential_velocities[r_bin] = np.mean(vtheta_list)
    
    return avg_tangential_velocities

def average_filtered_tangential_velocity_by_radius(radial_velocities, blue_dots_data, dynamic_centers, epsilon=5, threshold_factor=3.0):
    radius_bins = {}
    magnitudes = []

    # Collect tangential velocities and their magnitudes
    for i in range(len(radial_velocities)):
        center_x, center_y = dynamic_centers[i]  # Obtener el centro dinámico para el frame actual

        for j, (_, vtheta) in enumerate(radial_velocities[i]):
            x, y = blue_dots_data[i][j]
            r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            if vtheta > 0:
                magnitude = vtheta
                magnitudes.append(magnitude)

                bin_key = round(r / epsilon) * epsilon

                if bin_key not in radius_bins:
                    radius_bins[bin_key] = []

                radius_bins[bin_key].append(vtheta)

    # Calcular el umbral para filtrar
    if magnitudes:
        mean_magnitude = np.mean(magnitudes)
        std_dev_magnitude = np.std(magnitudes)
        threshold = mean_magnitude + threshold_factor * std_dev_magnitude
    else:
        threshold = 0

    # Filtrar las velocidades tangenciales según el umbral
    filtered_radius_bins = {}
    for bin_key, vtheta_list in radius_bins.items():
        filtered_list = [v for v in vtheta_list if v <= threshold]
        if filtered_list:  # Solo agregar listas no vacías
            filtered_radius_bins[bin_key] = filtered_list

    # Calcular promedios y errores
    avg_tangential_velocities = {}
    tangential_velocity_errors = {}

    for r_bin, vtheta_list in filtered_radius_bins.items():
        avg_tangential_velocities[r_bin] = np.mean(vtheta_list)
        tangential_velocity_errors[r_bin] = np.std(vtheta_list, ddof=1) / np.sqrt(len(vtheta_list))  # SEM

    return avg_tangential_velocities, tangential_velocity_errors

def plot_avg_tangential_velocity_vs_radius(avg_tangential_velocities, tangential_velocity_errors):
    radii = sorted(avg_tangential_velocities.keys())
    avg_vtheta = [avg_tangential_velocities[r] for r in radii]
    errors = [tangential_velocity_errors.get(r, 0) for r in radii]
    
    plt.figure()
    plt.errorbar(radii, np.abs(avg_vtheta)/np.max(np.abs(avg_vtheta)), yerr=errors/np.max(np.abs(avg_vtheta)), fmt='o', linestyle="none", capsize=5, capthick=1, elinewidth=1, color='b')
    plt.xlabel(r'Radio $r$',fontsize = 16)
    plt.ylabel(r'$v_{\theta}/v_{\theta}^{max}$',fontsize = 16)
    plt.title('Velocidad Tangencial Promedio Vs Radio')
    plt.grid(True)
    plt.show()
    
def average_filtered_radial_velocity_by_radius(radial_velocities, red_dots_data, dynamic_centers, epsilon=5, threshold_factor=3.0):
    radius_bins = {}
    magnitudes = []

    # Collect radial velocities and their magnitudes
    for i in range(len(radial_velocities)):
        center_x, center_y = dynamic_centers[i]  # Obtener el centro dinámico para el frame actual

        for j, (vr, _) in enumerate(radial_velocities[i]):
            x, y = red_dots_data[i][j]
            r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            if vr < 0:
                magnitude = vr
                magnitudes.append(magnitude)

                bin_key = round(r / epsilon) * epsilon

                if bin_key not in radius_bins:
                    radius_bins[bin_key] = []

                radius_bins[bin_key].append(vr)

    # Calcular el umbral para filtrar
    if magnitudes:
        mean_magnitude = np.mean(magnitudes)
        std_dev_magnitude = np.std(magnitudes)
        threshold = mean_magnitude + threshold_factor * std_dev_magnitude
    else:
        threshold = 0

    # Filtrar las velocidades radiales según el umbral
    filtered_radius_bins = {}
    for bin_key, vr_list in radius_bins.items():
        filtered_list = [vr for vr in vr_list if vr >= -threshold]  # Dado que vr es negativo, usa -threshold
        if filtered_list:  # Solo agregar listas no vacías
            filtered_radius_bins[bin_key] = filtered_list

    # Calcular promedios y errores
    avg_radial_velocities = {}
    radial_velocity_errors = {}

    for r_bin, vr_list in filtered_radius_bins.items():
        avg_radial_velocities[r_bin] = np.mean(vr_list)
        radial_velocity_errors[r_bin] = np.std(vr_list, ddof=1) / np.sqrt(len(vr_list))  # SEM

    return avg_radial_velocities, radial_velocity_errors

def plot_avg_radial_velocity_vs_radius(avg_radial_velocities, radial_velocity_errors):
    radii = sorted(avg_radial_velocities.keys())
    avg_vr = [avg_radial_velocities[r] for r in radii]
    errors = [radial_velocity_errors[r] for r in radii]
    
    plt.figure()
    plt.errorbar(radii, avg_vr/np.abs(np.max(avg_vr)), yerr=errors/np.abs(np.max(avg_vr)), fmt='o', linestyle="none", capsize=5, capthick=1, elinewidth=1, color='r')
    plt.xlabel('Radio $r$')
    plt.ylabel(r'$v_{r}/v_{\theta}^{max}$',fontsize = 16)
    plt.title('Velocidad Radial Promedio Vs Radio',fontsize = 16)
    plt.grid(True)
    plt.show()
    
def detect_circle_edges(image, show_image=False):
    """
    Detecta los bordes del círculo en la imagen utilizando el filtro de Sobel y devuelve
    la coordenada de los bordes izquierdo y derecho del círculo.
    """
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro Sobel para detectar bordes horizontales
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel_x = np.abs(sobel_x)
    edges = np.uint8(abs_sobel_x)

    # Umbral para obtener bordes binarios
    _, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    if show_image:
        plt.imshow(binary_edges, cmap='gray')
        plt.title('Bordes del círculo')
        plt.show()

    # Encontrar los bordes del círculo
    edges_positions = np.where(binary_edges[gray_image.shape[0] // 2] == 255)[0]

    if len(edges_positions) >= 2:
        left_edge = edges_positions[0]
        right_edge = edges_positions[-1]
    else:
        left_edge = None
        right_edge = None

    return left_edge, right_edge

def calibrate_pixel_to_meter(left_edge, right_edge, known_diameter_meters):
    """
    Calcula la relación de conversión de píxeles a metros utilizando la distancia conocida
    en metros del diámetro del círculo.
    """
    if left_edge is not None and right_edge is not None:
        pixel_distance = right_edge - left_edge
        pixel_to_meter_ratio = known_diameter_meters / pixel_distance
        return pixel_to_meter_ratio
    else:
        print("No se pudieron detectar correctamente los bordes del círculo.")
        return None

#%%

'''
Consideraciones:
    - Rango de los frames ponerlo a conveniencia, algunos videos pueden ser muy pesados si se graban con mucho reflejo.
    - Radio y centro lo sacamos a mano. El radio es hasta la zona que toma puntos, por lo general el borde mas cercano si el 
    vórtice no está en el medio.
    - Intensidades mínimas que detecta como partículas, para las partes superior e inferior del video. Ir jugando.
    - En dt poner los fps a los que se graba el video
    - Epsilon es la cantidad de píxeles que va a contener cada radio para los gráficos. Toma para hacer los gráficos anillos
    de 'r+epsilon'.
    
    min_intensity_botton = 0.6
    min_intensity_top = 0.4
'''

video = r"/home/juan/Documents/Laboratorio 5/videos 3/70_82.mp4"

im = get_frame(video, 20)
#%% Parámetros

radius = 319
initial_center_x = 931
initial_center_y = 522

min_intensity_botton = 0.6
min_intensity_top = 0.7
max_intensity = 1
dt = 1 / 60
epsilon = 3
#%% Primer frame para obtener el centro

plt.imshow(im)
plt.title('Frame Original')
plt.show()
#%% Calibración de los umbrales

filtro(im, min_intensity_top, min_intensity_botton, max_intensity,True) 
#%%
frames = [get_frame(video, i) for i in range(20,220)]

initial_center = (initial_center_x, initial_center_y)
displacement_data, blue_dots_data, dynamic_centers = calculate_displacement(frames, radius, min_intensity_top,min_intensity_botton, max_intensity, initial_center)

velocities = calculate_velocities(displacement_data, dt)
radial_velocities = calculate_radial_velocities(velocities, blue_dots_data, dynamic_centers)

# Calcular velocidad tangencial promedio por radio (ya filtrada)
avg_tangential_velocities, tangential_velocity_errors = average_filtered_tangential_velocity_by_radius(radial_velocities, blue_dots_data, dynamic_centers, epsilon)

# Plotear velocidad tangencial promedio vs. radio
plot_avg_tangential_velocity_vs_radius(avg_tangential_velocities, tangential_velocity_errors)

# Calcular velocidad radial promedio filtrada por radio
avg_radial_velocities, radial_velocity_errors = average_filtered_radial_velocity_by_radius(radial_velocities, blue_dots_data, dynamic_centers, epsilon)

# Plotear velocidad radial promedio vs. radio
plot_avg_radial_velocity_vs_radius(avg_radial_velocities, radial_velocity_errors)

#%%
'''
Detecto los bordes del círculo (el tupper) para obtener la escala.
Cuidado: detecta el círculo, no el interior. Por perspectiva deja mal la escala. Esto debido a que
todos los líquidos tienen volúmenes diferentes, entonces la altura de su superficie es diferente.
Lo que hicimos fue aproximar una altura de la superficie, y calibrar en función de esa viendo cuanto 
ocupaban las paredes del cilindro. Mejorar de ser posible.
'''
left_edge, right_edge = detect_circle_edges(im, show_image=True)

# Calibrar la conversión de píxeles a metros
known_diameter_meters = 0.238  # Cambia esto por el diámetro real en metros del círculo 
pixel_to_meter_ratio = calibrate_pixel_to_meter(left_edge, right_edge, known_diameter_meters)*100 #(en centimetros)

#%%

'''
Velocidades normalizadas en arrays para guardar
'''

radii_t = sorted(avg_tangential_velocities.keys())

avg_vtheta = np.array([avg_tangential_velocities[r] for r in radii_t])
avg_vtheta_norm = avg_vtheta/np.max(avg_vtheta)

errorss_t = np.array([tangential_velocity_errors[r] for r in radii_t])
errorss_t_norm = errorss_t/np.max(avg_vtheta)

radii_r = sorted(avg_radial_velocities.keys())

avg_vrad = np.array([avg_radial_velocities[r] for r in radii_r])
avg_vtrad_norm = avg_vrad/np.max(avg_vrad)

errorss_r = np.array([radial_velocity_errors[r] for r in radii_r])
errorss_r_norm = errorss_r/np.max(avg_vrad)
#%% 
'''
Guarda en un .csv los datos (radios, velocidades, errores)

Para que este en un mismo .csv relleno con vacíos si no tienen la misma
cantidad de puntos.
''' 
# Convertir a float antes de rellenar con NaN
def pad_with_nan(arr, length):
    arr = np.array(arr, dtype=float)  # Convertir a float
    return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)

# Determinar la longitud máxima
max_len = max(len(radii_t), len(radii_r))

# Rellenar los arrays
radii_t = pad_with_nan(radii_t, max_len)
avg_vtheta_norm = pad_with_nan(avg_vtheta_norm, max_len)
errorss_t_norm = pad_with_nan(errorss_t_norm, max_len)

radii_r = pad_with_nan(radii_r, max_len)
avg_vtrad_norm = pad_with_nan(avg_vtrad_norm, max_len)
errorss_r_norm = pad_with_nan(errorss_r_norm, max_len)


# Apilar y guardar en CSV
data = np.column_stack((radii_t, avg_vtheta_norm, errorss_t_norm, 
                        radii_r, avg_vtrad_norm, errorss_r_norm))
#%%
'''
cambio de carpeta a la que quiero guardar
Añado al txt las velocidades máximas y el factor de conversión como comentario al principio.
'''

os.chdir('/home/juan/Documents/Laboratorio 5/videos 2/20%') #carpeta en la que se guarda
np.savetxt("20_1.csv", data, delimiter=",",
           header="radii_t // avg_vtheta_norm // errorss_t_norm // radii_r // avg_vtrad_norm // errorss_r_norm", 
           comments=f'ratio: {pixel_to_meter_ratio} cm/px \n max_vtheta:{np.max(avg_vtheta)}\n max_vratio: {np.max(avg_vrad)}\n ', fmt='%.4f')


#%% para importar
data = np.loadtxt('40_1.csv', delimiter = ',', skiprows = 5)








