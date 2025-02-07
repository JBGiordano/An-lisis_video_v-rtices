# README - Análisis de Velocidades en Video

Este código fue hecho para la práctica de Fluidos de la materia Laboratorio 5 de la facultad FCEyN UBA. La experiencia propone generar un vórtice en un fluido y caracterizar su campo de velocidades mediante análisis de video, utilizando glitter como trazador. Este código analiza el comportamiento de dichas partículas detectándolas en cada frame, calculando su desplazamiento y obteniendo los gráficos de las velocidades en función del radio, filtrando valores atípicos.


## Funcionamiento del Código


1. **Filtrado de imagen**: Se aplica un método de umbralización para detectar partículas dividiendo la imagen en dos regiones (superior e inferior) usando `filtro()`. 
2. **Detección de partículas**: Se identifican los puntos de mayor intensidad en la imagen con `identify_and_mark_blue_dots()`, dentro del rango dado.
3. **Cálculo del centro dinámico**: Se determina el centro de la región de interés en cada frame con `calculate_dynamic_center()`. 
4. **Cálculo de desplazamientos**: Se mide el desplazamiento de cada partícula en frames consecutivos con `calculate_displacement()`.
5. **Cálculo de velocidades**: Se obtienen dividiendo el desplazamiento por el tiempo entre frames (`calculate_velocities()`). Además, esta se descompone en sus componentes tangencial y radial
6. **Agrupación por radio**: Se agrupan las velocidades según su distancia al centro dinámico en anillos de ancho `epsilon` píxeles, calculando la media y la desviación estándar `sigma` en dichas regiones.
7. **Filtrado de valores atípicos**: Se eliminan velocidades mayores a `media + 3*sigma`, las cuales pueden deberse a reflejos no deseados o imperfecciones.
8. **Conversión de unidades**: Detectando el borde del recipiente (`detect_circle_edges()`) y conociendo su tamaño se obtiene el factor de conversión de píxeles a metros (`calibrate_pixel_to_meter()`).

## Parámetros 

- `video`: Ruta del video a analizar.
- `radius`: Radio de detección de partículas. Se obtiene con la distancia del vórtice al borde más cercano.
- `initial_center_x, initial_center_y`: Coordenadas iniciales del centro dinámico. Por lo general termina siendo casi constante.
- `min_intensity_top, min_intensity_bottom`: Umbrales de intensidad mínima para detección de partículas.
- `dt`: Intervalo de tiempo entre frames (según FPS del video).
- `epsilon`: Tamaño de los anillos radiales para calcular las velocidades.


## Notas Importantes
- Los parámetros `radius,initial_center_x, initial_center_y` se calculan manualmente con la imagen generada del primer frame analizado.
- Los valores de `min_intensity_top` y `min_intensity_bottom` necesitan ajustarse según la iluminación de cada video y la cantidad de trazador de la muestra.
- La detección de bordes del recipiente puede verse afectada por la perspectiva del video. Esto debido a que mide únicamente el borde del recipiente y, por lo general, la superficie del fluido llega a mucho menos de la mitad de este. Ver como solucionar. Se podría medir para cada muestra el radio de la suerficie, y modificar el código para que tome este como borde.
- Lo más importante de todo termina siendo medir bien. Procurar de grabar, si es con un celular, con flash prendido y cubriendo todo el recipiente para que no haya otro tipo de luz externa.





