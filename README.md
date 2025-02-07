# README - Análisis de Velocidades en Video

## Descripción del Proyecto
Este código analiza el movimiento de partículas en un fluido a partir de un video. Se detectan partículas en cada frame, se calcula su desplazamiento y se obtiene su velocidad en función del radio. Además, se filtran valores atípicos y se normalizan los resultados para facilitar la comparación entre distintos experimentos.

## Funcionamiento del Código

1. **Carga del video**: Se extraen los frames del video para su análisis.
2. **Filtrado de imagen**: Se aplica un método de umbralización para detectar partículas en dos regiones de la imagen (superior e inferior) usando `filtro()`.
3. **Detección de partículas**: Se identifican los puntos de mayor intensidad en la imagen con `identify_and_mark_blue_dots()`.
4. **Cálculo del centro dinámico**: Se determina el centro de la región de interés en cada frame con `calculate_dynamic_center()`.
5. **Cálculo de desplazamientos**: Se mide el desplazamiento de cada partícula en frames consecutivos con `calculate_displacement()`.
6. **Cálculo de velocidades**:
   - Se obtiene la velocidad dividiendo el desplazamiento por el tiempo entre frames (`calculate_velocities()`).
   - Se descompone en componentes radial y tangencial (`calculate_radial_velocities()`).
7. **Agrupación por radio**:
   - Se agrupan las velocidades según su distancia al centro dinámico en intervalos de `epsilon` píxeles.
   - Se calcula la media y la desviación estándar.
8. **Filtrado de valores atípicos**:
   - Se eliminan velocidades mayores a `media + 3*sigma`.
   - Se calcula el error estándar de la media (SEM) para cada bin radial.
9. **Conversión de unidades**:
   - Se detecta el borde del recipiente (`detect_circle_edges()`).
   - Se obtiene el factor de conversión de píxeles a metros (`calibrate_pixel_to_meter()`).
10. **Exportación de resultados**:
    - Se guardan los datos en un archivo CSV para su análisis posterior.

## Uso del Código

1. **Configurar los parámetros** en la sección de configuración:
   - `video`: Ruta del video a analizar.
   - `radius`: Radio de detección de partículas.
   - `initial_center_x, initial_center_y`: Coordenadas iniciales del centro dinámico.
   - `min_intensity_top, min_intensity_bottom`: Umbrales de intensidad para detección de partículas.
   - `dt`: Intervalo de tiempo entre frames (según FPS del video).
   - `epsilon`: Tamaño de los anillos radiales para el análisis.
2. **Ejecutar el script** y revisar los gráficos generados.
3. **Guardar los resultados** en un CSV para su análisis posterior.

## Notas Importantes
- Los valores de `min_intensity_top` y `min_intensity_bottom` pueden necesitar ajuste según el video.
- La detección de bordes del recipiente puede verse afectada por la perspectiva.
- Se recomienda probar con distintos valores de `epsilon` para mejorar la resolución de los gráficos.

## Ejemplo de Uso
```python
video = "ruta/al/video.mp4"
frames = [get_frame(video, i) for i in range(10, 200)]
displacement_data, blue_dots_data, dynamic_centers = calculate_displacement(frames, radius, min_intensity_top, min_intensity_bottom, max_intensity, initial_center)
velocities = calculate_velocities(displacement_data, dt)
radial_velocities = calculate_radial_velocities(velocities, blue_dots_data, dynamic_centers)
avg_tangential_velocities, tangential_velocity_errors = average_filtered_tangential_velocity_by_radius(radial_velocities, blue_dots_data, dynamic_centers, epsilon)
plot_avg_tangential_velocity_vs_radius(avg_tangential_velocities, tangential_velocity_errors)
```


