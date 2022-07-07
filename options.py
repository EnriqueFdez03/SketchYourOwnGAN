import os
import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathNetwork", type=str, required=True, help="directorio hacia el pkl donde está almacenada StyleGan2 junto con sus pesos")

    parser.add_argument("--pathG", type=str, default=None, help="(resume training) directorio donde se almacena los pesos de G")
    parser.add_argument("--pathDImage", type=str, default=None, help="(resume training) cada cuanto almacenar una imagen del proceso de entrenamiento")
    parser.add_argument("--pathDSketch", type=str, default=None, help="(resume training) número de imágenes en el collage de ejemplo")

    parser.add_argument("--path_sketch", type=str, required=True, help="directorio donde se encuentran los bocetos")
    parser.add_argument("--path_images", type=str, required=True, help="directorio de imágenes para llevar a cabo la regularización")

    parser.add_argument("--example_folder", type=str, required=True, help="directorio donde se almacenaran los experimentos")
    parser.add_argument("--example_each", type=int, default=500, help="cada cuantas iteraciones almacenar una imagen del proceso de entrenamiento")
    parser.add_argument("--example_elements", type=int, default=32, help="número de imágenes en el collage de ejemplo")

    parser.add_argument("--batch_size", type=int, required=True, help="tamaño del batch para el entrenamiento")
    parser.add_argument("--num_epochs", type=int, default=2000, help="máximo número de epochs para el entrenamiento")
    parser.add_argument("--max_iters", type=int, default=10000, help="máximo número de iteraciones totales para el entrenamiento")
    
    
   
    parser.add_argument("--stats_each", type=int, default=50, help="cada cuantas iteraciones mostrar en pantalla los losses")
    parser.add_argument("--save_stats_each", type=int, default=1000, help="cada cuantas iteraciones almacenar en un pkl los losses")

    parser.add_argument("--save_network_each", type=int, default=5000, help="cada cuantas iteraciones hacer un guardado de los pesos")

    
    parser.add_argument("--disable_d_reg", default=True, action='store_false', help="Desactivar regularización R1 sobre el discriminador")
    parser.add_argument("--d_reg_each", type=int, default=16, help="Cada cuántas iteraciones aplicar regularización sobre el discriminador")
    parser.add_argument("--r1", type=int, default=10, help="intensidad de la regularización r1")
    parser.add_argument("--repeat", type=int, default=0, help="repeticiones de un entrenamiento. Ver comentario train_repeat.py")
    parser.add_argument("--enable_aug", default=False, action='store_true', help="Activar aumentación de traslación sobre los bocetos")
    
    parser.add_argument("--z_vectors", type=str, default=None, help="vectores latentes para los ejemplos")
   

    opt = parser.parse_args()
    return opt, parser