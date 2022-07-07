import os
import shutil
from unicodedata import name

# durante las pruebas se crean muchas carpetas de experimentos.
# Busca eliminar aquellos experimentos de los cuales poco
# tiempo se ha estado ejecutando
def clean_experiments(path="experiments", num_imgs = 1):
    names = os.listdir(path)
    num = 0
    for name_experiments in names:
        experiments = os.listdir(os.path.join(path, name_experiments))
        for experiment in experiments:
            imgs_elems = len(os.listdir(os.path.join(path,name_experiments, experiment, "images")))
            if imgs_elems<=3:
                shutil.rmtree(os.path.join(path,name_experiments, experiment))
                num = num+1
    print("Se han eliminado " + str(num) + " directorios")

if __name__ == "__main__":    
    clean_experiments()