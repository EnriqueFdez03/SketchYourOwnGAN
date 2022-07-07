# Tiene como objetivo generar imágenes dado un generador. La idea es usarlo
# una vez hayamos reentrenado el modelo con los bocetos tantas veces como se desee.

from PIL import Image
from model import SketchGANModel
import math
import dnnlib, legacy

from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import make_grid
from datetime import datetime
import torch,os
from networks.usePhotoSketch import PhotoSketchInfer
import glob



dir_save = "generated"

## tiene como objetivo que se observe las transiciones de los modelos
def generate_transitions(exp_dir,limit=None,stride=1,is_parent=False):
    def obtain_grids():
        if not is_parent:
            return glob.glob(os.path.join(exp_dir,"images","*.pt"))
        else:
            dirs = os.listdir(exp_dir)
            grids = []
            for dir in dirs:
                grids.extend(glob.glob(os.path.join(exp_dir,dir,"images","*.pt")))
            return grids
    grids = obtain_grids()

    if not os.path.exists(os.path.join(exp_dir,"images","transitions")) and not is_parent:
        os.mkdir(os.path.join(exp_dir,"images","transitions"))
    elif not os.path.exists(os.path.join(exp_dir,"transitions")) and is_parent:
        os.mkdir(os.path.join(exp_dir,"transitions"))
    
    grids = sorted(grids, key=lambda x: float(x.split("_")[-1].split(".")[0]))[:limit]

    acum = []
    num_imgs = None
    for imgs in grids:
        grid = torch.load(imgs)
        acum.append(grid)
        if not num_imgs:
            num_imgs = grid.size()[0]
    
    num_grids = len(acum)
    for i in range(num_imgs):
        same_z = []
        now = datetime.strftime(datetime.now(), "%a %B %d %Y %I.%M %S.%f")
        for j in range(0,num_grids,stride):
            same_z.append(acum[j][i])
        if not is_parent:
            ToPILImage()(make_grid(same_z)).save(os.path.join(exp_dir,"images","transitions", "{}.jpg".format(now)))
        else:
            ToPILImage()(make_grid(same_z)).save(os.path.join(exp_dir,"transitions", "{}.jpg".format(now)))
    

# tiene como objetivo generar imágenes con la gan preentrenada y tras haber sido 
# modificada para que el output coincida con los bocetos
def generate_pics_sketchgan(path_model,path_gen_weights,num_imgs):
    gan = SketchGANModel(path_model=path_model) # no vamos a entrenar
    z = torch.randn(num_imgs, gan.G.z_dim, device=gan.device)
    gan.G.load_state_dict(torch.load(path_gen_weights))
    imgsAfter = generate_grid(gan.postProcessImg(gan.generate_fake(z)).to(torch.uint8))
    ToPILImage()(imgsAfter).save(os.path.join(dir_save, "{}-{}.jpg".format(os.path.split("/")[-1],str(datetime.now())))) # antes a la izquierda, después a la derecha...

def generate_grid(imgs):
    nrows = int(math.sqrt(imgs.size()[0]))
    return make_grid(imgs,nrow=nrows)
    
# tiene como objetivo dada una gan preentrenada generar imágenes y si obtainSketch es 
# True, generar los sketches de photosketch
def generate_pics_initGan(path, num_imgs, obtainSketch=True):
    gan = SketchGANModel(path)
    z = torch.randn(num_imgs, gan.G.z_dim, device=gan.device)
    imgs = gan.postProcessImg(gan.generate_fake(z)).to(torch.uint8)
    grid = generate_grid(imgs)
    ToPILImage()(grid).save(os.path.join(dir_save, "{}-{}.jpg".format(path.split("/")[-1],str(datetime.now()))))
    print("Guardado imgs en " + os.path.join(dir_save, "{}-{}.jpg".format(path.split("/")[-1],str(datetime.now()))))  
    if (obtainSketch):
        sketches = gan._generate_sketch(imgs)
        grid = make_grid(generate_grid(sketches))
        ToPILImage()(grid).save(os.path.join(dir_save, "{}-{}.jpg".format(path.split("/")[-1],str(datetime.now()))))
        print("Guardado boceto en " + os.path.join(dir_save, "{}-{}.jpg".format(path.split("/")[-1],str(datetime.now())))) 

def generate_sketches(pathImgs, pathSketches):
    photosketch = PhotoSketchInfer()
    for imgDir in os.listdir(pathImgs):
        img = ToTensor()(Image.open(os.path.join(pathImgs, imgDir)))
        sketch = photosketch.obtain_sketch(img)[0]
        sketch.save(os.path.join(pathSketches,imgDir))


def stylegan_save_G_D(path):
    with dnnlib.util.open_url(path,'rb') as f:
        gan = legacy.load_network_pkl(f)
        G = gan['G_ema']
        D = gan['D']
        torch.save(G.state_dict(), os.path.join(path.split("/")[0], "G.pth"))
        torch.save(D.state_dict(), os.path.join(path.split("/")[0], "D.pth"))
        

if __name__ == "__main__":    
    #generate_pics_initGan('weights/stylegan256.pkl',1)
    #generate_sketches('imagenes/arboles',"sketches-arboles")
    #generate_pics_sketchgan('weights/arboles256.pkl',"/media/enrique/HDD/TFG/arboles-reg/arboles-estrellado/2022-07-03 22_26_08.930292/snaps/G_epoch_1000",16)
    generate_transitions('experiments/arboles-simpleDcha',stride=2, is_parent=True)
