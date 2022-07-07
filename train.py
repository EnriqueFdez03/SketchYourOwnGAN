from model import SketchGANModel
from dataloader import create_dataloader,yield_data
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import make_grid
from torchvision import transforms

import os

import torch, time, os, math
from datetime import datetime, timedelta
import pickle
from options import get_opt

from generate import generate_transitions
from PIL import Image


def training_loop():
    opt,parser = get_opt()
    
    def saveNetwork(model,path_sketch,time,epoch,f):
        torch.save(model.G.state_dict(), os.path.join(opt.example_folder,path_sketch.split("/")[1], time, "snaps", "G_epoch_{}".format(epoch)))
        torch.save(model.D_sketch.state_dict(), os.path.join(opt.example_folder,path_sketch.split("/")[1], time, "snaps", "D_sketch_epoch_{}".format(epoch)))
        torch.save(model.D_image.state_dict(), os.path.join(opt.example_folder,path_sketch.split("/")[1], time, "snaps", "D_image_epoch_{}".format(epoch)))
        log(f,"Guardando las redes... Hecho")
    
    def saveStats(model,path_sketch,time,f):
        g_losses = open(os.path.join(opt.example_folder,path_sketch.split("/")[1],time,"g_losses.pkl"), "wb")
        pickle.dump(model.g_losses, g_losses)
        g_losses.close()
        d_losses = open(os.path.join(opt.example_folder,path_sketch.split("/")[1],time,"d_losses.pkl"), "wb")
        pickle.dump(model.d_losses, d_losses)
        d_losses.close()
        log(f,"Guardando losses... Hecho")
  
    def showStats(G_loss,D_loss,D_reg_loss,epoch,iters,iter_end,start_training,f):
        if not D_reg_loss:
            log(f,"({}/{} epoch, {} iters) G_sketch: {:.5f} G_image: {:.5f} D_fake_sketch: {:.5f} D_real_sketch: {:.5f} D_fake_image: {:.5f} D_real_image: {:.5f} Time: {} \n"
            .format(epoch,opt.num_epochs,iters,G_loss["G_loss_fakeSketch"],G_loss["G_loss_fakeImg"],D_loss["D_loss_fakeSketch"],
                D_loss["D_loss_realSketch"],D_loss["D_loss_fakeImg"],D_loss["D_loss_realImg"],str(timedelta(seconds=iter_end-start_training))))
        else:
            log(f,"({}/{} epoch, {} iters) G_sketch: {:.5f} G_image: {:.5f} D_fake_sketch: {:.5f} D_real_sketch: {:.5f} D_fake_image: {:.5f} D_real_image: {:.5f} r1_loss_sketch: {:.5f} r1_loss_image: {:.5f} r1_loss: {:.5f} Time: {} \n"
            .format(epoch,opt.num_epochs,iters,G_loss["G_loss_fakeSketch"],G_loss["G_loss_fakeImg"],D_loss["D_loss_fakeSketch"],
                D_loss["D_loss_realSketch"],D_loss["D_loss_fakeImg"],D_loss["D_loss_realImg"],D_reg_loss["r1_loss_sketch"],D_reg_loss["r1_loss_imgs"],D_reg_loss["r1_loss"],str(timedelta(seconds=iter_end-start_training))))

    assert opt.example_elements%2 == 0, "El número de ejemplos debe ser par"

    # Nota: El bucle interno de cada epoch recorre en minibatches los user sketches, como son muy
    # pocos 1-8 los epochs suceden rápidamente
    torch.backends.cudnn.benchmark = True

    pathSketches = opt.path_sketch.split(",") # se pueden encadenar entrenamientos
    for path_sketch in pathSketches:
        # gestión  de ficheros para almacenar experimentos.
        now = datetime.strftime(datetime.now(), "%a %B %d %Y %I.%M %S.%f")
        os.makedirs(os.path.join(opt.example_folder,path_sketch.split("/")[1], now))
        os.mkdir(os.path.join(opt.example_folder,path_sketch.split("/")[1], now, "images")) # creamos carpeta donde se guardarán imagenes 
        os.mkdir(os.path.join(opt.example_folder,path_sketch.split("/")[1], now, "snaps")) # creamos la carpeta donde se guardaran los generadores y disc
        f = open(os.path.join(opt.example_folder,path_sketch.split("/")[1], now,"log.log"), "a")
        log(f,"Configuración: {}".format(vars(opt)))
        # si en path_sketch hay menos bocetos que tam batch, tomar como batch tam
        # el num de bocetos
        num_bocetos = len(os.listdir(path_sketch))
        assert num_bocetos>=1, "No hay bocetos en la ruta indicada"
        if (num_bocetos<opt.batch_size):
            log(f,"WARNING: Hay menos bocetos que el tamaño de batch. batch_size: {}".format(num_bocetos))
            opt.batch_size = len(os.listdir(path_sketch))

        # dataloaders
        dataloader_images = yield_data(create_dataloader(opt.path_images, 256, opt.batch_size))

        log(f,"Iniciando entrenamiento " + path_sketch)
        dataloader_sketch = create_dataloader(path_sketch, 256, opt.batch_size, img_channel=1) 
        # guardo sketches en memoria y me aseguro que no sean RGBA (los provenientes de la web lo son)
        check_sketches_save(path_sketch,os.path.join(opt.example_folder,path_sketch.split("/")[1], now, "sketches_provided.jpg"))

        log(f,"Aumentación de los bocetos {}".format("activada" if opt.enable_aug else "desactivada"))
        log(f,"Regularización de pesos {}".format("activada" if opt.disable_d_reg else "desactivada"))
        model = SketchGANModel(path_model=opt.pathNetwork,disable_d_reg=opt.disable_d_reg,r1=opt.r1,reg_each=opt.d_reg_each,aug=opt.enable_aug,batch_size=opt.batch_size)

        if (opt.pathG and opt.pathDSketch and opt.pathDImage):
            model.resume_training(opt.pathG,opt.pathDSketch,opt.pathDImage)
            log(f,"Continuando entrenamiento...")

        if (opt.z_vectors):
            z_vectors = torch.load(opt.z_vectors,map_location=model.device)
            log(f,"z_vectors proporcionados")
        else:    
            z_vectors = torch.randn(opt.example_elements, model.G.z_dim, device=model.device)
            torch.save(z_vectors,os.path.join(opt.example_folder,path_sketch.split("/")[1], now,"z_vectors.pt"))

        start_training = time.time()
        iters = 0
        for epoch in range(opt.num_epochs):
            exceediters = False
            for i,sketches in enumerate(dataloader_sketch): 
                if(iters%opt.example_each==0):
                    generated = (model.generate_fake(z_vectors) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    torch.save(generated, os.path.join(opt.example_folder,path_sketch.split("/")[1], now, "images","iter_{}.pt".format(iters)))
                    generated = ToPILImage()(make_grid(generated, nrow=int(opt.example_elements/(int(math.sqrt(opt.example_elements))-1))))
                    generated.save(os.path.join(opt.example_folder,path_sketch.split("/")[1], now, "images","iter_{}.jpg".format(iters)))
                    log(f,"Guardando imagen... Hecho\n")

                # ENTRENAMIENTO #
                realImages = next(dataloader_images)
                data = {'image': realImages, 'sketch': sketches}
                model(data,iters) # el entrenamiento se realiza en la llamada al forward.

                G_loss = model.g_losses[-1]
                D_loss = model.d_losses[-1]
                D_reg_loss = model.d_reg_losses[-1] if len(model.d_reg_losses)!=0 else None
                iter_end = time.time()
                # ENTRENAMIENTO #
                
                # estadisticas, guardados...
                if(iters%opt.stats_each==0):
                    showStats(G_loss,D_loss,D_reg_loss,epoch,iters,iter_end,start_training,f)

                if(iters!=0 and iters%opt.save_network_each==0): # no tiene sentido guardar la red en el primer epoch..
                    saveNetwork(model,path_sketch,now,epoch,f)

                if(iters!=0 and iters%opt.save_stats_each==0):
                    saveStats(model,path_sketch,now,f)

                if(iters>=opt.max_iters):
                    log(f,"Máximo número de iteraciones alcanzado")
                    saveNetwork(model,path_sketch,now,epoch,f)
                    saveStats(model,path_sketch,now,f)
                    exceediters = True
                    break
            
                iters = iters+1
            if exceediters: break
        
        generate_transitions(os.path.join(opt.example_folder,path_sketch.split("/")[1], now),stride=2)
        torch.save(model.G.state_dict(), os.path.join(opt.example_folder,path_sketch.split("/")[1], now, "snaps", "G_epoch_{}".format(epoch)))
        torch.save(model.D_sketch.state_dict(), os.path.join(opt.example_folder,path_sketch.split("/")[1], now, "snaps", "D_sketch_epoch_{}".format(epoch)))
        torch.save(model.D_image.state_dict(), os.path.join(opt.example_folder,path_sketch.split("/")[1], now, "snaps", "D_image_epoch_{}".format(epoch)))
        log(f,"Guardando las redes... Hecho\n")
        log(f,"El entrenamiento ha finalizado")   
        f.close()


def check_sketches_save(path,dest):
    acum = []
    for sk in os.listdir(path):
        img = ToTensor()(Image.open(os.path.join(path, sk)))
        if (img.size()[0]==4):
            img = ToPILImage()(img[2]).convert("L")
            img.save(os.path.join(path, sk))
            img = ToTensor()(img)
        else:
            img = ToTensor()(ToPILImage()(img).convert("L"))
        img = transforms.Resize((256,256))(img)
        acum.append(img)
    ToPILImage()(make_grid(acum, nrow=int(math.sqrt(len(acum))))).save(dest)

def log(f, text):
    print(text)
    f.write(text+"\n")

if __name__ == "__main__":    
    training_loop()