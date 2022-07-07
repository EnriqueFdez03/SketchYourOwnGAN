import torch
import dnnlib
import legacy
from networks.usePhotoSketch import PhotoSketchInfer
from torch.nn import functional as F
from torch import nn, autograd
from dataloader import SketchTransforms


import copy
import random


class SketchGANModel(torch.nn.Module):
    def __init__(self,path_model,disable_d_reg=True,r1=None,reg_each=None,aug=False,batch_size=8):
        super().__init__()
        self.aug = aug
        self.batch_size=batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.G, self.D_sketch, self.D_image = self.__load_model(path_model)
        self.beta1 = 0.0
        self.beta2 = 0.99
        self.G_lr = 0.002
        self.D_lr = 0.002
        self.G_params = []
        for name, param in self.G.named_parameters():
            if "mapping" in name: # sólo vamos a cambiar los pesos de la mapping network. Ver: Pág:
                self.G_params.append(param)
        self.D_params = list(self.D_sketch.parameters())+list(self.D_image.parameters())
        self.__set_requires_grad(self.G.parameters(), False)
        self.__set_requires_grad(self.D_params, False)
        self.optimizer_G, self.optimizer_D = self.__create_optimizer()
        self.R1reg = R1_regularization()
        self.photoSketch = PhotoSketchInfer()
        self.l_regu = 0.7 # Se trata de la intensidad con la que aplicaremos image regularization
        self.g_losses = []
        self.d_losses = []
        self.d_reg_losses = []
        self.obtainSketch_transforms = SketchTransforms(toSketch=True,augment=self.aug) # recibe imagen y la transforma a boceto con 3 canales
        self.realSketch_transforms = SketchTransforms(augment=self.aug) # recibe boceto y lo pasa a 3 canales
        self.r1 = r1
        self.reg_each = reg_each
        self.disable_d_reg = disable_d_reg
        


    # En cada paso hacia adelante se requiere de una imagen y sketch.
    # La regularización de imágenes es obligatoria.
    # data es un diccionario que contiene 'image' y 'sketch'.
    def forward(self, data, iters):
        image = data['image'] # para el image regularization
        sketch = data['sketch']
        if self.device == 'cuda':
            image = image.cuda()
            sketch = sketch.cuda()
    
        # calculamos el loss del discriminador y actualizamos el discriminador
        self.__set_requires_grad(self.G_params, False)
        self.__set_requires_grad(self.D_params, True)
        d_loss = self.discriminator_loss(sketch, image)
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        
        if self.disable_d_reg and iters%self.reg_each==0:
        # calculamos el r1 loss del discriminador
            self.__set_requires_grad(self.G_params, False)
            self.__set_requires_grad(self.D_params, True)
            r1_loss = self.R1_discriminator_reg(sketch,image)
            self.optimizer_D.zero_grad()
            r1_loss.backward()
            self.optimizer_D.step()
  
        # calculamos el loss del generador 
        self.__set_requires_grad(self.G_params, True)
        self.__set_requires_grad(self.D_params, False)
        g_loss, fake_img = self.generator_loss()
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()
  
        return fake_img

    # Calcula, Ldiscriminator = Lsketch + λimageLimage
    def discriminator_loss(self, real_sketches, real_imgs):
        D_loss_log = {}
        # StyleGan genera imágenes falsas
        with torch.no_grad():
            fake_imgs = self.random_image().detach()
        # Transformamos esas imágenes falsas en bocetos.
        real_sketches = self.realSketch_transforms(real_sketches)
        fake_sketches = self.obtainSketch_transforms(fake_imgs)
        fake_sketches.detach()
     
        pred_realSketch = self.D_sketch(c=None,img=real_sketches).cpu()
        pred_fakeSketch = self.D_sketch(c=None,img=fake_sketches).cpu() 
        loss_realSk = F.binary_cross_entropy_with_logits(pred_realSketch, self.__get_targetVector(pred_realSketch,self.get_target(True, not self.disable_d_reg)))
        loss_fakeSk = F.binary_cross_entropy_with_logits(pred_fakeSketch, self.__get_targetVector(pred_fakeSketch,self.get_target(False, not self.disable_d_reg)))

        D_loss_log["D_loss_realSketch"] = loss_realSk
        D_loss_log["D_loss_fakeSketch"] = loss_fakeSk
       
        # IMAGE REGULARIZATION
        pred_realImg = self.D_image(c=None, img=real_imgs).cpu()
        pred_fakeImg = self.D_image(c=None, img=fake_imgs).cpu()
        
        loss_realImg = F.binary_cross_entropy_with_logits(pred_realImg, self.__get_targetVector(pred_realSketch,self.get_target(True, self.disable_d_reg)))
        loss_fakeImg = F.binary_cross_entropy_with_logits(pred_fakeImg, self.__get_targetVector(pred_realSketch,self.get_target(False, not self.disable_d_reg)))

        D_loss_log["D_loss_realImg"] = self.l_regu*loss_realImg
        D_loss_log["D_loss_fakeImg"] = self.l_regu*loss_fakeImg
        
        # Lsketch = Ey∼pdata(y) log(Dsketch(y)) + Ez∼p(z) log(1 − Dsketch (F(G(z))))
        loss_sketch = loss_realSk + loss_fakeSk
        # Limage = Ex∼pdata(x) log(Dimage(x)) + Ez∼p(z) log(1 − Dimage(G(z)))
        loss_image = loss_realImg + loss_fakeImg
            
        loss = loss_sketch + self.l_regu*loss_image  
        D_loss_log["D_loss"] = loss
        self.d_losses.append(D_loss_log)
        return loss
    
    def R1_discriminator_reg(self, real_sketches, real_imgs):
        D_reg_log = {}
        real_sketches = self.realSketch_transforms(real_sketches)

        real_sketches.requires_grad = True
        pred_realSketch = self.D_sketch(c=None,img=real_sketches)
        r1_loss = self.R1reg(pred_realSketch,real_sketches)
        r1_loss_sketch = self.r1/2 * r1_loss * self.reg_each
        
        # regularización discriminador imagenes
        real_imgs.requires_grad = True
        pred_realImg = self.D_image(c=None, img=real_imgs)
        r1_loss = self.R1reg(pred_realImg,real_imgs)
        r1_loss_imgs = self.l_regu * self.r1/2 * r1_loss * self.reg_each
        
        D_reg_log["r1_loss_sketch"] = r1_loss_sketch
        D_reg_log["r1_loss_imgs"] = r1_loss_imgs
        r1_loss = r1_loss_sketch +  r1_loss_imgs
        D_reg_log["r1_loss"] = r1_loss
        self.d_reg_losses.append(D_reg_log)
        return r1_loss
        

    def generator_loss(self):
        G_loss_log = {}
        fake_imgs = self.random_image()

        fake_sketches = self.obtainSketch_transforms(fake_imgs)
        pred_fakeSketch = self.D_sketch(c=None,img=fake_sketches).cpu()
        loss_fakeSk = F.binary_cross_entropy_with_logits(pred_fakeSketch, self.__get_targetVector(pred_fakeSketch,1.0))
        # Image regularization
        pred_fakeImg = self.D_image(c=None, img=fake_imgs).cpu()
        loss_fakeImg = F.binary_cross_entropy_with_logits(pred_fakeImg, self.__get_targetVector(pred_fakeImg,0.0))
        
        G_loss_log["G_loss_fakeSketch"] = loss_fakeSk
        G_loss_log["G_loss_fakeImg"] = loss_fakeImg
        loss = loss_fakeSk + self.l_regu*loss_fakeImg
        G_loss_log["G_loss"] = loss
        self.g_losses.append(G_loss_log)
        
        return loss, fake_imgs

    def random_image(self):
        z_vectors = torch.randn(self.batch_size, self.G.z_dim, device=self.device)
        generated_imgs = self.G(z_vectors,None) # c es None
        return generated_imgs
    
    def generate_fake(self, z):
        with torch.no_grad():
            generated_imgs = self.G(z,None) # c es None
            return generated_imgs

    def postProcessImg(self,img):
        return (img* 127.5 + 128).clamp(0, 255)
        
    
    def __create_optimizer(self):
        optimizer_G = torch.optim.Adam(self.G_params, lr=self.G_lr, betas=(self.beta1, self.beta2))
        optimizer_D = torch.optim.Adam(self.D_params, lr=self.D_lr, betas=(self.beta1, self.beta2))

        return optimizer_G, optimizer_D
    
    def __set_requires_grad(self, params, value):
        for param in params:
            param.requires_grad = value
            

    # Realiza la carga de la GAN preentrenada y retorna la red generadora y discrimininadora.
    def __load_model(self,path_model):
        with dnnlib.util.open_url(path_model,'rb') as f:
            gan = legacy.load_network_pkl(f)
            G = gan['G_ema'].train().requires_grad_(False).to(self.device)
            D_image = gan['D'].train().requires_grad_(False).to(self.device)
            D_sketch = copy.deepcopy(gan['D']).train().requires_grad_(False).to(self.device)
            return G,D_image,D_sketch
            
    # Obtiene el boceto de una imagen falsa mediante el método indicado por method.
    def _generate_sketch(self, imgs, method="photosketch"):
        if (method=="photosketch"):
            generatedSketches = self.photoSketch.obtain_sketch(imgs)
            return generatedSketches
            
        else:
            return 
    
    # Dado un vector de predicciones y un valor, genera un vector de dicho tamaño de 
    # ese valor. Si se espera falsa, ese valor es 0, en caso contrario, es 1.
    def __get_targetVector(self, input, value=0.):
        tensor = torch.FloatTensor(1).fill_(value)
        tensor.requires_grad_(False)
        return tensor.expand_as(input)

    def resume_training(self, pathG, pathDsketch, pathDimage):
        self.G.load_state_dict(torch.load(pathG))
        self.D_image.load_state_dict(torch.load(pathDimage))
        self.D_sketch.load_state_dict(torch.load(pathDsketch))

    def get_target(self, value, rand=False):
        if value and rand:
            return random.uniform(0.9, 1.0)
        elif value:
            return 1.
        elif not value and rand:
            return random.uniform(0.0, 0.1)
        else:
            return 0.

class R1_regularization(nn.Module):
    def forward(self, real_pred, real_img):
        outputs = real_pred.reshape(real_pred.shape[0], -1).mean(1).sum()
        grad_real, = autograd.grad(
            outputs=outputs, inputs=real_img, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty