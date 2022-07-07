# Script que facilita el uso de photosketch en model.py
from networks.photosketch.models.pix2pix_model import Pix2PixInfer
from networks.photosketch.models.networks import ResnetGenerator
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, ToPILImage
import torch 

class PhotoSketchInfer():
    def __init__(self):
        self.pix2pix = Pix2PixInfer()
        self.pix2pix.initialize()

    def obtain_sketch(self, img):
        dict_img = self.__preprocessImg(img)
        self.pix2pix.set_input(dict_img)
        self.pix2pix.test()
        return self.pix2pix.post_processImg()

    # img is a batch of imgs
    def __preprocessImg(self,img):
        if (len(img.size())!=4): # debe ser BCWH
            img = torch.unsqueeze(img,0)
        
        img = img/255 #see: https://discuss.pytorch.org/t/why-is-this-transform-resulting-in-a-divide-by-zero-error/86755
        img = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        b,c,w,h = img.size()
      
        return {'A': img, 'w': w, 'h': h}

def PhotoSketchTrain(weights):
    generator = ResnetGenerator(3, 1, n_blocks=9, use_dropout=False)
    state_dict = torch.load(weights, map_location='cpu')
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    generator.load_state_dict(state_dict)
    generator.train()
    for param in generator.parameters():
        param.requires_grad = False
    return generator.cuda()

# Ejemplo de uso
def create_model():
    model = PhotoSketchInfer()
    img = ToTensor()(Image.open("prueba.png"))
    img2 = ToTensor()(Image.open("prueba.png"))
    imgs =  torch.tensor([img.numpy(), img2.numpy()])
    sketches = model.obtain_sketch(imgs)
    for sketch in sketches:
        sketch.show()