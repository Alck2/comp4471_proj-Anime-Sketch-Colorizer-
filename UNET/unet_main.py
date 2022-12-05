import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from mymodels import weights_init
from .unet_part import encoder_block, decoder_block, conv_block

class UNet(nn.Module):
    def __init__(self,pretrained):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(6, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)        
        """ Bottleneck """
        self.b = conv_block(512, 1024)        
        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)        
        """ Classifier """
        self.outputs = nn.Conv2d(64, 3, kernel_size=1, padding=0)   
        if not pretrained:
            self.apply(weights_init)  
        else:
            print('Loading pretrained {0} model...'.format('Sketch2Color'), end=' ')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/edge2color/ckpt60_4999.pth') # change this to a new saved pytorch weights
            self.load_state_dict(checkpoint['netG'], strict=True)
            print("Done!")
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)        
        """ Bottleneck """
        b = self.b(p4)        
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)         
        """ Classifier """
        outputs = self.outputs(d4)       
        return outputs