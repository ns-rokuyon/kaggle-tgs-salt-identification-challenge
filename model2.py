import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision.models.resnet import BasicBlock, ResNet

class ConvBn2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                      stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.layer(x)


class Decoder(nn.Module):
    def __init__(self, in_ch, in_ch_enc, concat_ch, out_ch=64):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, concat_ch // 2,
                                     kernel_size=2, stride=2)
        self.enc_conv = nn.Conv2d(in_ch_enc, concat_ch // 2, kernel_size=1)
        self.conv = nn.Conv2d(concat_ch, out_ch,
                              kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, dec, enc):
        print(dec.shape)
        print(enc.shape)
        dec = self.up(dec)
        enc = self.enc_conv(enc)
        print(dec.shape)
        print(enc.shape)
        concat = torch.cat([dec, enc], dim=1)
        concat = self.conv(concat)
        return F.relu(self.bn(concat))


class UNetRes34(nn.Module):
    """ https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65933
    """
    def __init__(self, n_classes=1):
        super().__init__()
        self.n_classes = n_classes

        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.resnet.layer1,
        )   # out: 64
        self.encoder3 = self.resnet.layer2  # out: 128
        self.encoder4 = self.resnet.layer3  # out: 256
        self.encoder5 = self.resnet.layer4  # out: 512

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(256, 512, 512, 64)
        self.decoder4 = Decoder(64, 256, 256, 64)
        self.decoder3 = Decoder(64, 128, 128, 64)
        self.decoder2 = Decoder(64, 64, 64, 64)
        self.decoder1 = Decoder(64, 64, 32, 64)

        self.conv = nn.Conv2d(64,  1, kernel_size=1, padding=0)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        f = self.center(e5)

        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        return self.conv(d1)


class UNetRes34HcAux:
    """ https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65933
    """
    def __init__(self ):
        super().__init__()
        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1 )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.resnet.layer1,
        )
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        self.center = nn.Sequential(
            ConvBn2d( 512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d( 512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(256, 512, 512, 64)
        self.decoder4 = Decoder( 64, 256, 256, 64)
        self.decoder3 = Decoder( 64, 128, 128, 64)
        self.decoder2 = Decoder( 64,  64,  64, 64)
        self.decoder1 = Decoder( 64,  64,  32, 64)

        self.logit_pixel  = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  1, kernel_size=1, padding=0),
        )

        self.logit_image = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        batch_size,C,H,W = x.shape

        mean=[0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x-mean[2])/std[2],
            (x-mean[1])/std[1],
            (x-mean[0])/std[0],
        ],1)


        e1 = self.encoder1(x )  #; print('e1',e1.size())
        e2 = self.encoder2(e1)  #; print('e2',e2.size())
        e3 = self.encoder3(e2)  #; print('e3',e3.size())
        e4 = self.encoder4(e3)  #; print('e4',e4.size())
        e5 = self.encoder5(e4)  #; print('e5',e5.size())

        f = self.center(e5)                #; print('f',f.size())

        d5 = self.decoder5( f,e5)          #; print('d5',f.size())
        d4 = self.decoder4(d5,e4)          #; print('d4',f.size())
        d3 = self.decoder3(d4,e3)          #; print('d3',f.size())
        d2 = self.decoder2(d3,e2)          #; print('d2',f.size())
        d1 = self.decoder1(d2,e1)          #; print('d1',f.size())

        f = torch.cat((
            d1,
            F.upsample(d2,scale_factor= 2, mode='bilinear',align_corners=False),
            F.upsample(d3,scale_factor= 4, mode='bilinear',align_corners=False),
            F.upsample(d4,scale_factor= 8, mode='bilinear',align_corners=False),
            F.upsample(d5,scale_factor=16, mode='bilinear',align_corners=False),
        ),1)
        f = F.dropout(f, p=0.50, training=self.training)
        logit_pixel = self.logit_pixel(f)


        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size,-1)
        f = F.dropout(f, p=0.50, training=self.training)
        logit_image = self.logit_image(f).view(-1)

        return logit_pixel, logit_image