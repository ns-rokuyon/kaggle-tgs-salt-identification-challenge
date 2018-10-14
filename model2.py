import torch
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision.models.resnet import BasicBlock, ResNet


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
}


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


class SCSEModule(nn.Module):
    """https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/67699
    """
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch//re, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch // re, ch, 1),
                                 nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


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
        dec = self.up(dec)
        enc = self.enc_conv(enc)
        concat = torch.cat([dec, enc], dim=1)
        concat = self.conv(concat)
        return F.relu(self.bn(concat))


class UNetRes34(nn.Module):
    """ https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65933
    """
    def __init__(self, n_classes=1, pretrained_resnet=True):
        super().__init__()
        self.n_classes = n_classes

        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes)
        del self.resnet.fc
        if pretrained_resnet:
            self.resnet.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
            print('Loaded pretrained resnet weights')

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
            nn.MaxPool2d(kernel_size=2, stride=2)   # Add
        )

        self.decoder5 = Decoder(256, 512, 512, 64)
        self.decoder4 = Decoder(64, 256, 256, 64)
        self.decoder3 = Decoder(64, 128, 128, 64)
        self.decoder2 = Decoder(64, 64, 64, 64)
        self.decoder1 = Decoder(64, 64, 32, 64)

        self.conv = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        e1 = self.encoder1(x)   # shape(B, 64, 128, 128)
        e2 = self.encoder2(e1)  # shape(B, 64, 64, 64)
        e3 = self.encoder3(e2)  # shape(B, 128, 32, 32)
        e4 = self.encoder4(e3)  # shape(B, 256, 16, 16)
        e5 = self.encoder5(e4)  # shape(B, 512, 8, 8)

        f = self.center(e5)     # shape(B, 256, 4, 4)

        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        return self.conv(d1)


class UNetRes34HcAux(nn.Module):
    """ https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65933
    """
    def __init__(self, n_classes=1, pretrained_resnet=True):
        super().__init__()
        self.n_classes = n_classes

        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes)
        del self.resnet.fc
        if pretrained_resnet:
            self.resnet.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
            print('Loaded pretrained resnet weights')

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
            nn.MaxPool2d(kernel_size=2, stride=2)   # Add
        )

        self.decoder5 = Decoder(256, 512, 512, 64)
        self.decoder4 = Decoder(64, 256, 256, 64)
        self.decoder3 = Decoder(64, 128, 128, 64)
        self.decoder2 = Decoder(64, 64, 64, 64)
        self.decoder1 = Decoder(64, 64, 32, 64)

        self.logit_pixel  = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.logit_feat = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )   # 3 classes

    def forward(self, x):
        batch_size, C, H, W = x.shape

        e1 = self.encoder1(x)   # shape(B, 64, 128, 128)
        e2 = self.encoder2(e1)  # shape(B, 64, 64, 64)
        e3 = self.encoder3(e2)  # shape(B, 128, 32, 32)
        e4 = self.encoder4(e3)  # shape(B, 256, 16, 16)
        e5 = self.encoder5(e4)  # shape(B, 512, 8, 8)

        f = self.center(e5)     # shape(B, 256, 4, 4)

        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        hc = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        hc = F.dropout(hc, p=0.5, training=self.training)
        logit_pixel = self.logit_pixel(hc)

        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        f = F.dropout(f, p=0.5, training=self.training)
        logit_feat = self.logit_feat(f)

        if not self.training:
            return logit_pixel

        return logit_pixel, logit_feat


class UNetRes34HcAuxSCSE(nn.Module):
    """
    """
    def __init__(self, n_classes=1, pretrained_resnet=True):
        super().__init__()
        self.n_classes = n_classes

        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes)
        del self.resnet.fc
        if pretrained_resnet:
            self.resnet.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
            print('Loaded pretrained resnet weights')

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

        self.enc_se1 = SCSEModule(64)
        self.enc_se2 = SCSEModule(64)
        self.enc_se3 = SCSEModule(128)
        self.enc_se4 = SCSEModule(256)
        self.enc_se5 = SCSEModule(512)

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # Add
        )

        self.decoder5 = Decoder(256, 512, 512, 64)
        self.decoder4 = Decoder(64, 256, 256, 64)
        self.decoder3 = Decoder(64, 128, 128, 64)
        self.decoder2 = Decoder(64, 64, 64, 64)
        self.decoder1 = Decoder(64, 64, 32, 64)

        self.dec_se5 = SCSEModule(64)
        self.dec_se4 = SCSEModule(64)
        self.dec_se3 = SCSEModule(64)
        self.dec_se2 = SCSEModule(64)
        self.dec_se1 = SCSEModule(64)

        self.logit_pixel  = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.logit_feat = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )   # 3 classes

    def forward(self, x):
        batch_size, C, H, W = x.shape

        e1 = self.enc_se1(self.encoder1(x))   # shape(B, 64, 128, 128)
        e2 = self.enc_se2(self.encoder2(e1))  # shape(B, 64, 64, 64)
        e3 = self.enc_se3(self.encoder3(e2))  # shape(B, 128, 32, 32)
        e4 = self.enc_se4(self.encoder4(e3))  # shape(B, 256, 16, 16)
        e5 = self.enc_se5(self.encoder5(e4))  # shape(B, 512, 8, 8)

        f = self.center(e5)     # shape(B, 256, 4, 4)

        d5 = self.dec_se5(self.decoder5(f, e5))
        d4 = self.dec_se4(self.decoder4(d5, e4))
        d3 = self.dec_se3(self.decoder3(d4, e3))
        d2 = self.dec_se2(self.decoder2(d3, e2))
        d1 = self.dec_se1(self.decoder1(d2, e1))

        hc = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        hc = F.dropout(hc, p=0.5, training=self.training)
        logit_pixel = self.logit_pixel(hc)

        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        f = F.dropout(f, p=0.5, training=self.training)
        logit_feat = self.logit_feat(f)

        if not self.training:
            return logit_pixel

        return logit_pixel, logit_feat


class UNetRes34HcAuxSCSEv2(nn.Module):
    """
    """
    def __init__(self, n_classes=1, pretrained_resnet=True):
        super().__init__()
        self.n_classes = n_classes

        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes)
        del self.resnet.fc
        if pretrained_resnet:
            self.resnet.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
            print('Loaded pretrained resnet weights')

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

        self.enc_se1 = SCSEModule(64)
        self.enc_se2 = SCSEModule(64)
        self.enc_se3 = SCSEModule(128)
        self.enc_se4 = SCSEModule(256)
        self.enc_se5 = SCSEModule(512)

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # Add
        )

        self.decoder5 = Decoder(256, 512, 512, 64)
        self.decoder4 = Decoder(64, 256, 256, 64)
        self.decoder3 = Decoder(64, 128, 128, 64)
        self.decoder2 = Decoder(64, 64, 64, 64)
        self.decoder1 = Decoder(64, 64, 32, 64)

        self.dec_se5 = SCSEModule(64)
        self.dec_se4 = SCSEModule(64)
        self.dec_se3 = SCSEModule(64)
        self.dec_se2 = SCSEModule(64)
        self.dec_se1 = SCSEModule(64)

        self.bn_hc = nn.BatchNorm2d(320)
        self.fuse_pixel  = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.logit_pixel_conv = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.bn_gap = nn.BatchNorm1d(512)
        self.fuse_feat = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True)
        )
        self.logit_feat_fc = nn.Linear(64, 3)

        self.logit = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape

        e1 = self.enc_se1(self.encoder1(x))   # shape(B, 64, 128, 128)
        e2 = self.enc_se2(self.encoder2(e1))  # shape(B, 64, 64, 64)
        e3 = self.enc_se3(self.encoder3(e2))  # shape(B, 128, 32, 32)
        e4 = self.enc_se4(self.encoder4(e3))  # shape(B, 256, 16, 16)
        e5 = self.enc_se5(self.encoder5(e4))  # shape(B, 512, 8, 8)

        f = self.center(e5)     # shape(B, 256, 4, 4)

        d5 = self.dec_se5(self.decoder5(f, e5))
        d4 = self.dec_se4(self.decoder4(d5, e4))
        d3 = self.dec_se3(self.decoder3(d4, e3))
        d2 = self.dec_se2(self.decoder2(d3, e2))
        d1 = self.dec_se1(self.decoder1(d2, e1))

        hc = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        hc = self.bn_hc(hc)
        fuse_pixel = self.fuse_pixel(hc)
        logit_pixel = self.logit_pixel_conv(fuse_pixel)

        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        f = self.bn_gap(f)
        fuse_feat = self.fuse_feat(f)
        logit_feat = self.logit_feat_fc(fuse_feat)

        fuse = torch.cat((
            fuse_pixel,
            F.upsample(fuse_feat.view(batch_size, -1, 1, 1), scale_factor=128, mode='nearest')
        ), 1)
        logit = self.logit(fuse)

        if not self.training:
            return logit

        return logit, logit_pixel, logit_feat

