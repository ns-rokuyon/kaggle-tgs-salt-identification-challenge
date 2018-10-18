from model2 import *


class UNetRes34AuxSCSE(nn.Module):
    """
    """
    def __init__(self, n_classes=1, n_aux_classes=8, pretrained_resnet=True):
        super().__init__()
        self.n_classes = n_classes
        self.n_aux_classes = n_aux_classes

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

        self.logit = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.logit_aux = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.n_aux_classes)
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

        logit = self.logit(d1)

        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        logit_aux = self.logit_aux(f)

        if not self.training:
            return logit

        return logit, logit_aux


class UNetRes34HcAuxSCSEv3(nn.Module):
    """
    """
    def __init__(self, n_classes=1, n_aux_classes=8, pretrained_resnet=True):
        super().__init__()
        self.n_classes = n_classes
        self.n_aux_classes = n_aux_classes

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

        self.logit = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.hypercolumn = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

        self.logit_aux = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.n_aux_classes)
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

        logit = self.logit(d1)

        hc = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        logit_hc = self.hypercolumn(hc)

        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        logit_aux = self.logit_aux(f)

        if not self.training:
            return logit

        return logit, logit_hc, logit_aux


class UNetRes34AuxSimple(nn.Module):
    """
    """
    def __init__(self, n_classes=1, n_aux_classes=8, pretrained_resnet=True):
        super().__init__()
        self.n_classes = n_classes
        self.n_aux_classes = n_aux_classes

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

        self.logit = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.logit_aux = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.n_aux_classes)
        )

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

        logit = self.logit(d1)

        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        logit_aux = self.logit_aux(f)

        if not self.training:
            return logit

        return logit, logit_aux


class UNetRes34AuxSimple2(nn.Module):
    """
    """
    def __init__(self, n_classes=1, n_aux_classes=8, pretrained_resnet=True):
        super().__init__()
        self.n_classes = n_classes
        self.n_aux_classes = n_aux_classes

        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes)
        del self.resnet.fc
        if pretrained_resnet:
            self.resnet.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
            print('Loaded pretrained resnet weights')

        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        ) # out: 64
        self.encoder2 = self.resnet.layer1  # out: 64
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
        self.decoder1 = LastDecoder(64, 32, 64)

        self.logit = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.logit_aux = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.n_aux_classes)
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape   # (B, 3, 128, 128)

        e1 = self.encoder1(x)   # shape(B, 64, 64, 64)
        e2 = self.encoder2(e1)  # shape(B, 64, 64, 64)
        e3 = self.encoder3(e2)  # shape(B, 128, 32, 32)
        e4 = self.encoder4(e3)  # shape(B, 256, 16, 16)
        e5 = self.encoder5(e4)  # shape(B, 512, 8, 8)

        f = self.center(e5)     # shape(B, 256, 4, 4)

        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2)

        logit = self.logit(d1)

        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        logit_aux = self.logit_aux(f)

        if not self.training:
            return logit

        return logit, logit_aux


class UNetRes34HcAuxSCSEv4(nn.Module):
    """
    """
    def __init__(self, n_classes=1, n_aux_classes=8, pretrained_resnet=True):
        super().__init__()
        self.n_classes = n_classes
        self.n_aux_classes = n_aux_classes

        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes)
        del self.resnet.fc
        if pretrained_resnet:
            self.resnet.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
            print('Loaded pretrained resnet weights')

        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        ) # out: 64
        self.encoder2 = self.resnet.layer1  # out: 64
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
        self.decoder1 = LastDecoder(64, 32, 64)

        self.dec_se5 = SCSEModule(64)
        self.dec_se4 = SCSEModule(64)
        self.dec_se3 = SCSEModule(64)
        self.dec_se2 = SCSEModule(64)
        self.dec_se1 = SCSEModule(64)

        self.hypercolumn = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

        self.logit_aux = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.n_aux_classes)
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
        d1 = self.dec_se1(self.decoder1(d2))

        hc = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        hc = F.dropout2d(hc, p=0.5, training=self.training)
        logit = self.hypercolumn(hc)

        f = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        logit_aux = self.logit_aux(f)

        if not self.training:
            return logit

        return logit, logit_aux
