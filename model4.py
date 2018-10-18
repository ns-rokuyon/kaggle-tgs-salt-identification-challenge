from model2 import *


class SCSEDecoder(nn.Module):
    def __init__(self, in_ch, ch, out_ch):
        super().__init__()
        self.conv1 = ConvBn2d(in_ch, ch)
        self.conv2 = ConvBn2d(ch, out_ch)
        self.scse = SCSEModule(out_ch)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], dim=1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.scse(x)

        return x


class UNetRes34BilinearHcSCSEv5(nn.Module):
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

        self.decoder5 = SCSEDecoder(256 + 512, 512, 64)
        self.decoder4 = SCSEDecoder(64 + 256, 256, 64)
        self.decoder3 = SCSEDecoder(64 + 128, 128, 64)
        self.decoder2 = SCSEDecoder(64 + 64, 64, 64)
        self.decoder1 = SCSEDecoder(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
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
        d1 = self.decoder1(d2)

        hc = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        hc = F.dropout2d(hc, p=0.5, training=self.training)
        logit = self.logit(hc)

        return logit