# full assembly of the sub-parts to form the complete net
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision.models.resnet import BasicBlock
from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNetRes(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = conv3x3(n_channels, 64)
        self.down1 = down(64, 128, down_block=BasicResBlock)
        self.down2 = down(128, 256, down_block=BasicResBlock)
        self.down3 = down(256, 512, down_block=BasicResBlock)
        self.down4 = down(512, 512, down_block=BasicResBlock)
        self.up1 = up(1024, 256, up_block=BasicResBlock)
        self.up2 = up(512, 128, up_block=BasicResBlock)
        self.up3 = up(256, 64, up_block=BasicResBlock)
        self.up4 = up(128, 64, up_block=BasicResBlock)
        self.outc = conv3x3(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


def get_sfs_idxs(sfs, last=True):
    """
    Return the saved feature indexes that will be concatenated
    Inputs:
        sfs (list): saved features by hook function, in other words intermediate activations
        last (bool): whether to concatenate only last different activation, or all from the encoder model
    """
    if last:
        feature_szs = [sfs_feats.features.size()[-1] for sfs_feats in sfs]
        sfs_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
        if feature_szs[0] != feature_szs[1]: sfs_idxs = [0] + sfs_idxs
    else: sfs_idxs = list(range(len(sfs)))
    return sfs_idxs


def conv_bn_relu(in_c, out_c, kernel_size, stride, padding):
    return [
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.BatchNorm2d(out_c)]


class UnetBlock(nn.Module):
    #TODO: ADAPT KERNEL SIZE, STRIDE AND PADDING SO THAT ANY SIZE DECAY WILL BE SUPPORTED
    def __init__(self, up_in_c, x_in_c):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(up_in_c, up_in_c // 2, 2, 2) # H, W -> 2H, 2W
        self.conv1 = nn.Conv2d(x_in_c + up_in_c // 2, (x_in_c + up_in_c // 2) // 2, 3, 1, 1)
        self.conv2 = nn.Conv2d((x_in_c + up_in_c // 2) // 2, (x_in_c + up_in_c // 2) // 2, 3, 1, 1)
        self.bn = nn.BatchNorm2d((x_in_c + up_in_c // 2) // 2)

    def forward(self, up_in, x_in):
        up_out = self.upconv(up_in)
        cat_x = torch.cat([up_out, x_in], dim=1)
        x = F.relu(self.conv1(cat_x))
        x = F.relu(self.conv2(x))
        return self.bn(x)


class SaveFeatures():
    """ Extract pretrained activations"""
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()


class DynamicUnet(nn.Module):
    """
    A dynamic implementation of Unet architecture, because calculating connections
    and channels suck!. When an encoder is passed, this network will
    automatically construct a decoder after the first single forward pass for any
    given encoder architecture.

    Decoder part is heavily based on the original Unet paper:
    https://arxiv.org/abs/1505.04597.

    Inputs:
        encoder(nn.Module): Preferably a pretrained model, such as VGG or ResNet
        last (bool): Whether to concat only last activation just before a size change
        n_classes (int): Number of classes to output in final step of decoder

    Important Note: If architecture directly reduces the dimension of an image as soon as the
    first forward pass then output size will not be same as the input size, e.g. ResNet.
    In order to resolve this problem architecture will add an additional extra conv transpose
    layer. Also, currently Dynamic Unet expects size change to be H,W -> H/2, W/2. This is
    not a problem for state-of-the-art architectures as they follow this pattern but it should
    be changed for custom encoders that might have a different size decay.
    """

    def __init__(self, encoder, last=True, n_classes=3):
        super().__init__()
        self.encoder = encoder
        self.n_children = len(list(encoder.children()))
        #self.sfs = [SaveFeatures(encoder[i]) for i in range(self.n_children)]
        self.sfs = [SaveFeatures(c) for c in encoder.children()]
        self.last = last
        self.n_classes = n_classes

    def forward(self, x):
        dtype = x.data.type()

        # get imsize
        imsize = x.size()[-2:]

        # encoder output
        x = F.relu(self.encoder(x))

        # initialize sfs_idxs, sfs_szs, middle_in_c and middle_conv only once
        if not hasattr(self, 'middle_conv'):
            self.sfs_szs = [sfs_feats.features.size() for sfs_feats in self.sfs]
            self.sfs_idxs = get_sfs_idxs(self.sfs, self.last)
            middle_in_c = self.sfs_szs[-1][1]
            middle_conv = nn.Sequential(*conv_bn_relu(middle_in_c, middle_in_c * 2, 3, 1, 1),
                                        *conv_bn_relu(middle_in_c * 2, middle_in_c, 3, 1, 1))
            self.middle_conv = middle_conv.type(dtype)

        # middle conv
        x = self.middle_conv(x)

        # initialize upmodel, extra_block and 1x1 final conv
        if not hasattr(self, 'upmodel'):
            x_copy = Variable(x.data, requires_grad=False)
            upmodel = []
            for idx in self.sfs_idxs[::-1]:
                up_in_c, x_in_c = int(x_copy.size()[1]), int(self.sfs_szs[idx][1])
                unet_block = UnetBlock(up_in_c, x_in_c).type(dtype)
                upmodel.append(unet_block)
                x_copy = unet_block(x_copy, self.sfs[idx].features)
                self.upmodel = nn.Sequential(*upmodel)

            if imsize != self.sfs_szs[0][-2:]:
                extra_in_c = self.upmodel[-1].conv2.out_channels
                self.extra_block = nn.ConvTranspose2d(extra_in_c, extra_in_c, 2, 2).type(dtype)

            final_in_c = self.upmodel[-1].conv2.out_channels
            self.final_conv = nn.Conv2d(final_in_c, self.n_classes, 1).type(dtype)

        # run upsample
        for block, idx in zip(self.upmodel, self.sfs_idxs[::-1]):
            x = block(x, self.sfs[idx].features)
        if hasattr(self, 'extra_block'):
            x = self.extra_block(x)

        out = self.final_conv(x)
        return out


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.final(p), self.classifier(auxiliary)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, input_channel=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model