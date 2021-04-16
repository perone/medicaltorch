import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F


class NoPoolASPP(Module):
    """
    .. image:: _static/img/nopool_aspp_arch.png
        :align: center
        :scale: 25%

    An ASPP-based model without initial pooling layers.

    :param drop_rate: dropout rate.
    :param bn_momentum: batch normalization momentum.

    .. seealso::
        Perone, C. S., et al (2017). Spinal cord gray matter
        segmentation using deep dilated convolutions.
        Nature Scientific Reports link:
        https://www.nature.com/articles/s41598-018-24304-3

    """
    def __init__(self, drop_rate=0.4, bn_momentum=0.1,
                 base_num_filters=64):
        super().__init__()

        self.conv1a = nn.Conv2d(1, base_num_filters, kernel_size=3, padding=1)
        self.conv1a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv1a_drop = nn.Dropout2d(drop_rate)
        self.conv1b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=1)
        self.conv1b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv1b_drop = nn.Dropout2d(drop_rate)

        self.conv2a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=2, dilation=2)
        self.conv2a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv2a_drop = nn.Dropout2d(drop_rate)
        self.conv2b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=2, dilation=2)
        self.conv2b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.conv2b_drop = nn.Dropout2d(drop_rate)

        # Branch 1x1 convolution
        self.branch1a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=1)
        self.branch1a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch1a_drop = nn.Dropout2d(drop_rate)
        self.branch1b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=1)
        self.branch1b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch1b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 6
        self.branch2a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=6, dilation=6)
        self.branch2a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch2a_drop = nn.Dropout2d(drop_rate)
        self.branch2b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=6, dilation=6)
        self.branch2b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch2b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 12
        self.branch3a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=12, dilation=12)
        self.branch3a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch3a_drop = nn.Dropout2d(drop_rate)
        self.branch3b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=12, dilation=12)
        self.branch3b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch3b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 18
        self.branch4a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=18, dilation=18)
        self.branch4a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch4a_drop = nn.Dropout2d(drop_rate)
        self.branch4b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=18, dilation=18)
        self.branch4b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch4b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 24
        self.branch5a = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=24, dilation=24)
        self.branch5a_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch5a_drop = nn.Dropout2d(drop_rate)
        self.branch5b = nn.Conv2d(base_num_filters, base_num_filters, kernel_size=3, padding=24, dilation=24)
        self.branch5b_bn = nn.BatchNorm2d(base_num_filters, momentum=bn_momentum)
        self.branch5b_drop = nn.Dropout2d(drop_rate)

        self.concat_drop = nn.Dropout2d(drop_rate)
        self.concat_bn = nn.BatchNorm2d(6*base_num_filters, momentum=bn_momentum)

        self.amort = nn.Conv2d(6*base_num_filters, base_num_filters*2, kernel_size=1)
        self.amort_bn = nn.BatchNorm2d(base_num_filters*2, momentum=bn_momentum)
        self.amort_drop = nn.Dropout2d(drop_rate)

        self.prediction = nn.Conv2d(base_num_filters*2, 1, kernel_size=1)

    def forward(self, x):
        """Model forward pass.

        :param x: input data.
        """
        x = F.relu(self.conv1a(x))
        x = self.conv1a_bn(x)
        x = self.conv1a_drop(x)

        x = F.relu(self.conv1b(x))
        x = self.conv1b_bn(x)
        x = self.conv1b_drop(x)

        x = F.relu(self.conv2a(x))
        x = self.conv2a_bn(x)
        x = self.conv2a_drop(x)
        x = F.relu(self.conv2b(x))
        x = self.conv2b_bn(x)
        x = self.conv2b_drop(x)

        # Branch 1x1 convolution
        branch1 = F.relu(self.branch1a(x))
        branch1 = self.branch1a_bn(branch1)
        branch1 = self.branch1a_drop(branch1)
        branch1 = F.relu(self.branch1b(branch1))
        branch1 = self.branch1b_bn(branch1)
        branch1 = self.branch1b_drop(branch1)

        # Branch for 3x3 rate 6
        branch2 = F.relu(self.branch2a(x))
        branch2 = self.branch2a_bn(branch2)
        branch2 = self.branch2a_drop(branch2)
        branch2 = F.relu(self.branch2b(branch2))
        branch2 = self.branch2b_bn(branch2)
        branch2 = self.branch2b_drop(branch2)

        # Branch for 3x3 rate 6
        branch3 = F.relu(self.branch3a(x))
        branch3 = self.branch3a_bn(branch3)
        branch3 = self.branch3a_drop(branch3)
        branch3 = F.relu(self.branch3b(branch3))
        branch3 = self.branch3b_bn(branch3)
        branch3 = self.branch3b_drop(branch3)

        # Branch for 3x3 rate 18
        branch4 = F.relu(self.branch4a(x))
        branch4 = self.branch4a_bn(branch4)
        branch4 = self.branch4a_drop(branch4)
        branch4 = F.relu(self.branch4b(branch4))
        branch4 = self.branch4b_bn(branch4)
        branch4 = self.branch4b_drop(branch4)

        # Branch for 3x3 rate 24
        branch5 = F.relu(self.branch5a(x))
        branch5 = self.branch5a_bn(branch5)
        branch5 = self.branch5a_drop(branch5)
        branch5 = F.relu(self.branch5b(branch5))
        branch5 = self.branch5b_bn(branch5)
        branch5 = self.branch5b_drop(branch5)

        # Global Average Pooling
        global_pool = F.avg_pool2d(x, kernel_size=x.size()[2:])
        global_pool = global_pool.expand(x.size())

        concatenation = torch.cat([branch1,
                                   branch2,
                                   branch3,
                                   branch4,
                                   branch5,
                                   global_pool], dim=1)

        concatenation = self.concat_bn(concatenation)
        concatenation = self.concat_drop(concatenation)

        amort = F.relu(self.amort(concatenation))
        amort = self.amort_bn(amort)
        amort = self.amort_drop(amort)

        predictions = self.prediction(amort)
        predictions = torch.sigmoid(predictions)

        return predictions


class DownConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv1_drop = nn.Dropout2d(drop_rate)

        self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv2_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x


class UpConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(UpConv, self).__init__()
        self.up1 = nn.functional.interpolate
        self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

    def forward(self, x, y):
        x = self.up1(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x


class Unet(Module):
    """A reference U-Net model.

    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597
    """
    def __init__(self, drop_rate=0.4, bn_momentum=0.1):
        super(Unet, self).__init__()

        #Downsampling path
        self.conv1 = DownConv(1, 64, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)

        # Bottom
        self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)

        # Upsampling path
        self.up1 = UpConv(512, 256, drop_rate, bn_momentum)
        self.up2 = UpConv(384, 128, drop_rate, bn_momentum)
        self.up3 = UpConv(192, 64, drop_rate, bn_momentum)

        self.conv9 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.mp1(x1)

        x3 = self.conv2(x2)
        x4 = self.mp2(x3)

        x5 = self.conv3(x4)
        x6 = self.mp3(x5)

        # Bottom
        x7 = self.conv4(x6)

        # Up-sampling
        x8 = self.up1(x7, x5)
        x9 = self.up2(x8, x3)
        x10 = self.up3(x9, x1)

        x11 = self.conv9(x10)
        preds = torch.sigmoid(x11)

        return preds


class UNet3D(nn.Module):
    """A reference of 3D U-Net model.

    Implementation origin :
    https://github.com/shiba24/3d-unet/blob/master/pytorch/model.py

    .. seealso::
        Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox
        and Olaf Ronneberger (2016). 3D U-Net: Learning Dense Volumetric
        Segmentation from Sparse Annotation
        ArXiv link: https://arxiv.org/pdf/1606.06650.pdf
    """
    def __init__(self, in_channel, n_classes):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.down_conv(self.in_channel, 32, bias=False, batchnorm=False)
        self.ec1 = self.down_conv(32, 64, bias=False, batchnorm=False)
        self.ec2 = self.down_conv(64, 64, bias=False, batchnorm=False)
        self.ec3 = self.down_conv(64, 128, bias=False, batchnorm=False)
        self.ec4 = self.down_conv(128, 128, bias=False, batchnorm=False)
        self.ec5 = self.down_conv(128, 256, bias=False, batchnorm=False)
        self.ec6 = self.down_conv(256, 256, bias=False, batchnorm=False)
        self.ec7 = self.down_conv(256, 512, bias=False, batchnorm=False)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.up_conv(512, 512, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.down_conv(256 + 512, 256, bias=False)
        self.dc7 = self.down_conv(256, 256, bias=False)
        self.dc6 = self.up_conv(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.down_conv(128 + 256, 128, bias=False)
        self.dc4 = self.down_conv(128, 128, bias=False)
        self.dc3 = self.up_conv(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.down_conv(64 + 128, 64, bias=False)
        self.dc1 = self.down_conv(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.down_conv(64, n_classes, kernel_size=1, stride=1, padding=0, bias=False)


    def down_conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU())
        return layer


    def up_conv(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.LeakyReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2), dim=1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0
