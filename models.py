import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super.__init__(UNet, self)
        self.downconv0        = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                        stride=2, padding=1)
        self.downconv1        = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                        stride=2, padding=1)
        self.downconv2        = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                        stride=2, padding=1)
        self.downconv3        = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                        stride=2, padding=1)

        self.downrelu        = nn.LeakyReLU(0.2, True)
        self.downnorm        = BatchNorm2d(inner_nc)
        self.uprelu          = nn.ReLU(True)
        self.upnorm          = BatchNorm2d(outer_nc)

class UNetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
            submodule=None, outermost=False, innermost=False,
            norm_layer=nn.BatchNorm2d, use_dropout=False, freeze_encoder=False):
        super(UNetSkipConnectionBlock, self).__init__()

        if input_nc is None:
            input_nc    = outer_nc

        self.outermost  = outermost
        downconv        = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                        stride=2, padding=1)
        

        downrelu        = nn.LeakyReLU(0.2, True)
        downnorm        = norm_layer(inner_nc)
        uprelu          = nn.ReLU(True)
        upnorm          = norm_layer(outer_nc)

        if freeze_encoder:
            for param in downconv.parameters():
                param.requires_grad = False
            for param in downnorm.parameters():
                param.requires_grad = False

        if outermost:
            upconv      = nn.ConvTranspose2d(inner_nc*2, outer_nc,
                                        kernel_size=4, stride=2, padding=1)
            down        = [downconv]
            #up          = [uprelu, upconv, nn.Tanh()]
            # I think we want logits here which can be passed BCE or softmax depending
            up          = [uprelu, upconv]
            model       = down + [submodule] + up
        elif innermost:
            upconv      = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
            down        = [downrelu, downconv]
            up          = [uprelu, upconv, upnorm]
            model       = down + up
        else:
            upconv      = nn.ConvTranspose2d(inner_nc*2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
            down        = [downrelu, downconv, downnorm]
            up          = [uprelu, upconv, upnorm]

            if use_dropout:
                model   = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model   = down + [submodule] + up
        self.model  = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            y = self.model(x)  
            y = nn.functional.pad(y, (0,x.size()[-2]-y.size()[-2],0,x.size()[-1]-y.size()[-1]))
            return torch.cat([x, y], 1)


# much of thi the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, freeze_encoder=False):
        super(UNetGenerator, self).__init__()

        # construct unet structure
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, 
                                            input_nc=None, submodule=None, 
                                            norm_layer=norm_layer, innermost=True, 
                                            freeze_encoder=freeze_encoder)

        #for i in range(num_downs - 5):
        #    unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, 
        #                                        input_nc=None, submodule=unet_block, 
        #                                        norm_layer=norm_layer, use_dropout=use_dropout,
        #                                        freeze_encoder=freeze_encoder)
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, 
                                            submodule=unet_block, norm_layer=norm_layer, 
                                            freeze_encoder=freeze_encoder)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, 
                                            input_nc=None, submodule=unet_block, norm_layer=norm_layer, 
                                            freeze_encoder=freeze_encoder)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, 
                                            submodule=unet_block, norm_layer=norm_layer, 
                                            freeze_encoder=freeze_encoder)
        unet_block = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, 
                                            submodule=unet_block, outermost=True, norm_layer=norm_layer, 
                                            freeze_encoder=freeze_encoder)

        self.model = unet_block

    def forward(self, x):
        return self.model(x) 

class UNet3d(nn.Module):
    def __init__(self):
        super.__init__(UNet3d, self)
        self.downconv0        = nn.Conv3d(input_nc, inner_nc, kernel_size=3,
                                        stride=2, padding=1)
        self.downconv1        = nn.Conv3d(input_nc, inner_nc, kernel_size=3,
                                        stride=2, padding=1)
        self.downconv2        = nn.Conv3d(input_nc, inner_nc, kernel_size=3,
                                        stride=2, padding=1)
        self.downconv3        = nn.Conv3d(input_nc, inner_nc, kernel_size=3,
                                        stride=2, padding=1)

        self.downrelu        = nn.LeakyReLU(0.2, True)
        self.downnorm        = BatchNorm3d(inner_nc)
        self.uprelu          = nn.ReLU(True)
        self.upnorm          = BatchNorm3d(outer_nc)

class UNetSkipConnectionBlock3d(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
            submodule=None, outermost=False, innermost=False,
            norm_layer=nn.BatchNorm3d, use_dropout=False, freeze_encoder=False):
        super(UNetSkipConnectionBlock3d, self).__init__()

        if input_nc is None:
            input_nc    = outer_nc

        self.outermost  = outermost
        downconv        = nn.Conv3d(input_nc, inner_nc, kernel_size=3,
                                        stride=2, padding=1)
        

        downrelu        = nn.LeakyReLU(0.2, True)
        downnorm        = norm_layer(inner_nc)
        uprelu          = nn.ReLU(True)
        upnorm          = norm_layer(outer_nc)

        if freeze_encoder:
            for param in downconv.parameters():
                param.requires_grad = False
            for param in downnorm.parameters():
                param.requires_grad = False

        if outermost:
            upconv      = nn.ConvTranspose3d(inner_nc*2, outer_nc,
                                        kernel_size=3, stride=2, padding=1)
            down        = [downconv]
            #up          = [uprelu, upconv, nn.Tanh()]
            # I think we want logits here which can be passed BCE or softmax depending
            up          = [uprelu, upconv]
            model       = down + [submodule] + up
        elif innermost:
            upconv      = nn.ConvTranspose3d(inner_nc, outer_nc,
                                            kernel_size=3, stride=2,
                                            padding=1)
            down        = [downrelu, downconv]
            up          = [uprelu, upconv, upnorm]
            model       = down + up
        else:
            upconv      = nn.ConvTranspose3d(inner_nc*2, outer_nc,
                                            kernel_size=3, stride=2,
                                            padding=1)
            down        = [downrelu, downconv, downnorm]
            up          = [uprelu, upconv, upnorm]

            if use_dropout:
                model   = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model   = down + [submodule] + up
        self.model  = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            x = self.model(x)
            import pdb; pdb.set_trace()
            return x
        else:
            y = self.model(x)  
            print(y.size())
            if len(y.size()) > 4: 
                y = nn.functional.pad(y, (0, x.size()[-3]-y.size()[-3], \
                        0,x.size()[-2]-y.size()[-2],0,x.size()[-1]-y.size()[-1]))
            else:
                y = nn.functional.pad(y, (0,x.size()[-2]-y.size()[-2],0,x.size()[-1]-y.size()[-1]))
            return torch.cat([x, y], 1)


# much of thi the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UNetGenerator3d(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False, freeze_encoder=False):
        super(UNetGenerator3d, self).__init__()

        # construct unet structure
        unet_block = UNetSkipConnectionBlock3d(ngf * 8, ngf * 8, 
                                            input_nc=None, submodule=None, 
                                            norm_layer=norm_layer, innermost=True, 
                                            freeze_encoder=freeze_encoder)
        unet_block = UNetSkipConnectionBlock3d(ngf * 4, ngf * 8, input_nc=None, 
                                            submodule=unet_block, norm_layer=norm_layer, 
                                            freeze_encoder=freeze_encoder)
        unet_block = UNetSkipConnectionBlock3d(ngf * 2, ngf * 4, 
                                            input_nc=None, submodule=unet_block, norm_layer=norm_layer, 
                                            freeze_encoder=freeze_encoder)
        unet_block = UNetSkipConnectionBlock3d(ngf, ngf * 2, input_nc=None, 
                                            submodule=unet_block, norm_layer=norm_layer, 
                                            freeze_encoder=freeze_encoder)
        unet_block = UNetSkipConnectionBlock3d(output_nc, ngf, input_nc=input_nc, 
                                            submodule=unet_block, outermost=True, norm_layer=norm_layer, 
                                            freeze_encoder=freeze_encoder)

        self.model = unet_block

    def forward(self, x):
        return self.model(x) 

# code was stolen from/inspired by:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2ecf15f8a7f87fa56e784e0504136e9daf6b93d6/models/networks.py#L259 

