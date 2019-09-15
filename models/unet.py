from torch import nn
import torch

class Interpolate(nn.Module):
    '''Interpolation to use in upsampling'''
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x
    
class basic_conv_block(nn.Module):
    '''Basic U-Net block (Conv(3x3) -> BatchNormalization -> ReLU) x 2'''
    def __init__(self, input_ch, output_ch):
        super(basic_conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_ch, output_ch, 3, padding=1),
                                  nn.BatchNorm2d(output_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(output_ch, output_ch, 3, padding=1),
                                  nn.BatchNorm2d(output_ch),
                                  nn.ReLU(inplace=True))
    def forward(self, x):
        output = self.conv(x)
        return output
    
class downward(nn.Module):
    '''Block for downsampling in U-Net (MaxPooling(2x2) -> BasicBlock)'''
    def __init__(self, input_ch, output_ch):
        super(downward, self).__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2),
                                  basic_conv_block(input_ch, output_ch))
    def forward(self, x):
        output = self.down(x)
        return output
    
class upward(nn.Module):
    '''Block for upsampling in U-Net (Interpolation -> BasicBlock) + concatenation with data from contracting path'''
    def __init__(self, input_ch, output_ch, interp_mode='bilinear'):
        super(upward, self).__init__()
        self.upsample = Interpolate(scale_factor=2, mode=interp_mode)
        self.conv = basic_conv_block(input_ch, output_ch)
    def forward(self, x_up, x_across):
        x_up = self.upsample(x_up)
        
        #we are going to pad x_up before concatenation so it matches the sizes of x_across
        #it is done so that dimensions of output image of U-Net match those of the input image
        #padding is done in 'reflection' mode as suggested in original paper on U-Net
        W_diff = x_across.size()[3] - x_up.size()[3]
        H_diff = x_across.size()[2] - x_up.size()[2]
        
        x_up = nn.functional.pad(x_up, (W_diff // 2, W_diff - W_diff // 2, H_diff, H_diff - H_diff // 2), mode='reflect')
        x = torch.cat([x_across, x_up], dim=1)
        output = self.conv(x)
        return output
    
class UNet(nn.Module):
    '''U-Net model'''
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = basic_conv_block(n_channels, 64)
        self.down1 = downward(64, 128)
        self.down2 = downward(128, 256)
        self.down3 = downward(256, 512)
        self.down4 = downward(512, 512)
        self.up1 = upward(1024, 256)
        self.up2 = upward(512, 128)
        self.up3 = upward(256, 64)
        self.up4 = upward(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

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
        return nn.functional.sigmoid(x)
        
    
    
    
    
    
    
    