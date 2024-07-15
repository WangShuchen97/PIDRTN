import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels=1, out_1x1=8, red_3x3=4, out_3x3=8, red_5x5=4, out_5x5=8, out_pool=8):
        super(InceptionBlock, self).__init__()

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1x1_output = self.branch1x1(x)
        branch3x3_output = self.branch3x3(x)
        branch5x5_output = self.branch5x5(x)
        branch_pool_output = self.branch_pool(x)

        outputs = [branch1x1_output, branch3x3_output, branch5x5_output, branch_pool_output]
        return torch.cat(outputs, 1)  
    

    
class BasicConv(nn.Module):

    def   __init__(self, in_channels, out_channels, kernel=3, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding="same"),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding="same"),
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            BasicConv(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = BasicConv(in_channels, out_channels, kernel=kernel)
        
    def forward(self, x1, x2):
        y = self.up(x1)
        y  = torch.cat([x2, y], dim=1)
        y  = self.conv(y)
        return y


class Outc(nn.Module):

    def __init__(self, in_channels, out_channels,mid_channels, kernel=3):
        super().__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channels, mid_channels, kernel),
            BasicConv(mid_channels, out_channels, kernel),
            nn.Conv2d(out_channels, out_channels, kernel_size=1) 
        )
 
    def forward(self, x):
        x=self.conv(x)
        return x