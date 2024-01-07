import torch.nn as nn
import torch

class LevelConvs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lvl_modules = []

        for i in range(2):
            if i == 0:
                self.lvl_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                self.lvl_modules.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            self.lvl_modules.append(nn.BatchNorm2d(out_channels))
            self.lvl_modules.append(nn.ReLU())
            
        self.lvl_modules = nn.Sequential(*self.lvl_modules)
    
    def forward(self, input):

        level_result = self.lvl_modules(input)

        return level_result

class DownConv(LevelConvs):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, input):
        level_result = self.lvl_modules(input)

        return level_result, self.maxpool(level_result)

class UpConv(LevelConvs):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.upconv = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)

    def forward(self, input):
        level_result = self.lvl_modules(input)

        return self.upconv(level_result)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downers = nn.ModuleList([DownConv(3 if  i==0 else 64 * 2**(i-1), 64 * 2**i) for i in range(4)])
        self.uppers = nn.ModuleList([UpConv(512, 1024)])
        for i in range(3):
            self.uppers.append(UpConv(1024 // 2**i, 512 // 2**i))

        self.end = LevelConvs(128, 64)
        self.end.lvl_modules.add_module("fc", nn.Conv2d(64, 2, kernel_size=1))
            
    def forward(self, input):
        results = []
        for i in range(len(self.downers)):
            cur_res, input = self.downers[i](input)
            results.append(cur_res)

        for i in range(len(self.uppers)):
            if i == 0:
                res = self.uppers[i](input)
            else:
                res = self.uppers[i](torch.concat([results[-i], res], dim=1))


        return self.end(torch.concat([results[0], res], dim=1))
                