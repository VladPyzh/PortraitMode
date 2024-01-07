import torch.nn as nn

class ConvBatchMish(nn.Module):
    def __init__(self, channels, padding):
        super().__init__()
        self.padding = padding
        self.layers = nn.Sequential(*[nn.Conv2d(channels, channels, kernel_size=3, padding=padding),
                                      nn.BatchNorm2d(channels),
                                      nn.Mish(),
                                      nn.Conv2d(channels, channels, kernel_size=3, padding=padding),
                                      nn.BatchNorm2d(channels),
                                      nn.Mish(),
                                      nn.Conv2d(channels, channels, kernel_size=3, padding=padding),
                                      nn.BatchNorm2d(channels),
                                      nn.Mish()])
    def forward(self, input):
        if self.padding == 0:
            return self.layers(input)
        return input + self.layers(input)

class EncodDecod(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base = model

        self.base.conv_head = nn.Identity()
        self.base.bn2 = nn.Identity()
        self.base.global_pool = nn.Identity()
        self.base.classifier = nn.Identity()
        
        self.decoder = []
        start = 384
        denom  = 1
        for i in range(5):
            self.decoder.append(ConvBatchMish(start // denom, padding=1))
            if i % 3 == 0:
                self.decoder.append(nn.ConvTranspose2d(in_channels=start // denom,
                                                  out_channels=start // (denom * 2),
                                                  kernel_size=2,
                                                  stride=2))
                denom *= 2
            else:
                self.decoder.append(nn.ConvTranspose2d(in_channels=start // denom,
                                                  out_channels=start // denom,
                                                  kernel_size=2,
                                                  stride=2))
                

        self.decoder.append(nn.Conv2d(start // denom, start // denom, kernel_size=3, padding=1))
        self.decoder.append(nn.BatchNorm2d(start // denom))
        self.decoder.append(nn.Mish())
        self.decoder.append(nn.Conv2d(start // denom, 2, kernel_size=1))

        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, input):
        encoded = self.base(input)
        return self.decoder(encoded)