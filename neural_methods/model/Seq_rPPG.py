import torch
import torch.nn as nn
import torch.fft as fft


class SpectralTransform(nn.Module):

    def __init__(self, c, size=3):
        super(SpectralTransform, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=c, out_channels=c*2, kernel_size=size, stride=1, padding=size//2),
            nn.BatchNorm1d(c*2),
            nn.ReLU()
        )

    def forward(self, y):
        x = torch.transpose(y, 1, 2)
        x = fft.rfft(x)
        r, i = torch.real(x), torch.imag(x)
        x = torch.cat([r, i], dim=-2)
        x = torch.transpose(x, 1, 2)
        x = self.conv1(x)
        x = torch.transpose(x, 1, 2)
        r, i = torch.split(x, x.size(-2) // 2, dim=-2)
        x = torch.complex(r, i)
        x = fft.irfft(x)
        x = torch.transpose(y, 1, 2)
        return x + y

class Seq_rPPG(nn.Module):
    
    def __init__(self):
        super(Seq_rPPG, self).__init__()
        self.a = nn.Sequential(
            nn.Flatten(),
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=3)
        )
        self.ST1 = SpectralTransform(64, 5)
        self.conv1_1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=10, stride=1, padding=10//2)
        self.atv1 = nn.Sequential(
            nn.BatchNorm1d(64), 
            nn.ReLU()
        )
        self.ST2 = SpectralTransform(64, 3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=5//2)
        self.atv2 = nn.Sequential(
            nn.BatchNorm1d(32), 
            nn.ReLU()
        )
        self.z = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1, stride=1),
            nn.Flatten(start_dim=0, end_dim=-1)
        )
        self.w = {}

    def forward(self, x):
        x = (x - torch.reshape(torch.mean(x, dim=(1, 2, 3)), (-1, 1, 1, 1))) / torch.reshape(torch.std(x, dim=(1, 2, 3)), (-1, 1, 1, 1))
        x = x.permute(0, 1, 4, 2, 3)
        x = self.a(x)
        x = self.ST1(x)
        x = self.conv1_1(x)
        x = self.atv1(x)
        x = self.ST2(x)
        x = self.conv2(x)
        x = self.atv2(x)
        return self.z(x)

    def cross(self):
        if 'bn' not in self.w:
            self.w['bn'] = self.atv1, self.atv2
            self._ = nn.Sequential(
                nn.BatchNorm1d(64),
                nn.ReLU()
            ), nn.Sequential(
                nn.BatchNorm1d(32),
                nn.ReLU()
            )
            self._[0].load_state_dict(self.atv1.state_dict())
            self._[1].load_state_dict(self.atv2.state_dict())
            self.atv1 = lambda x: self._[0](x)
            self.atv2 = lambda x: self._[1](x)
    
    def intra(self):
        if 'bn' in self.w:
            self.atv1, self.atv2 = self.w['bn']
            self.w.clear()