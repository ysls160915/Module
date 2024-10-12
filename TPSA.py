import torch
from torch import nn


class RSPC(nn.Module):

    def __init__(self, channels, reduction=16):
        super(RSPC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


class TPSA(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(TPSA, self).__init__()
        self.conv_1 = nn.Conv3d(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = nn.Conv3d(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = nn.Conv3d(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = nn.Conv3d(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.rspc = RSPC(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)

        channels = feats.shape[1]
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3], feats.shape[4])

        x1_se = self.rspc(x1)
        x2_se = self.rspc(x2)
        x3_se = self.rspc(x3)
        x4_se = self.rspc(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        out = feats_weight[:, 0, :, :, :, :]
        for i in range(1, 4):
            out = torch.cat((out, feats_weight[:, i, :, :, :, :]), 1)

        return out

# 示例使用
if __name__ == '__main__':
    # 创建一个PSAModule3D模块，输入通道为64，输出通道为128
    psa_block = TPSA(inplans=64, planes=128)

    # 随机生成一个带有时间维度的输入特征图，形状为 (batch_size=1, channels=64, depth=16, height=64, width=64)
    input_tensor = torch.rand(1, 64, 16, 64, 64)

    # 通过PSAModule3D模块处理输入特征图
    output_tensor = psa_block(input_tensor)

    # 打印输入和输出的尺寸
    print(f"Input shape: {input_tensor.size()}")   # 输出: torch.Size([1, 64, 16, 64, 64])
    print(f"Output shape: {output_tensor.size()}") # 输出形状取决于你在模块中设置的planes，例如: torch.Size([1, 128, 16, 64, 64])