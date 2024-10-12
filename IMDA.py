import torch
import torch.nn as nn
from thop import profile  # 引入thop库来计算模型的FLOPs和参数数量

# 定义IMDA模块
class IMDA_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(IMDA_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1) # 用于将输入数据的空间维度（深度、高度、宽度）缩减到1
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.Sigmoid()  # 使用Sigmoid激活函数
        self.e_lambda = e_lambda  # 定义平滑项e_lambda，防止分母为0

    def forward(self, x):
        b, c, d, h, w = x.size()  # 获取输入x的尺寸
        n = d * w * h - 1  # 计算特征图的元素数量减一，用于下面的归一化

        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)

        spational_channel = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5

        b, c, d, h, w = spational_channel.size()
        spational_channel = self.avg_pool(x).view(b, c, -1)
        spational_channel = self.conv(spational_channel.transpose(1, 2))  # Shape: [batch, 1, channel]
        spational_channel = self.sigmoid(spational_channel).transpose(1, 2)  # Shape: [batch, channel, 1]
        spational_channel = spational_channel.view(b, c, 1, 1, 1)  # [batch, channel, 1, 1, 1]
        # 返回经过时间注意力加权的输入特征
        Temporal = x * spational_channel.expand_as(x)
        return x * self.act(Temporal)


# 示例使用
if __name__ == '__main__':
    model = IMDA_module().cuda()  # 实例化SimAM模块并移到GPU上
    x = torch.randn(1, 64, 5, 64, 64).cuda()  # 创建一个随机输入并移到GPU上，注意这里的深度设置为5
    y = model(x)  # 将输入传递给模型
    print(y.size())  # 打印输出尺寸
    # # 使用thop库计算模型的FLOPs和参数数量
    # flops, params = profile(model, inputs=(x,))
    # print(flops / 1e9)  # 打印以Giga FLOPs为单位的浮点操作数
    # print(params)  # 打印模型参数数量