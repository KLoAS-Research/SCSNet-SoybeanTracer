import torch
import torch.nn as nn
from nets.maxvit import MaxViT
from .attention import se_block, cbam_block, eca_block, SimAM

attention_blocks = [se_block, cbam_block, eca_block, SimAM]
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class multifeat(nn.Module):
    def __init__(self):
        super(multifeat, self).__init__()
        self.conv1 = nn.Conv2d(448, 64, kernel_size=3, padding=1)
        self.recover4 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.recover3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.recover2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention_blocks[0](64)
        self.simAttention = attention_blocks[3](64,64)

    def forward(self, up3, up2, up1):
        up3 = self.recover3(up3)
        up2 = self.recover2(up2)
        outputs1 = torch.cat([ up3, up2, up1], 1)
        outputs = self.conv1(outputs1)
        outputs = self.relu(outputs)
        outputs = self.simAttention(outputs)
        return outputs

class SCSNet(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(SCSNet, self).__init__()

        self.maxvit = MaxViT()

        self.featfusion = multifeat()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.GELU(),
        )

        in_filters = [192, 320, 384, 768]
        out_filters = [64, 128, 256, 512]

        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):

        # 448 448 64
        feat1 = self.stem(inputs)
        # 224 224 64
        feat2 = self.maxvit.stages[0](feat1)
        # 112 112 128
        feat3 = self.maxvit.stages[1](feat2)
        # 56 56 256
        feat4 = self.maxvit.stages[2](feat3)

        up3 = self.up_concat3(feat3, feat4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        output = self.featfusion(up3,up2,up1)

        final = self.final(output)
        return final

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()


