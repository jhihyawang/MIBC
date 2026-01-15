

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import conv3x3


'''
whole net without dilation
'''

class contralateralNN(nn.Module):
    def __init__(self, 
                 input_size = 256,
                 gn_groups: int = 16):
        super(contralateralNN, self).__init__()

        # self.pre_contra_norm = nn.GroupNorm(valid_groups(C, 16), C)

        g1 = self._valid_groups(input_size, gn_groups)
        g2 = self._valid_groups(input_size * 2, gn_groups)
        self.pre_contra_norm = nn.GroupNorm(g1, input_size)
        

        self.fc1 = nn.Sequential(
            nn.Conv2d(input_size * 2, input_size, kernel_size=1, bias=False, padding=0),
            nn.GroupNorm(g1, input_size),
            nn.SiLU(inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(input_size * 2, input_size * 2, kernel_size=1, bias=True),
            nn.GroupNorm(g2, input_size * 2),
        )


        self.gate = nn.Sequential(
            nn.Conv2d(input_size * 2, input_size * 2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.fc2[0].weight)
        nn.init.zeros_(self.fc2[0].bias)
        nn.init.constant_(self.gate[0].bias, -2.0)

    @staticmethod
    def _valid_groups(c: int, g: int) -> int:
        g = min(g, c)
        while c % g != 0 and g > 1:
            g -= 1
        return g
    

    def forward(self, l_x, r_x):
        l_x = self.pre_contra_norm(l_x)
        r_x = self.pre_contra_norm(r_x)

        diff = torch.abs(l_x - r_x)
        concat = torch.cat((l_x, r_x), dim=1)
        con_out = self.fc1(concat)

        # out = self.fc2(torch.cat((con_out, substract), dim=1))
        # l_x = l_x + out
        # r_x = r_x + out

        z = torch.cat([con_out, diff], dim=1)
        delta = self.fc2(z)                           # [B, 2C, H, W]
        g = self.gate(z)                              # [B, 2C, H, W]
        delta = delta * g

        delta_l, delta_r = torch.chunk(delta, 2, dim=1)
        l_x = l_x + delta_l
        r_x = r_x + delta_r
        

        return l_x, r_x
    

class BasicBlockV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride = stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = downsample
            
        self.stride = stride

    def forward(self, x):
        residual = x

        # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out
    


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class CSPBlockV2(nn.Module):
    """
    CSP block for ResNet-style backbones (split -> heavy/light -> concat -> fuse)

    """

    def __init__(self, in_c, out_c, stride=1, fuse_act=True):
        super().__init__()
        assert in_c % 2 == 0, f"in_c must be even for channel split, got {in_c}"
        assert out_c % 2 == 0, f"out_c must be even for concat halves, got {out_c}"

        mid_in = in_c // 2
        mid_out = out_c // 2

        # split projections
        # self.cv1 = ConvBNAct(in_c, mid_in, k=1, s=1, p=0, act=True)   # heavy
        # self.cv2 = ConvBNAct(in_c, mid_in, k=1, s=1, p=0, act=True)   # light

        self.cv1 = ConvBNAct(mid_in, mid_out, k=1, s=1, p=0, act=True)   # heavy
        self.cv2 = ConvBNAct(mid_in, mid_out, k=1, s=1, p=0, act=True)   # light

        # heavy branch: ResNet block(s)
        # IMPORTANT: BasicBlockV2 should handle stride internally on the heavy branch
        # self.block = BasicBlockV2(mid_in, mid_out, stride=stride)
        self.block = BasicBlockV2(mid_out, mid_out, stride=stride)

        # light branch alignment (downsample and/or channel align)
        # need_align = (stride != 1) or (mid_in != mid_out)
        # # self.align = ConvBNAct(mid_in, mid_out, k=1, s=stride, p=0, act=False) if need_align else nn.Identity()
        # self.align = ConvBNAct(mid_out, mid_out, k=1, s=stride, p=0, act=False) if need_align else nn.Identity()

        need_align = (stride != 1)
        self.align = ConvBNAct(mid_out, mid_out, k=1, s=stride, p=0, act=False) if need_align else nn.Identity()

        # fuse after concat
        # If you want "more ResNet-like", set fuse_act=False (i.e., BN only)
        self.fuse = ConvBNAct(out_c, out_c, k=1, s=1, p=0, act=False)

    def forward(self, x):

        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)

        x1 = self.cv1(x1)
        x1 = self.block(x1)

        x2 = self.cv2(x2)
        x2 = self.align(x2)

        out = torch.cat([x1, x2], dim=1)
        out = self.fuse(out)
        return out
    


class ViewResNetV3(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """
    def __init__(self,
                input_channels = 1, 
                num_filters = 16,
                first_layer_kernel_size = 7,
                first_layer_conv_stride = 2,
                blocks_per_layer_list=[2, 2, 2, 2, 2], 
                block_strides_list=[1, 2, 2, 2, 2],
                first_layer_padding=0,
                first_pool_size=3,
                first_pool_stride=2,
                first_pool_padding=0,
                growth_factor=2, 
                block_fn = None,
                kaiming_init = True):  # 新增參數
        
        super(ViewResNetV3, self).__init__()


        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=num_filters,
                kernel_size=first_layer_kernel_size,
                stride=first_layer_conv_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=first_pool_size,
                stride=first_pool_stride,
                padding=first_pool_padding,
            )
        )


        self.stages = nn.ModuleList()
        in_ch = num_filters
        for stage_idx in range(5):
            out_ch = num_filters * (growth_factor ** stage_idx)
            stride = block_strides_list[stage_idx]

            if stage_idx >=2:
                stage = nn.ModuleDict({
                    # first CSP block in stage
                    "csp1": CSPBlockV2(in_ch, out_ch, stride=stride),

                    # ipsilateral integration conv: cat(lc,lm)=2*out_ch -> out_ch
                    # and cat(rc,rm)=2*out_ch -> out_ch (share weights like your original)
                    # "ipsi_fuse": nn.Conv2d(2 * out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    "ipsi_fuse": ConvBNAct(2*out_ch, out_ch, k=1, s=1, p=0, act=True),

                    # second CSP block in stage (always stride=1 in your original)
                    "csp2": CSPBlockV2(out_ch, out_ch, stride=1),

                    # contralateral integration module (per stage)
                    "contra": contralateralNN(out_ch),
                })
            else:
                stage = nn.ModuleDict({
                    # first CSP block in stage
                    "csp1": CSPBlockV2(in_ch, out_ch, stride=stride),

                    # ipsilateral integration conv: cat(lc,lm)=2*out_ch -> out_ch
                    # and cat(rc,rm)=2*out_ch -> out_ch (share weights like your original)
                    # "ipsi_fuse": nn.Conv2d(2 * out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    "ipsi_fuse": ConvBNAct(2*out_ch, out_ch, k=1, s=1, p=0, act=True),

                    # second CSP block in stage (always stride=1 in your original)
                    "csp2": CSPBlockV2(out_ch, out_ch, stride=1),

                    # contralateral integration module (per stage)
                    "contra": None,
                })

            self.stages.append(stage)
            in_ch = out_ch

        # Kaiming 初始化
        if kaiming_init:
            self._kaiming_init_weights()
            print("✅ ViewResNetV3 Kaiming 初始化完成")

    def _kaiming_init_weights(self):
        """
        對 ViewResNetV3 進行 Kaiming 初始化
        
        - Conv2d: Kaiming Normal (fan_out, relu)
        - BatchNorm2d / GroupNorm: weight=1, bias=0
        - 保留 contralateralNN 的特殊初始化
        """
        for name, m in self.named_modules():
            # 跳過 contralateralNN 的特殊初始化層
            if 'contra' in name and ('fc2' in name or 'gate' in name):
                continue
                
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    @staticmethod
    def _ipsi_integrate(x_a, x_b, fuse_1x1):
        # concat then 1x1 fuse
        return fuse_1x1(torch.cat([x_a, x_b], dim=1))
    
    @staticmethod
    def _res_add_to_pair(integrated, x_a, x_b):
        # your pattern: x_a = integrated + x_a ; x_b = integrated + x_b
        return integrated + x_a, integrated + x_b
    

    def _stage_forward(self, stage, lc, lm, rc, rm):
        # 1) first CSP
        lc = stage["csp1"](lc)
        lm = stage["csp1"](lm)
        rc = stage["csp1"](rc)
        rm = stage["csp1"](rm)

        # 2) ipsilateral integration (left: lc+lm, right: rc+rm)
        left = self._ipsi_integrate(lc, lm, stage["ipsi_fuse"])
        right = self._ipsi_integrate(rc, rm, stage["ipsi_fuse"])

        # 3) second CSP + residual add back to each component
        left_int = stage["csp2"](left)
        right_int = stage["csp2"](right)

        lc, lm = self._res_add_to_pair(left_int, lc, lm)
        rc, rm = self._res_add_to_pair(right_int, rc, rm)

        # 4) contralateral integration (lc<->rc and lm<->rm)
        if stage["contra"] is not None:
            lc, rc = stage["contra"](lc, rc)
            lm, rm = stage["contra"](lm, rm)

        return lc, lm, rc, rm
    

    def forward(self, lc, lm, rc, rm):
        lc = self.stem(lc)
        lm = self.stem(lm)
        rc = self.stem(rc)
        rm = self.stem(rm)

        # stages
        for stage in self.stages:
            lc, lm, rc, rm = self._stage_forward(stage, lc, lm, rc, rm)

        return lc, lm, rc, rm




if __name__ == "__main__":
    model = ViewResNetV3(input_channels=3)
    lc = torch.randn(1, 3, 224, 224)
    lm = torch.randn(1, 3, 224, 224)
    rc = torch.randn(1, 3, 224, 224)
    rm = torch.randn(1, 3, 224, 224)

    # calculate params
    from thop import clever_format, profile
    macs, params = profile(model, inputs=(lc, lm, rc, rm))
    macs, params = clever_format([macs, params], "%.3f")
    print("MACs: ", macs)
    print("Params: ", params)
    print("Model successfully ran!")   #2.052M

    lc, lm, rc, rm = model(lc, lm, rc, rm)

    # save to pt file for netron
    # torch.save(model.state_dict(), "view_resnetv3.pt")

    # torch.save(model, "view_resnetv3.pt")
    

    



    print(y.shape)

    



        