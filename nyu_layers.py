# NYU breast_cancer_classifier ResNet22 layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3

class BasicBlockV2(nn.Module):
    """
    Adapted from NYU breast_cancer_classifier, converted to v2
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

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


class ViewResNetV2(nn.Module):
    """
    NYU ResNet22 architecture adapted from torchvision ResNet, converted to v2
    """
    def __init__(self,
                 input_channels, num_filters,
                 first_layer_kernel_size, first_layer_conv_stride,
                 blocks_per_layer_list, block_strides_list, block_fn,
                 first_layer_padding=0,
                 first_pool_size=None, first_pool_stride=None, first_pool_padding=0,
                 growth_factor=2):
        super(ViewResNetV2, self).__init__()
        self.first_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False,
        )
        self.first_pool = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding,
        )

        self.layer_list = nn.ModuleList()
        current_num_filters = num_filters
        self.inplanes = num_filters
        for i, (num_blocks, stride) in enumerate(zip(
                blocks_per_layer_list, block_strides_list)):
            self.layer_list.append(self._make_layer(
                block=block_fn,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride,
            ))
            current_num_filters *= growth_factor
        self.final_bn = nn.BatchNorm2d(
            current_num_filters // growth_factor * block_fn.expansion
        )
        self.relu = nn.ReLU()

        # Expose attributes for downstream dimension computation
        self.num_filters = num_filters
        self.growth_factor = growth_factor

    def forward(self, x):
        h = self.first_conv(x)
        h = self.first_pool(h)
        for i, layer in enumerate(self.layer_list):
            h = layer(h)
        h = self.final_bn(h)
        h = self.relu(h)
        return h

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
        )

        layers_ = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_)


def kaiming_init_resnet22(model):
    """
    對 ResNet22 模型進行 Kaiming 初始化
    
    使用 He initialization（Kaiming initialization）:
    - Conv2d: Kaiming Normal，適用於 ReLU
    - BatchNorm2d: weight=1, bias=0
    - Linear: Kaiming Normal
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    print("✅ ResNet22 Kaiming 初始化完成")
    return model


def resnet22_nyu(input_channels=1, kaiming_init=True):
    """
    NYU ResNet22 for mammography - 確實的架構實現
    
    Layers計算:
    - First conv: 1 層
    - 5 stages × 2 blocks × 2 convs = 20 層  
    - Final BN: 1 層
    Total: 22 層
    
    Args:
        input_channels: 輸入通道數 (預設為 1，灰階)
        kaiming_init: 是否使用 Kaiming 初始化 (預設為 True)
    """
    model = ViewResNetV2(
        input_channels=input_channels,
        num_filters=16,                        # 起始 filter 數量
        first_layer_kernel_size=7,             # 第一層 kernel 大小  
        first_layer_conv_stride=2,             # 第一層 stride
        blocks_per_layer_list=[2, 2, 2, 2, 2], # 每層的 block 數量
        block_strides_list=[1, 2, 2, 2, 2],    # 每層的 stride
        block_fn=BasicBlockV2,
        first_layer_padding=0,                 # 原始實現使用 "valid" padding
        first_pool_size=3,                     # MaxPool 大小
        first_pool_stride=2,                   # MaxPool stride
        first_pool_padding=0,                  # MaxPool padding (valid)
        growth_factor=2                        # Filter 增長倍數: 16→32→64→128→256
    )
    
    if kaiming_init:
        model = kaiming_init_resnet22(model)
        print('✅ ResNet22 Kaiming 初始化完成')
        return model
    
    return model


def load_nyu_pretrained_weights(model, weights_path, view='L-CC'):
    """
    載入 NYU breast cancer classifier 的預訓練權重到 ResNet22 模型
    
    Args:
        model: 我們的 ResNet22 模型實例
        weights_path: NYU 權重檔案路徑 (.p)
        view: 視角 (L-CC, R-CC, L-MLO, R-MLO)，僅用來決定載入 cc 或 mlo 的權重
        
    Returns:
        model: 載入權重後的模型
    """
    import torch
    import pickle
    
    print(f"開始載入 NYU 權重檔案: {weights_path}")
    
    # Load weights file using torch.load (NYU 使用這種格式)
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']
        print(f"成功載入 NYU 權重檔案，包含 {len(state_dict)} 個參數")
    except Exception as e:
        print(f"載入權重檔案失敗: {e}")
        return model
    
    # 決定要載入哪個視角的權重 (cc 或 mlo)
    view_angle = view.lower().split("-")[-1]  # cc or mlo
    if view_angle not in ['cc', 'mlo']:
        print(f"警告: 視角 {view} 不被支援，使用 cc 視角權重")
        view_angle = 'cc'
        
    view_prefix = f"four_view_resnet.{view_angle}."
    print(f"載入視角: {view_angle}")
    
    # 提取對應視角的權重並移除前綴
    filtered_dict = {}
    for key, value in state_dict.items():
        if key.startswith(view_prefix):
            # 移除前綴，例如: four_view_resnet.cc.first_conv.weight -> first_conv.weight
            new_key = key.replace(view_prefix, "")
            filtered_dict[new_key] = value
    
    print(f"找到 {len(filtered_dict)} 個匹配的權重參數")
    
    if len(filtered_dict) == 0:
        print(f"錯誤: 沒有找到視角 {view_angle} 的權重")
        print(f"可用的權重前綴: {set(k.split('.')[0] + '.' + k.split('.')[1] + '.' for k in state_dict.keys() if 'four_view_resnet' in k)}")
        return model
    
    # 載入權重到模型
    try:
        missing_keys, unexpected_keys = model.load_state_dict(filtered_dict, strict=False)
        
        print(f"✅ 成功載入 NYU ResNet22 權重")
        if missing_keys:
            print(f"   缺少的 keys: {len(missing_keys)} 個")
            print(f"   前5個: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"   多餘的 keys: {len(unexpected_keys)} 個") 
            print(f"   前5個: {unexpected_keys[:5]}")
            
        # 檢查一些關鍵層是否成功載入
        key_layers = ['first_conv.weight', 'layer_list.0.0.conv1.weight']
        loaded_layers = [k for k in key_layers if k in filtered_dict]
        print(f"   關鍵層載入狀況: {len(loaded_layers)}/{len(key_layers)}")
        
    except Exception as e:
        print(f"載入權重到模型失敗: {e}")
    
    return model