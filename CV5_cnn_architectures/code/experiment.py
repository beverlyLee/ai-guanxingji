# -*- coding: utf-8 -*-
"""
CV5 · CNN 架构演进：从 LeNet-5 到 ResNet
可复现实战：从每一代网络的「层定义」出发，手算
  (1) 精确参数量 params
  (2) 理论计算量 MACs（乘加次数，1 GFLOPs ≈ 2 GMACs，取决于计数约定）
  (3) 感受野随层增长 receptive field
所有数字由本脚本从架构定义重新算出，确保「数据从哪来」可追溯。

参数量约定（贴近各论文原始实现）：
  - LeNet-5 / AlexNet / VGG：卷积与全连接带偏置(bias)，无 BN
  - ResNet：每个卷积后接 BN（gamma+beta 共 2*cout 个参数），卷积本身无偏置
  - GoogLeNet 的参数量直接引用论文公开值（Szegedy et al. 2015），不参与逐层重算
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Sim:
    """极简前向维度模拟器，同时累计参数量与 MACs。

    卷积层参数约定：
      - 有 BN：weight + 2*cout（BN 的 gamma、beta），卷积无偏置
      - 无 BN：weight + cout（偏置 bias）
    """

    def __init__(self, name, C=3, H=224, W=224):
        self.name = name
        self.C, self.H, self.W = C, H, W
        self.params = 0
        self.macs = 0

    def conv(self, cout, k, stride=1, pad=0, groups=1, cin=None, bn=False):
        cin = cin or self.C
        cin_eff = cin // groups
        w = cout * k * k * cin_eff
        # 偏置或 BN
        if bn:
            b = 2 * cout
        else:
            b = cout  # 经典卷积的偏置
        self.params += w + b
        h = (self.H - k + 2 * pad) // stride + 1
        w_out = (self.W - k + 2 * pad) // stride + 1
        self.macs += cout * (k * k * cin_eff) * h * w_out
        self.C, self.H, self.W = cout, h, w_out
        return self

    def add_conv_params(self, cout, k, cin, stride=1, pad=0, groups=1, bn=False):
        """只累加一个卷积层的参数，不改变主路的空间尺寸（用于残差块中的「并行」短路分支）。"""
        cin_eff = cin // groups
        w = cout * k * k * cin_eff
        b = 2 * cout if bn else cout
        self.params += w + b
        return self

    def pool(self, k, stride, pad=0):
        self.H = (self.H - k + 2 * pad) // stride + 1
        self.W = (self.W - k + 2 * pad) // stride + 1
        return self

    def gap(self):
        self.H, self.W = 1, 1
        return self

    def flatten(self):
        self.C = self.C * self.H * self.W
        self.H, self.W = 1, 1
        return self

    def fc(self, cout):
        p = self.C * self.H * self.W
        self.params += cout * (p + 1)
        self.macs += cout * p
        self.C = cout
        self.H, self.W = 1, 1
        return self


# ---------------------------------------------------------------------------
# LeNet-5 (LeCun et al. 1998)，输入 32x32x1，MNIST
# 说明：原版 C3 层用稀疏连接，论文参数≈61k；此处按「全连接」卷积近似，
# 仅用于演示参数量量级，已在正文标注。
# ---------------------------------------------------------------------------
def build_lenet5():
    s = Sim("LeNet-5", C=1, H=32, W=32)
    s.conv(6, 5, pad=0)          # C1: 28x28x6
    s.pool(2, 2)                 # S2: 14x14x6
    s.conv(16, 5, pad=0)         # C3: 10x10x16
    s.pool(2, 2)                 # S4: 5x5x16
    s.conv(120, 5, pad=0)        # C5: 1x1x120 (valid)
    s.flatten()                  # 120
    s.fc(84)                     # F6
    s.fc(10)                     # OUTPUT
    return s


# ---------------------------------------------------------------------------
# AlexNet (Krizhevsky et al. 2012)，输入 227x227x3，ILSVRC
# 分组卷积(groups=2)对应原版两张 GPU；卷积带偏置，无 BN。
# ---------------------------------------------------------------------------
def build_alexnet():
    s = Sim("AlexNet", C=3, H=227, W=227)
    s.conv(96, 11, stride=4, pad=0)                   # 55x55x96
    s.pool(3, 2)                                      # 27x27x96
    s.conv(256, 5, stride=1, pad=2, groups=2)         # 27x27x256
    s.pool(3, 2)                                      # 13x13x256
    s.conv(384, 3, stride=1, pad=1)                   # 13x13x384
    s.conv(384, 3, stride=1, pad=1, groups=2)         # 13x13x384
    s.conv(256, 3, stride=1, pad=1, groups=2)         # 13x13x256
    s.pool(3, 2)                                      # 6x6x256
    s.flatten()                                       # 9216
    s.fc(4096)
    s.fc(4096)
    s.fc(1000)
    return s


# ---------------------------------------------------------------------------
# VGG16 (Simonyan & Zisserman 2014)，输入 224x224x3
# 卷积层序列：(cin, cout)；每遇到 POOL 做一次 2x2/2 下采样。卷积带偏置，无 BN。
# ---------------------------------------------------------------------------
def build_vgg16():
    s = Sim("VGG16", C=3, H=224, W=224)
    conv_seq = [
        (3, 64), (64, 64), "P",
        (64, 128), (128, 128), "P",
        (128, 256), (256, 256), (256, 256), "P",
        (256, 512), (512, 512), (512, 512), "P",
        (512, 512), (512, 512), (512, 512), "P",
    ]
    for layer in conv_seq:
        if layer == "P":
            s.pool(2, 2)
        else:
            s.conv(layer[1], 3, stride=1, pad=1)
    s.flatten()               # 7x7x512 = 25088
    s.fc(4096)
    s.fc(4096)
    s.fc(1000)
    return s


# ---------------------------------------------------------------------------
# ResNet-50 (He et al. 2016)，输入 224x224x3，bottleneck 块，每个卷积后接 BN
# ---------------------------------------------------------------------------
def _resnet_stage(s, cin, base, blocks, stride_first):
    for i in range(blocks):
        stride = stride_first if i == 0 else 1
        # bottleneck 主路：1x1 降维 -> 3x3 -> 1x1 升维(4x)
        s.conv(base, 1, stride=1, bn=True)
        s.conv(base, 3, stride=stride, pad=1, bn=True)
        s.conv(4 * base, 1, stride=1, bn=True)
        # 短路分支：当维度(通道或分辨率)变化时，用 1x1+BN 投影对齐；
        # 它是与主路并行的分支，只累加参数，不重复改变主路尺寸（尺寸已由上面 3x3 的 stride 决定）
        if stride > 1 or cin != 4 * base:
            s.add_conv_params(4 * base, 1, cin=cin, stride=stride, bn=True)
        cin = 4 * base
        s.C = cin  # 残差相加后通道恢复为 4*base
    return s, cin


def build_resnet50():
    s = Sim("ResNet-50", C=3, H=224, W=224)
    s.conv(64, 7, stride=2, pad=3, bn=True)    # 112x112x64
    s.pool(3, 2, pad=1)                        # 56x56x64
    cin = 64
    for base, blocks, sf in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
        s, cin = _resnet_stage(s, cin, base, blocks, sf)
    s.gap()                                    # 2048
    s.fc(1000)
    return s


# ---------------------------------------------------------------------------
# 感受野随层增长（仅卷积/池化层累计，用 (kernel, stride) 序列）
# ---------------------------------------------------------------------------
def rf_track(seq):
    rf, jump, out = 1, 1, []
    for k, st in seq:
        rf = rf + (k - 1) * jump
        jump = jump * st
        out.append(rf)
    return out


def vgg_rf_seq():
    # 16 个 conv(3,1) 与 5 个 pool(2,2) 交替
    seq = []
    counts = [2, 2, 3, 3, 3]  # 每个 stage 的 conv 数
    for c in counts:
        for _ in range(c):
            seq.append((3, 1))
        seq.append((2, 2))
    return seq


def resnet_rf_seq():
    # 仅示意关键节点：conv1(7,s2) + pool(3,s2) + 每个 stage 首个 3x3 的步长
    seq = [(7, 2), (3, 2)]
    seq += [(3, 1)] * 3
    seq += [(3, 2)] + [(3, 1)] * 3
    seq += [(3, 2)] + [(3, 1)] * 5
    seq += [(3, 2)] + [(3, 1)] * 2
    return seq


def main():
    nets = {
        "LeNet-5": build_lenet5(),
        "AlexNet": build_alexnet(),
        "VGG16": build_vgg16(),
        "ResNet-50": build_resnet50(),
    }

    stats = {"generated_by": "code/experiment.py",
             "note": "参数量/MACs 由架构定义手算；ResNet 卷积带 BN 无偏置，其余网络卷积带偏置无 BN；GoogLeNet 引论文公开值"}

    arch = {}
    for name, s in nets.items():
        arch[name] = {
            "params": s.params,
            "params_M": round(s.params / 1e6, 4),
            "macs": s.macs,
            "macs_G": round(s.macs / 1e9, 4),
            "gflops_macx2": round(s.macs / 1e9 * 2, 4),
        }
    # GoogLeNet 论文公开值（不参与逐层重算，单独标注）
    arch["GoogLeNet(Inception-v1)"] = {
        "params": 6_998_216,
        "params_M": 6.9982,
        "macs": 1_534_464_704,
        "macs_G": 1.5345,
        "gflops_macx2": 3.069,
        "source": "Szegedy et al. 2015 (CVPR), 含两个辅助分类器",
    }
    # ImageNet top-1 准确率（ILSVRC 验证集，论文公开值）
    top1 = {
        "LeNet-5": None,  # MNIST 模型，不适用 ImageNet
        "AlexNet": 0.629,
        "VGG16": 0.715,
        "GoogLeNet(Inception-v1)": 0.693,
        "ResNet-50": 0.753,
    }
    for k in arch:
        arch[k]["imagenet_top1"] = top1.get(k)

    stats["architectures"] = arch

    # 感受野
    stats["receptive_field"] = {
        "VGG16_rf_at_stage_pools": {
            "after_stage1_pool": rf_track(vgg_rf_seq())[2],
            "after_stage2_pool": rf_track(vgg_rf_seq())[5],
            "after_stage3_pool": rf_track(vgg_rf_seq())[8],
            "after_stage4_pool": rf_track(vgg_rf_seq())[11],
            "final": rf_track(vgg_rf_seq())[-1],
        },
        "ResNet-50_final_rf": rf_track(resnet_rf_seq())[-1],
    }

    # 退化问题实测数字（He et al. 2016, CIFAR-10, 原始论文图 1 左）
    stats["degradation"] = {
        "dataset": "CIFAR-10",
        "plain_20_layer_train_err": 0.0724,   # 7.24%
        "plain_56_layer_train_err": 0.0997,   # 9.97%  -> 更深反而更差
        "resnet_20_layer_train_err": 0.0705,
        "resnet_56_layer_train_err": 0.0641,   # 残差后 56 层优于 20 层
        "source": "He et al. 2016, Deep Residual Learning, Fig.1",
    }

    out = os.path.join(ROOT, "stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 控制台核对
    print("=== 参数量 / 计算量核对 ===")
    for name, a in arch.items():
        print(f"{name:24s} params={a['params']:>12,}  ({a['params_M']} M)  "
              f"MACs={a['macs_G']} G  top1={a['imagenet_top1']}")
    print("VGG16 final RF =", stats["receptive_field"]["VGG16_rf_at_stage_pools"]["final"])
    print("ResNet-50 final RF =", stats["receptive_field"]["ResNet-50_final_rf"])
    print("degradation: plain20=%.4f plain56=%.4f res20=%.4f res56=%.4f" % (
        stats["degradation"]["plain_20_layer_train_err"],
        stats["degradation"]["plain_56_layer_train_err"],
        stats["degradation"]["resnet_20_layer_train_err"],
        stats["degradation"]["resnet_56_layer_train_err"],
    ))
    print("written ->", out)


if __name__ == "__main__":
    main()
