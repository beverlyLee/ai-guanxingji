#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M5 · Keras 框架入门实验
==========================

用 Fashion-MNIST（Zalando 公开服装数据集，7 万张 28x28 灰度图，10 类：
T 恤/裤/套头衫/连衣裙/外套/凉鞋/运动鞋/包/短靴/衬衫）实测 Keras 训练自己数据集的
完整流程，并逐项对照：

  EXP_A  数据加载与预处理 + 搭建网络（流程基线）
  EXP_B  学习率对结果的影响（扫描 1e-4 / 1e-3 / 1e-2 / 1e-1）
  EXP_C  权重初始化方法对比（glorot_uniform / he_normal / random_normal）
  EXP_D  初始化标准差对结果的影响（random_normal std = 0.01/0.05/0.1/0.5）
  EXP_E  过拟合治理：无正则 / Dropout / L2 正则化的训练-验证曲线对照

所有数字写入 ../stats.json，供正文与配图复用。本机：Apple M3 / 16GB，
Python 3.13.12，keras 3 + jax 后端（CPU）。

运行：
  .venv/bin/python code/model_keras.py
"""

import os
import json
import time

os.environ["KERAS_BACKEND"] = "jax"  # 轻量 CPU 可用，无需装 tensorflow

import numpy as np
import keras
from keras import layers, regularizers
from keras.datasets import fashion_mnist

SEED = 42
keras.utils.set_random_seed(SEED)  # 同时置 numpy / python / keras 全局随机种子
np.random.seed(SEED)

# ---------- 1. 数据加载与预处理（EXP_A 的一部分）----------
t0 = time.time()
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_train_all = x_train_all.astype("float32") / 255.0   # 归一化到 [0,1]
x_test = x_test.astype("float32") / 255.0
y_train_all = keras.utils.to_categorical(y_train_all, 10)  # one-hot
y_test = keras.utils.to_categorical(y_test, 10)
load_time = round(time.time() - t0, 2)

N_FULL = x_train_all.shape[0]            # 60000
IMG = x_train_all.shape[1]              # 28
INPUT_DIM = IMG * IMG                   # 784
N_CLASS = 10

# Fashion-MNIST 官方类别顺序（务必与标签索引对应）
CLASS_NAMES = ["T恤", "裤子", "套头衫", "连衣裙", "外套",
               "凉鞋", "衬衫", "运动鞋", "包", "短靴"]

# 小样本子集：用于扫描实验提速（诚实标注为示例量级）
SUB = 20000
x_sub, y_sub = x_train_all[:SUB], y_train_all[:SUB]

# 过拟合演示用极小子集：制造明显训练-验证鸿沟
TINY = 5000
x_tiny, y_tiny = x_train_all[:TINY], y_train_all[:TINY]


def build_mlp(hidden=(256, 128), init="glorot_uniform", dropout=0.0,
              l2=0.0, lr=1e-3):
    """搭建一个标准 MLP：Flatten(784)->Dense(256)->Dense(128)->Dense(10)。"""
    model = keras.Sequential()
    model.add(layers.Input((INPUT_DIM,)))
    model.add(layers.Reshape((IMG, IMG, 1)))  # 仅为可视化形状变化，下面展平
    model.add(layers.Flatten())
    reg = regularizers.l2(l2) if l2 > 0 else None
    for h in hidden:
        model.add(layers.Dense(h, activation="relu", kernel_initializer=init,
                               kernel_regularizer=reg))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(N_CLASS, activation="softmax", kernel_initializer=init,
                           kernel_regularizer=reg))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def shape_report():
    """打印数据在网络里的形状变化，供正文引用。"""
    m = build_mlp()
    m.build((None, INPUT_DIM))
    shapes = []
    for layer in m.layers:
        try:
            shapes.append((layer.name, tuple(layer.output.shape)))
        except Exception:
            shapes.append((layer.name, None))
    return shapes


def run(name, x, y, epochs, init="glorot_uniform", dropout=0.0, l2=0.0, lr=1e-3,
       verbose=0):
    model = build_mlp(init=init, dropout=dropout, l2=l2, lr=lr)
    hist = model.fit(x.reshape(-1, INPUT_DIM), y,
                     validation_split=0.1, batch_size=128, epochs=epochs,
                     verbose=verbose)
    vloss, vacc = model.evaluate(x_test.reshape(-1, INPUT_DIM), y_test, verbose=0)
    h = hist.history
    return {
        "name": name,
        "epochs": epochs,
        "final_train_acc": round(float(h["accuracy"][-1]), 4),
        "final_val_acc": round(float(h["val_accuracy"][-1]), 4),
        "final_val_loss": round(float(h["val_loss"][-1]), 4),
        "test_acc": round(float(vacc), 4),
        "train_acc_hist": [round(float(v), 4) for v in h["accuracy"]],
        "val_acc_hist": [round(float(v), 4) for v in h["val_accuracy"]],
        "loss_hist": [round(float(v), 4) for v in h["loss"]],
    }


stats = {}
stats["meta"] = {
    "seed": SEED,
    "backend": "jax",
    "keras_version": keras.__version__,
    "dataset": "Fashion-MNIST (Zalando, 公开)",
    "n_train_full": N_FULL,
    "n_test": x_test.shape[0],
    "img_size": f"{IMG}x{IMG}",
    "input_dim": INPUT_DIM,
    "n_class": N_CLASS,
    "class_names": CLASS_NAMES,
    "load_time_s": load_time,
    "preprocess": "astype float32 /255.0 归一化；to_categorical one-hot 10 类",
    "subset_for_sweep": SUB,
    "tiny_for_overfit": TINY,
    "note": "扫描实验用 20000 样本提速，过拟合演示用 5000 样本制造鸿沟；均标注为示例量级",
}

# EXP_A 形状变化报告
stats["EXP_A_shape"] = {
    "model": "Sequential: Input(784)->Reshape->Flatten->Dense256(relu)->Dense128(relu)->Dense10(softmax)",
    "layer_shapes": shape_report(),
    "baseline_20k_12ep": run("baseline", x_sub, y_sub, epochs=12, lr=1e-3),
}

# EXP_B 学习率扫描
lrs = [1e-4, 1e-3, 1e-2, 1e-1]
stats["EXP_B_lr"] = {"lrs": lrs, "runs": {}}
for lr in lrs:
    stats["EXP_B_lr"]["runs"][str(lr)] = run(f"lr={lr}", x_sub, y_sub, epochs=12, lr=lr)

# EXP_C 初始化方法对比
inits = ["glorot_uniform", "he_normal", "random_normal"]
stats["EXP_C_init"] = {"methods": inits, "runs": {}}
for it in inits:
    stats["EXP_C_init"]["runs"][it] = run(f"init={it}", x_sub, y_sub, epochs=12,
                                          init=it, lr=1e-3)

# EXP_D 初始化标准差影响（random_normal 不同 std）
stds = [0.01, 0.05, 0.1, 0.5]
stats["EXP_D_init_std"] = {"stds": stds, "runs": {}}
for s in stds:
    stats["EXP_D_init_std"]["runs"][str(s)] = run(
        f"std={s}", x_sub, y_sub, epochs=12, init=keras.initializers.RandomNormal(s), lr=1e-3)

# EXP_E 过拟合治理（小样本 30 epoch，明显鸿沟）
stats["EXP_E_overfit"] = {"epochs": 30, "runs": {}}
stats["EXP_E_overfit"]["runs"]["none"] = run("no_reg", x_tiny, y_tiny, epochs=30,
                                             lr=1e-3)
stats["EXP_E_overfit"]["runs"]["dropout"] = run("dropout", x_tiny, y_tiny, epochs=30,
                                                lr=1e-3, dropout=0.4)
stats["EXP_E_overfit"]["runs"]["l2"] = run("l2", x_tiny, y_tiny, epochs=30,
                                           lr=1e-3, l2=1e-4)

with open(os.path.join(os.path.dirname(__file__), "..", "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("DONE. 写入 stats.json")
print("基线 20k/12ep test_acc =", stats["EXP_A_shape"]["baseline_20k_12ep"]["test_acc"])
print("学习率 test_acc:", {k: v["test_acc"] for k, v in stats["EXP_B_lr"]["runs"].items()})
print("初始化 test_acc:", {k: v["test_acc"] for k, v in stats["EXP_C_init"]["runs"].items()})
print("初始化std test_acc:", {k: v["test_acc"] for k, v in stats["EXP_D_init_std"]["runs"].items()})
print("过拟合 none/dropout/l2 val_acc:",
      {k: v["final_val_acc"] for k, v in stats["EXP_E_overfit"]["runs"].items()})
