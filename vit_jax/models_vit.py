# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Tuple, Type

# Flax 是基于 JAX 的深度学习库
# linen 是 Flax 的一个子模块，用于构建深度学习模型, 类似 PyTorch 中的 torch.nn
import flax.linen as nn
# 导入 JAX 的 Numpy 接口，语法与 NumPy 几乎一致，但支持自动微分、GPU/TPU 加速
import jax.numpy as jnp

from vit_jax import models_resnet


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""
  """恒等变换层，不改变输入，仅用于在计算图中打标签"""

  # @nn.compact：Flax 装饰器，允许在 __call__ 中直接定义子模块或参数。
  @nn.compact
  def __call__(self, x):
    """直接返回输入"""
    return x


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer.
  """
  """为输入序列添加科学系的位置编码
  
  Args:
    posemb_init: 初始化函数
    param_dtype: 参数数据类型
  """

  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    """Applies the AddPositionEmbs module.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    # 校验输入形状
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    # 位置编码形状 (1, seq_len, emb_dim)
    # 第0维是1原因: 位置编码是全局的，不需要考虑 batch_size
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    # 创建一个可训练参数，四个参数分别为: 名称、初始化器、形状、数据类型
    pe = self.param(
        'pos_embedding', self.posemb_init, pos_emb_shape, self.param_dtype)
    # 输入 + 位置编码
    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  """
  Transformer MLP

  Args:
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    param_dtype: the dtype of the parameters (default: float32).
    out_dim: output dimension.
    dropout_rate: dropout rate.
    kernel_init: initializer for dense layer kernels.
    bias_init: initializer for dense layer biases.
  """

  mlp_dim: int
  dtype: Dtype = jnp.float32
  param_dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    """
    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer mlp block.
    """
    """
    Args:
      inputs: shape [batch_size, seq_len, emb_dim]
    """
    # 确定最终输出维度 (如果每指定则取输入的最后一维)
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    # 第1个全连接层: emb_dim -> mlp_dim
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            inputs)
    # 使用 GELU 激活函数（Vit/BERT标配）
    x = nn.gelu(x)
    # 应用 dropout, 由 deterministic 控制是否生效
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    # 第2个全连接层: mlp_dim -> actual_out_dim
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            x)
    # 再次 dropout
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """
  """
  单个 Transformer Encoder 层
  """

  # MLP 层维度
  mlp_dim: int
  # 注意力头数
  num_heads: int
  # 数据类型
  dtype: Dtype = jnp.float32
  # dropout 比例(用于 MLP 后)
  dropout_rate: float = 0.1
  # 注意力 dropout 比例
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    # 验证输入维度
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    # 层归一化
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    # 多头注意力 (输出形状与输入相同)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,      # 每个head独立应用dropout
        deterministic=deterministic,  # 是否应用dropout
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x, x)
    # dropout
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    # 残差连接
    x = x + inputs

    # MLP block.
    # 层归一化
    y = nn.LayerNorm(dtype=self.dtype)(x)
    # MLP
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)
    # 残差连接
    return x + y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """

  # 层数
  num_layers: int
  # MLP 维度
  mlp_dim: int
  # 注意力头数
  num_heads: int
  # dropout 比例
  dropout_rate: float = 0.1
  # 注意力 dropout 比例
  attention_dropout_rate: float = 0.1
  # 是否添加位置编码
  add_position_embedding: bool = True

  @nn.compact
  def __call__(self, x, *, train):
    """Applies Transformer model on the inputs.

    Args:
      x: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    # 验证输入维度
    assert x.ndim == 3  # (batch, len, emb)

    # 添加位置编码
    if self.add_position_embedding:
      x = AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(
              x)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # 经过N层Encoder
    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads)(
              x, deterministic=not train)
    # 层归一化
    encoded = nn.LayerNorm(name='encoder_norm')(x)

    return encoded


class VisionTransformer(nn.Module):
  """VisionTransformer."""

  # 分类数量
  num_classes: int
  # 应包含.size, 如(16, 16)
  patches: Any
  # dict: {num_layers, mlp_dim, num_heads, ...}
  transformer: Any
  # patch embedding 维度
  hidden_size: int
  # 若提供则先使用 ResNet 提取特征
  resnet: Optional[Any] = None
  # pre-logits 中间层
  representation_size: Optional[int] = None
  # 'token', 'gap', 'unpooled'
  classifier: str = 'token'
  head_bias_init: float = 0.
  # 可替换 Encoder 实现
  encoder: Type[nn.Module] = Encoder
  model_name: Optional[str] = None

  @nn.compact
  def __call__(self, inputs, *, train):

    x = inputs
    
    # Step1 可选 ResNet
    # (Possibly partial) ResNet root.
    if self.resnet is not None:
      # ResNet 初始通道数
      width = int(64 * self.resnet.width_factor)

      # ResNet 标准 stem：7x7 卷积 + GroupNorm + ReLU + MaxPool
      # “ResNet 标准 stem” 是计算机视觉领域中的一个术语，特指 ResNet（残差网络）在主干残差块（residual blocks）之前用于初步处理输入图像的初始卷积层序列。它的作用是快速下采样（降低分辨率）并提取低级特征，为后续的深层残差结构提供合适的输入。
      # Root block.
      x = models_resnet.StdConv(
          features=width,
          kernel_size=(7, 7),
          strides=(2, 2),
          use_bias=False,
          name='conv_root')(
              x)
      x = nn.GroupNorm(name='gn_root')(x)
      x = nn.relu(x)
      x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

      # 构建多个 ResNet stage（具体实现在 models_resnet.py）
      # ResNet stages.
      if self.resnet.num_layers:
        x = models_resnet.ResNetStage(
            block_size=self.resnet.num_layers[0],
            nout=width,
            first_stride=(1, 1),
            name='block1')(
                x)
        for i, block_size in enumerate(self.resnet.num_layers[1:], 1):
          x = models_resnet.ResNetStage(
              block_size=block_size,
              nout=width * 2**i,
              first_stride=(2, 2),
              name=f'block{i + 1}')(
                  x)

    # 获取当前特征图形状 (batch, height, width, channels)
    n, h, w, c = x.shape

    # Patch Embedding
    # 用卷积实现 patch 分割+线性投影
    # 1. 把输入图像(或特征图)切成不重叠的图像块(patches)
    # 2. 每个 patch 被线性投影到一个 hidden_size 维向量
    # 3. 输出是一个 (batch, height/kernel_size, height/kernel_size, hidden_size) 的矩阵
    # 虽然 ViT 论文中描述为 flatten each patch and apply a linear projection, 但用 strides=self.patches.size 的卷积等价于这一操作，且更高效
    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        features=self.hidden_size,      # 输出维度, 即嵌入维度
        kernel_size=self.patches.size,  # 卷积核的空间尺寸, 如(16, 16)
        strides=self.patches.size,      # 卷积步长, 如(16, 16), 与卷积核的空间尺寸共同作用实现无重叠地滑动卷积核, 正好覆盖整个图像
        padding='VALID',                # 不填充
        name='embedding')(              # 给这个卷积层命名, 便于调试/可视化计算图/加载与保存与训练权重
            x)

    # Here, x is a grid of embeddings.

    # (Possibly partial) Transformer.
    if self.transformer is not None:
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])

      # 添加 Class Token
      # If we want to add a class token, add it here.
      if self.classifier in ['token', 'token_unpooled']:
        # 创建可学习的 class token, 形状为 (1, 1, c)
        cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
        # 广播到 batch
        cls = jnp.tile(cls, [n, 1, 1])
        # 拼接到序列开头, 形状为 (batch, seq_len + 1, c)
        x = jnp.concatenate([cls, x], axis=1)

      # Transformer Encoder 处理
      x = self.encoder(name='Transformer', **self.transformer)(x, train=train)

    # Step 6 分类策略
    if self.classifier == 'token':
      # 取第0个token, 即 class token
      x = x[:, 0]
    elif self.classifier == 'gap':
      # 对所有token做平均
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    elif self.classifier in ['unpooled', 'token_unpooled']:
      pass
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    # Step7 表示层
    if self.representation_size is not None:
      # 如果指定了 representation_size，加一个 Dense + tanh
      x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
      # 应用 tanh 激活函数
      x = nn.tanh(x)
    else:
      # 用 IdentityLayer 打标签（不影响值）
      x = IdentityLayer(name='pre_logits')(x)

    # Step 8 分类头
    if self.num_classes:
      x = nn.Dense(
          features=self.num_classes,
          name='head',
          kernel_init=nn.initializers.zeros,
          bias_init=nn.initializers.constant(self.head_bias_init))(x)
    # Step 9 输出
    return x
