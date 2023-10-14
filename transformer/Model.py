"""
transformer
三类应用
    1. 机器翻译应用-Encoder和Decoder共同使用
    2. 文本分类BERT和图片分类VIT-只是用Encoder端
    3. 生成类模型-只是用Decoder端
"""
import numpy as np
import copy
# 机器学习包
import torch
import torch.nn as nn
import torch.nn.functional as F
# 数学包
import math
from torch.autograd import Variable


# 定义Embedding类实现文本嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        Embedding类的初始化函数
        :param d_model: 词嵌入的维度
        :param vocab: 词表的大小
        """
        super(Embeddings, self).__init__()
        # 获得词嵌入对象
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Embedding层的前向传播
        :param x: 输入给模型的文本经过映射后的张量
        :return: 词嵌入向量 (样本数, max_len, d_model)
        """

        return self.embedding(x) * math.sqrt(self.d_model)


# 定义位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        位置编码器的初始化函数
        :param d_model: 词嵌入的维度
        :param dropout: dropout就是dropout还能是啥
        :param max_len: 每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 定义位置编码矩阵，它是一个0真，矩阵的大小为max_len行，d_model列
        position_coding_matrix = torch.zeros(max_len, d_model)

        # 初始化绝对位置矩阵，在这里使用词汇的绝对位置索引进行表示
        position = torch.arange(0, max_len).unsqueeze(1)

        """
        这里应该考虑如何将绝对位置加入到位置编码矩阵中
        最简单的思路是将 max_len x 1 的绝对位置矩阵变换成 max_len x d_model的形状然后覆盖掉原来的位置矩阵
        则我们需要 1 x max_len的矩阵将其变换
        """
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))

        # 分别将值赋值给偶数位和奇数位
        position_coding_matrix[:, 0::2] = torch.sin(position * div_term)
        position_coding_matrix[:, 1::2] = torch.cos(position * div_term)

        # 这样我们就得到对应的位置编码，但是想要与embedding进行运算需要增加维度
        position_coding_matrix = position_coding_matrix.unsqueeze(0)

        # 将位置编码矩阵注册为模型的buffer，因为位置编码不需要随着模型参数一起进化
        self.register_buffer('position_coding_matrix', position_coding_matrix)

    def forward(self, x):
        """
        位置编码的前向传播，我们默认的max_len为5000很大，很难一个句子有5000个词汇
        所以在这里我们只需要直接截取成embedding合适的大小即可
        :param x: 文本的词嵌入表示
        :return: 经过位置编码后的embedding表示 (样本数, max_len, d_model)
        """

        # 不需要梯度更新
        x = x + Variable(self.position_coding_matrix[:, :x.size(1)], requires_grad=False)

        return self.dropout(x)


def SubsequentMask(size):
    """
    该函数用于生成掩码张量
    :param size: 掩码张量最后两个维度的大小
    :return:
    """

    # 定义掩码张量的形状
    shape = (1, size, size)
    # 生成上三角矩阵
    subsequent_mask = np.triu(np.ones(shape=shape), k=1).astype("uint8")

    # 将numpy类型变成torch中tensor类型，并将上三角矩阵变成下三角矩阵
    return torch.from_numpy(1 - subsequent_mask)


def Attention(query, key, value, mask=None, dropout=None):
    """
    注意力机制代码
    :param query: Q
    :param key: K
    :param value: V
    :param mask: 掩码张良
    :param dropout: 置0概率
    :return: 注意力张量，注意力分数
    """

    # 取出query的最后一个维度，一般等同于词嵌入维度
    d_k = query.size(-1)
    # 获取没有乘value以及softmax的权重系数矩阵，这里要对key中的最后两个维度进行转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 判断是否使用掩码张量
    if mask is not None:
        # 将掩码张量和scores张量中的每一个位置进行比较，将0的位置置换成-1e-9，这样经过softmax其影响权重效果就会接近于0
        scores = scores.masked_fill(mask == 0, -1e-9)

    # 对scores的最后一个维度进行softmax，并获取注意力张量
    p_attention = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attention = dropout(p_attention)

    attention = torch.matmul(p_attention, value)

    # 将query以及注意力分数返回
    return attention, p_attention


def CloneModel(module, N):
    """
    用于生成相同网络层的克隆函数，通过克隆来获取多头注意力机制中多个结构相同的线性层
    :param module: 需要克隆的目标网络层
    :param N: 需要克隆的数量
    :return: nn.ModuleList列表存放模型
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """
        多头注意力机制只会分割最后一个维的词嵌入向量。所谓的多头，将每一个头的获得的输入送到注意力机制中就形成了多头注意力机制
        这种结构涉及能够让每一个注意力机制优化每个词汇的不同特征部分，从而均衡同一种注意力机制可能带来的偏差
        :param head: 头的数目
        :param embedding_dim: 词嵌入的维度
        :param dropout: 置零比率，默认为1
        """
        super(MultiHeadedAttention, self).__init__()

        # 我们需要判断词嵌入的维度(embedding_dim)能否被头的数目所整除(head)，这里使用断言解决
        assert embedding_dim % head == 0, "词嵌入维度不能整除头的数目"

        # 获取每个头所拥有的词嵌入维度长度以及头的个数
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim

        # 获得四个大线性层，这里将不同头进行统一线性计算提高并行速率，4个下分别用于Q,K,V以及拼接矩阵
        self.linears = CloneModel(nn.Linear(embedding_dim, embedding_dim), 4)

        # 初始化获得的注意力张量
        self.p_attention = None

        # 初始化dropout对象
        self.p = dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        多头注意力模块前向传播
        :param query: Q
        :param key: K
        :param value: V
        :param mask: 掩码张量
        :return: 多头注意力张量
        """

        if mask is not None:
            # 拓展维度，代表多头中的第n头
            mask = mask.unsqueeze(1)

        # 获得batch_size的大小，代表当前批次有多少样本
        batch_size = query.size(0)

        """
        分别在线性层中对QKV进行运算，运算结束后使用view对维度进行重构，方便多头一起进入attention模块进行并行处理
        进行转置是因为让句子长度维度与词向量维度能够相邻，这样注意力机制才能够找到词义与句子位置的关系
        这里生成的数据在传入Attention函数后会将其最后两个维度进行自动计算
        """
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]

        # 在得到每一个头的输出之后接下两就可以将其传入attention当中
        x, self.p_attention = Attention(query=query, key=key, value=value, mask=mask, dropout=self.dropout)
        # 转换数据维度，相当于拼接头的操作
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后使用线性层将注意力机制的输出进行线性变换最终得到多头注意力结构的输出
        return self.linears[-1](x)


class FeedForwardDenseLayer(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        """
        这里在论文中的原本应该叫 position wise fully connected feed-forward network
        但其实该函数结构就是就是包含一个输入层，一个隐藏层，一个输出层的三层神经网络架构
        :param d_model: 线性层的输入维度
        :param d_ff: 线性层的隐藏维度
        :param dropout: dropout就是dropout
        """
        super(FeedForwardDenseLayer, self).__init__()
        # 初始化双层网络结构
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.p = dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        前馈神经网络前向传播
        :param x: 输入变量，代表上一层的输入
        :return: 前馈神经网络输出变量
        """
        # 线性层前向传播部分
        x = self.w1(x)
        # 经过relu函数激活
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w2(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        网络层中的规范化层，用于规范化前向传播中的参数变量，防止产生梯度爆炸的效果
        :param features: 词嵌入的维度
        :param eps: 用于规范化公式的分母，防止分母为0
        """
        super(LayerNorm, self).__init__()

        # 声明规范化层中的学习参数用于辅助规范化参数
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        # 将规范化分母传入类中
        self.eps = eps

    def forward(self, x):
        """
        规范化层的前向传播
        :param x: 上一层的输入
        :return: 经过规范化后的变量
        """
        # 对输入变量的最后一个维度计算均值以及标准差 (batch_size, max_len, 1)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 使用规范化公式 这里的a2, b2参数直接对最后一维的数据进行变换
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """
        残差结构
        :param size: 词嵌入维度大小
        :param dropout: dropout就是dropout
        """
        super(SubLayerConnection, self).__init__()
        # 实例化规范化参数
        self.norm = LayerNorm(size)
        # 实例化dropout对象
        self.dropout = nn.Dropout(p=dropout)
        self.size = size

    def forward(self, x, sublayer):
        """
        残差结构前向传播
        :param x: 接收上一层或者子层的输入
        :param sublayer: 子层连接中的子层函数, 这里指的是MultiHeadedAttention或者FeedForwardDenseLayer的前向传播
        :return: 残差结构的输出结果
        """

        # 这里的代码可能存在问题
        # 应该先使用sublayer对变量进行运算，再对运算后的结果进行规范化
        # return x + self.dropout(sublayer(self.norm(x)))
        return x + self.dropout(self.norm(sublayer(x)))


# 使用EncoderLayer实现编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout):
        """
        编码器层
        :param size: 词嵌入维度大小
        :param attn:多头自注意力机制实例化对象
        :param feed_forward:前馈全连接层的实例化对象
        :param dropout: dropout就是dropout
        """
        super(EncoderLayer, self).__init__()

        # 初始化多头自注意机制层以及前馈全连接层
        self.attn = attn
        self.feed_forward = feed_forward
        # 用于整体编码器后的规范化层
        self.size = size
        self.dropout = dropout

        # 两个残差结构
        self.subLayer = CloneModel(SubLayerConnection(self.size, dropout), 2)

    def forward(self, x, mask):
        """
        编码层内部的前向传播
        :param x: 上一层的输出
        :param mask: 掩码张量
        :return: 编码层的输出 (batch_size, max_len, 词嵌入维度)
        """

        x = self.subLayer[0](x, lambda x: self.attn(x, x, x, mask))

        return self.subLayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        编码器
        :param layer: 编码层
        :param N: 克隆层数
        """
        super(Encoder, self).__init__()
        # 克隆编码层
        # self.layers = CloneModel(layer, N)
        self.layers = []
        for i in range(N):
            attnTemp = MultiHeadedAttention(head=layer.attn.head, embedding_dim=layer.attn.embedding_dim,
                                            dropout=layer.attn.p)
            ffTemp = FeedForwardDenseLayer(d_model=layer.feed_forward.d_model, d_ff=layer.feed_forward.d_ff,
                                           dropout=layer.feed_forward.p)
            elTemp = EncoderLayer(size=layer.size, attn=attnTemp, feed_forward=ffTemp, dropout=layer.dropout)
            self.layers.append(elTemp)
        # 初始化规范化层，放入编码器之后
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        """
        编码器整体前向传播
        :param x:
        :param mask:
        :return: 编码器输出变量
        """

        for layer in self.layers:
            # 编码器中的mask应为None
            x = layer(x, mask)

        # 这里论文中的代码不同
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, attn, src_attn, feed_forward, dropout):
        """
        解码器层，作为解码器的组成单元，每个解码器根据给定的输入向目标方向进行特征提取操作
        :param size: 词嵌入维度，也代表解码器尺寸
        :param attn: 多头自注意力机制 Q = K = V
        :param src_attn: 多头自注意力对象 Q != K = V
        :param feed_forward: 前馈全连接层
        :param dropout: dropout就是dropout
        """
        super(DecoderLayer, self).__init__()
        # 将参数传入层中
        self.size = size
        self.dropout = dropout
        self.attn = attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 克隆三个子层连接对象
        self.subLayers = CloneModel(SubLayerConnection(size=size, dropout=dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        编码器层前向传播
        :param x: 上一层的输入
        :param memory: 编码器语义场存储变量
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        :return: 编码器输出变量
        """

        # 将x传入第一个子层结构，第一个子层的输入分别是x和attn函数
        # 最后一个参数是目标数据的掩码张量，这时要多目标数据进行遮掩
        x = self.subLayers[0](x, lambda x: self.attn(x, x, x, target_mask))

        # 模型第二层，这一层使用常规的注意力机制。Q是输入x 而K，V是编码器输出memory
        # 同样传入source_mask，但是进行数据遮掩的原因并非是抑制信息泄露，而是遮掉对结果没有意义的字符而产生的注意力值。这里原论文不同
        x = self.subLayers[1](x, lambda x: self.src_attn(x, memory, memory, source_mask))

        # 经过前馈全连接层
        return self.subLayers[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        """
        解码器
        :param layer: 解码层layer
        :param N: 解码器的个数N
        """
        super(Decoder, self).__init__()
        # 走过所有编码器层后需要进行规范化操作，这里和原论文不同
        self.norm = LayerNorm(layer.size)

        self.layers = []
        for i in range(N):
            # 初始化参数
            attn_head = layer.attn.head
            attn_embedding_dim = layer.attn.embedding_dim
            attn_dropout = layer.attn.p
            src_attn_head = layer.src_attn.head
            src_attn_embedding_dim = layer.src_attn.embedding_dim
            src_attn_dropout = layer.src_attn.p
            feed_forward_d_model = layer.feed_forward.d_model
            feed_forward_d_ff = layer.feed_forward.d_ff
            feed_forward_dropout = layer.feed_forward.p

            # 构建模型
            attn = MultiHeadedAttention(head=attn_head, embedding_dim=attn_embedding_dim, dropout=attn_dropout)
            src_attn = MultiHeadedAttention(head=src_attn_head, embedding_dim=src_attn_embedding_dim,
                                            dropout=src_attn_dropout)
            feed_forward = FeedForwardDenseLayer(d_model=feed_forward_d_model, d_ff=feed_forward_d_ff,
                                                 dropout=feed_forward_dropout)
            decoder = DecoderLayer(size=layer.size, attn=attn, src_attn=src_attn, feed_forward=feed_forward,
                                   dropout=layer.dropout)
            self.layers.append(decoder)

    def forward(self, x, memory, source_mask, target_mask):
        """
        编码器层的前向传播
        :param x: 目标数据的嵌入表示
        :param memory: 编码器的输出
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        :return:
        """
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)

        # 规范化数据
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        输出层实现
        :param d_model: 词嵌入维度
        :param vocab_size: 词表的大小
        """
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """
        编码器解码器结果
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param source_embed: 源数据嵌入函数
        :param target_embed: 目标数据嵌入函数
        :param generator: 输出部分的类别生成器
        """
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """
        编码器解码器前向传播
        :param source: 源数据
        :param target: 目标数据
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        :return: 解码器输出结果
        """

        # return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)
        # 这里做修改的原因是编码阶段需要自注意力注意全局的变量，如果传入掩码则不能够有效的观察全局信息
        return self.decode(self.encode(source, None), source_mask, target, target_mask)

    def encode(self, source, source_mask=None):
        """
        编码阶段
        :param source: 源数据
        :param source_mask: 源数据掩码张量
        :return:
        """

        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        """
        解码阶段
        :param memory: 编码器的输出
        :param source_mask: 源数据掩码张量
        :param target: 目标数据
        :param target_mask: 目标数据掩码张量
        :return: 解码器输出结果
        """

        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


def Transformer(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    """
    该函数用于构建transformer模型
    :param source_vocab: 源数据词汇特征总数
    :param target_vocab: 目标数据词汇特征总数
    :param N: 编码器解码器堆叠数目
    :param d_model: 词向量映射维度
    :param d_ff: 前馈全连接层中变换矩阵维度
    :param head: 多头注意力机制中头的数目
    :param dropout: dropout就是dropout
    :return:
    """

    """一般在模型构建是需要对模型结构进行深拷贝，但是在当前模型实例中Encoder与Decoder实现了在类中循环构建新模型所以不需要进行深拷贝"""

    # 多头自注意力机制
    attn = MultiHeadedAttention(head=head, embedding_dim=d_model)
    # 前馈全连接层
    ff = FeedForwardDenseLayer(d_model=d_model, d_ff=d_ff, dropout=dropout)

    # 实例化位置编码类
    position = PositionalEncoding(d_model=d_model, dropout=dropout)

    # 构建transformer模型
    model = EncoderDecoder(
        encoder=Encoder(layer=EncoderLayer(size=d_model, attn=attn, feed_forward=ff, dropout=dropout), N=N),
        decoder=Decoder(layer=DecoderLayer(size=d_model, attn=attn, src_attn=attn, feed_forward=ff, dropout=dropout), N=N),
        # 代码中的注释提到这里输入的应该只是embedding功能，但是我们在embedding后加入了位置编码。位置编码都用一样的即可，它不需要更新。
        source_embed=nn.Sequential(Embeddings(d_model=d_model,vocab=source_vocab), position),
        target_embed=nn.Sequential(Embeddings(d_model=d_model, vocab=target_vocab), position),
        generator=Generator(d_model=d_model, vocab_size=target_vocab)
    )

    # 初始化模型中的参数
    for param in model.parameters():
        if param.dim() > 1:
            # 服从均匀分布
            nn.init.xavier_uniform_(param)
    return model



