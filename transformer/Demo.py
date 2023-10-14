import copy

import numpy as np
# 机器学习包
import torch
import torch.nn as nn
import torch.nn.functional as F
# 数学包
import math
from torch.autograd import Variable
# 类包
from transformer.Model import *
from torchsummary import summary


def testSingleFunction():
    # 词嵌入维度是512
    d_model = 512
    # 词表大小是1000
    vocab = 1000
    dropout = 0.1
    max_len = 60

    # 测试embedding层
    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    embedding = Embeddings(d_model=d_model, vocab=vocab)
    answerAfterEmbedding = embedding(x)
    # 测试位置编码
    pe = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
    answerAfterPe = pe(answerAfterEmbedding)
    print(f"answerAfterPe: {answerAfterPe}")
    # 测试掩码张量
    mask = SubsequentMask(4)
    # 测试注意力机制
    query = key = value = answerAfterPe
    attention, p_attention = Attention(query=query, key=key, value=value, mask=mask)
    print(f"p_attention: {p_attention}")
    print(f"attention: {attention}")
    # 测试多头注意力机制
    head = 8
    embedding_dim = 512
    dropout = 0.2
    # 直接使用位置编码后的结果
    query = key = value = answerAfterPe
    mask = SubsequentMask(4)
    mha = MultiHeadedAttention(head=head, embedding_dim=embedding_dim, dropout=dropout)
    mhaResult = mha(query, key, value, mask)
    print(f"mhaResult: {mhaResult}")
    print(mhaResult.shape)
    # 测试前馈神经网络
    d_model = 512
    d_ff = 64
    dropout = 0.2
    ff = FeedForwardDenseLayer(d_model=d_model, d_ff=d_ff, dropout=dropout)
    ffResult = ff(mhaResult)
    print(f"ffResult: {ffResult}")
    print(f"ffResult shape: {ffResult.shape}")
    # 测试规范化层
    features = d_model = 512
    eps = 1e-6
    ln = LayerNorm(features=features, eps=eps)
    lnResult = ln(ffResult)
    print(f"lnResult: {lnResult}")
    print(f"lnResult shape: {lnResult.shape}")
    # 测试残差连接层
    size = 512
    dropout = 0.2
    head = 8
    x = answerAfterPe
    d_model = 512
    self_attn = MultiHeadedAttention(head=head, embedding_dim=d_model)
    subLayer = lambda x: self_attn(x, x, x, mask)
    sc = SubLayerConnection(size, dropout)
    scResult = sc(x, subLayer)
    print(f"scResult: {scResult}")
    print(f"scResult shape: {scResult.shape}")
    # 测试编码层
    size = 512
    head = 8
    d_model = 512
    d_ff = 2048
    x = answerAfterPe
    dropout = 0.2
    attn = MultiHeadedAttention(head=head, embedding_dim=d_model, dropout=dropout)
    ff = FeedForwardDenseLayer(d_model=d_model, d_ff=d_ff, dropout=dropout)
    mask = SubsequentMask(4)
    el = EncoderLayer(size=size, attn=attn, feed_forward=ff, dropout=dropout)
    elResult = el(x, mask=mask)
    print(f"elResult: {elResult}")
    print(f"elResult shape: {elResult.shape}")
    # 测试编码器
    encoder = Encoder(layer=el, N=8)
    encoderResult = encoder(answerAfterPe, None)
    print(f"encoder: {encoderResult}")
    print(f"encoder shape: {encoderResult.shape}")
    # 测试解码层
    head = 8
    size = 512
    d_model = 512
    d_ff = 64
    dropout = 0.2
    attn = src_attn = MultiHeadedAttention(head=head, embedding_dim=d_model, dropout=dropout)
    ff = FeedForwardDenseLayer(d_model=d_model, d_ff=d_ff, dropout=dropout)
    x = answerAfterPe
    memory = encoderResult
    target_mask = SubsequentMask(4)
    source_mask = None
    dl = DecoderLayer(size=size, attn=attn, src_attn=src_attn, feed_forward=ff, dropout=dropout)
    dlResult = dl(x=x, memory=memory, source_mask=source_mask, target_mask=target_mask)
    print(f"dlResult: {dlResult}")
    print(f"dlResult shape: {dlResult.shape}")
    # 测试解码器
    decoder = Decoder(layer=dl, N=8)
    decoderResult = decoder(x, memory, source_mask, target_mask)
    print(f"decoderResult: {decoderResult}")
    print(f"decoderResult shape: {decoderResult.shape}")
    # 测试输出层
    d_model = 512
    vocab_size = 1000
    x = decoderResult
    gen = Generator(d_model=d_model, vocab_size=vocab_size)
    genResult = gen(x)
    print(f"genResult: {genResult}")
    print(f"genResult shape: {genResult.shape}")
    # 测试编码器解码器
    vocab_size = 1000
    d_model = 512
    source_embed = nn.Embedding(vocab_size, d_model)
    target_embed = nn.Embedding(vocab_size, d_model)
    generator = gen
    source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    source_mask = target_mask = Variable(torch.zeros(1, 4, 4))
    ed = EncoderDecoder(encoder=encoder, decoder=decoder, source_embed=source_embed, target_embed=target_embed,
                        generator=generator)
    edResult = ed(source, target, source_mask, target_mask)
    print(f"edResult: {edResult}")
    print(f"edResult shape: {edResult.shape}")


def testModel():
    model = Transformer(source_vocab=11, target_vocab=11)
    print(model)


# testSingleFunction()
testModel()
