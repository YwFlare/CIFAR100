## SENet

核心思想：关注channel之间的关系，希望模块可以自动学习到不同channel特征的重要程度

<img src="images-master/202206062022741.png?token=APJNLFTA5JLJ7SL677ANTADCTXY3I" alt="image-20220606202213667" style="zoom:30%;" />

全局平均池化->全连接C/r->ReLU->全连接C->Sigmoid

<img src="images-master/202206062022023.png?token=APJNLFXUDCQF7ADGRE3BASLCTXY52" alt="image-20220606202255937" style="zoom:25%;" />

缺点：SE全局平均池化将空间信息嵌入通道信息导致信息丢失

## SKNet

选择性核（SK）关注不同分支的情况，

<img src="images-master/202206062025701.png?token=APJNLFWNXJUVHTTQQNABY23CTXZJG" alt="image-20220606202558645" style="zoom:50%;" />

分割：不同大小的核卷积获取多路分支，以捕捉更多的信息

融合：SE思想（全局平均池化->全连接C/r->ReLU->全连接C->Sigmoid）

选择：通道权重值乘起来后相加

## CooAtten

将通道和空间并行考虑的混合域注意力

给定一个input，将全局平均池化分成两步操作，用两个池化核（H,1）（1,W）沿着特征图的两个不同方向进行池化，得到两个嵌入后的信息特征图

两个特征图沿着空间维度拼接，卷积后激活再分离成两个特征图

<img src="images-master/202206062101831.png?token=APJNLFQQ77PZKG3I5RQAYYTCTX5PA" alt="image-20220606210138786" style="zoom:50%;" />

## CBAM

<img src="images-master/202206062029953.png?token=APJNLFW5KZJWGNNLGXBO4GTCTXZWI" alt="image-20220606202927899" style="zoom:40%;" />

1、混合注意力

- 通道域：平均池化和最大池化->两层全连接->相加、Sigmoid->通道注意力向量

- 空间域：全局平均和全局池化->卷积->激活函数->空间注意力向量

2、串行：通道域->空间域

3、使用：插入到卷积块里面

4、缺点：长程依赖问题没有解决（空间域卷积操作）

## Triplet Attention

<img src="images-master/202206062158007.png?token=APJNLFV37FZOI3SL4QQGO4LCTYEDI" alt="image-20220606215815966" style="zoom:40%;" />

## Non-local

<img src="images-master/202206062119054.png?token=APJNLFSZ3QMWVIZG6KVNXTDCTX7RW" alt="image-20220606211927016" style="zoom:50%;" />

三个1*1\*1卷积，后转换维度矩阵乘法，再和原来的相加
$$
\begin{array}{c}
f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=e^{\theta\left(\mathbf{x}_{i}\right)^{T} \phi\left(\mathbf{x}_{j}\right)} \quad g\left(\mathbf{x}_{j}\right)=W_{g} \mathbf{x}_{j} \\
\mathbf{y}_{i}=\frac{1}{\mathcal{C}(\mathbf{x})} \sum_{\forall j} f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right) g\left(\mathbf{x}_{j}\right) \\
\mathcal{C}(\mathbf{x})=\sum_{\forall j} f\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right) \\
\mathbf{z}_{i}=W_{z} \mathbf{y}_{i}+\mathbf{x}_{i}
\end{array}
$$
卷积操作在每一层获得的感受野有限，对于一些长距离的依赖，需要堆叠很多层卷积才能获得

Non-Local Block，这种结构使得网络可以直接获得两个位置之间的依赖关系，而不用考虑距离

(1) 只涉及到了位置注意力模块，而**没有涉及常用的通道注意力机制**

## A2-Nets

<img src="images-master/202206062231945.png?token=APJNLFT5KBJGBCGSOZ2ATV3CTYIBY" alt="image-20220606223158804" style="zoom:25%;" />

## GCNet

![image-20220606213531416](images-master/202206062135472.png?token=APJNLFTNBANEKKEEKELNCSTCTYBOA)

## mobilenext

<img src="images-master/202206062200251.png?token=APJNLFSDL755J7C3UWML4LTCTYEKS" alt="image-20220606220013216" style="zoom:50%;" />

## MLP-Mixer

不需要卷积核注意力机制

<img src="images-master/202206062148012.png?token=APJNLFTVDZC4TOUKAGLH2XDCTYC5A" alt="image-20220606214803956" style="zoom:50%;" />
