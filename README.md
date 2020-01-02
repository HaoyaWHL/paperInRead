author liwenhao
date from 2019.12

# 工具学习

tensorflow相关网站：

 - http://c.biancheng.net/view/1902.html


# 0、学习目录

视频：
- 李飞飞 cs231
- 吴恩达 机器学习
- 李宏毅 机器学习，**深度学习**

书本：
- **深度学习**
- 统计学习方法
- 西瓜书

这个假期必须学完一个视频的内容

# 1、李宏毅，深度学习

## 1.0、总览
- 视频学习链接 https://www.bilibili.com/video/av9770302
- PPT总览（简明版） http://speech.ee.ntu.edu.tw/~tlkagk/talk.html
- 视频对应的课程和详细PPT地址
```
http://speech.ee.ntu.edu.tw/~tlkagk/courses.html
Machine Learning and having it deep and structured (2017,Spring)
```


## 1.1、扩展阅读

alphaGo
> 近些年出现的跟AI强相关的产品，事件，作为算法工程师有必要进一步了解，李老师PPT关于CNN有涉及

nlp和cv的几大任务
> 要对这个领域有了解，比如nlp可以划分为文本翻译，分类，匹配，阅读理解等几个方面

dropout
>* https://blog.csdn.net/u012052268/article/details/77649215
>* [1]. Srivastava N, Hinton G, Krizhevsky A, et al. Dropout: A simple way to prevent neural networks from overfitting[J]. The Journal of Machine Learning Research, 2014, 15(1): 1929-1958.
>* [2]. Dropout as data augmentation. http://arxiv.org/abs/1506.08700

## 1.2、工具使用

Keras使用
```
François Chollet is the author of Keras.
He currently works for Google as a deep learning engineer and researcher.
Keras means horn in Greek
Documentation: http://keras.io/
Example: https://github.com/fchollet/keras/tree/master/examples
```

## 1.3、阅读PPT的笔记

### Lecture I: Introduction of Deep Learning

为什么要deep layer而不是fat layer的深层原因
> https://www.youtube.com/watch?v=XsC9byQkUH8&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=13

highway network
> Training Very Deep Networks
https://arxiv.org/pdf/1507.06228v2.pdf

residual network
> Deep Residual Learning for Image Recognition
http://arxiv.org/abs/1512.03385

Combination of Different Basic Layers
> Tara N. Sainath, Ron J. Weiss, Andrew Senior, Kevin W. Wilson, Oriol Vinyals, “Learning the Speech Front-endWith Raw Waveform CLDNNs,” In INTERPSEECH 2015

tensorflow for word vecotr
> https://www.youtube.com/watch?v=X7PH3NuYW0Q


Recipe of Deep learning
>* Choosing proper loss
    （http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf）
    交叉熵比均方误差在loss的图上更陡峭
>* Mini-batch
    速度更快，比all-batch
>* New activation function
    以前用RBM做pre-train
    现在用Relu，但是深层原因需要看一看资料（一个很大的与原因是为了解决梯度消失的问题）
>* Adaptive Learning Rate
>* Momentum（最后一个方法是啥，局部最优与全局最优的问题）
>* Early Stop
>* Regularization
>* Dropout
这个地方可以深入研究一下，dropout的理由，有对应的hiton和另一篇论文来着，还要看一下视频
>* Network Structure


常见的最优化算法需要了解
>* Adagrad [John Duchi, JMLR’11]
>* RMSprop
https://www.youtube.com/watch?v=O3sxAc4hxZU
>* Adadelta [Matthew D. Zeiler, arXiv’12]
>* “No more pesky learning rates” [Tom Schaul, arXiv’12]
>* AdaSecant [Caglar Gulcehre, arXiv’14]
>* Adam [Diederik P. Kingma, ICLR’15]
>* Nadam
http://cs229.stanford.edu/proj2015/054_report.pdf


dropout
>* More reference for dropout
[Nitish Srivastava, JMLR’14] [Pierre Baldi, NIPS’13][Geoffrey E. Hinton, arXiv’12]
>* Dropout works better with Maxout [Ian J. Goodfellow, ICML’13]
>* Dropconnect [Li Wan, ICML’13]
Dropout delete neurons
Dropconnect deletes the connection between neurons
>* Annealed dropout [S.J. Rennie, SLT’14]
Dropout rate decreases by epochs
>* Standout [J. Ba, NISP’13]
Each neural has different dropout rate

### Lecture II: Variants of Neural Networks(CNN&RNN)

cnn
>* cnn的核心是convolution和pooling，所以关于zero padding的部分有必要深入了解一下
>* 最后的flatten分类操作，得深入看一下

attention-based model
> http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/Attain%20(v3).ecm.mp4/index.html

### Lecture III: Beyond Supervised Learning(unsupervised learning&&Reinforcement learning)

document to vecotr
>* Paragraph Vector: Le, Quoc, and Tomas Mikolov. "Distributed Representations of Sentences and Documents.“ ICML, 2014
>* Seq2seq Auto-encoder: Li, Jiwei, Minh-Thang Luong, and Dan Jurafsky. "A hierarchical neural autoencoder for paragraphs and documents." arXiv preprint, 2015
>* Skip Thought: Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, Sanja Fidler, “Skip-Thought Vectors” arXiv preprint, 2015.
>* Exploiting other kind of labels:
(1) Huang, Po-Sen, et al. "Learning deep structured semantic models for web search using clickthrough data."ACM, 2013.
(2)Shen, Yelong, et al. "A latent semantic model with convolutional-pooling structure for information retrieval." ACM, 2014.
(3) Socher, Richard, et al. "Recursive deep models for semantic compositionality over a sentiment treebank." EMNLP, 2013.
(4) Tai, Kai Sheng, Richard Socher, and Christopher D. Manning. "Improved semantic representations from tree-structured long short-term memory networks." arXiv preprint, 2015.


generative models
>* https://openai.com/blog/generative-models/
>* https://www.quora.com/What-did-Richard-Feynman-mean-when-he-said-What-I-cannot-create-I-do-not-understand


PixelRNN
> Ref: Aaron van den Oord, Nal Kalchbrenner, Koray Kavukcuoglu, Pixel Recurrent Neural Networks, arXiv preprint, 2016


GAN
> Ref: Generative Adversarial Networks, http://arxiv.org/abs/1406.2661


autoencoder(对finetune和pre-train要有深入了解,还有variation auto-encoder,简称VAE)
>* Reference: Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. "Reducing the dimensionality of data with neural networks." Science 313.5786 (2006): 504-507
>* Vincent, Pascal, et al. "Extracting and composing robust features with denoising autoencoders." ICML, 2008.
>* Ref: Rifai, Salah, et al. "Contractive auto-encoders: Explicit invariance during feature extraction.“ Proceedings of the 28th International Conference on Machine Learning (ICML-11). 2011.
>* Ref: http://www.wired.co.uk/article/google-artificial-intelligence-poetry
>* Samuel R. Bowman,Luke Vilnis,Oriol Vinyals,Andrew M. Dai,Rafal Jozefowicz,Samy Bengio, Generating Sentences from a Continuous Space, arXiv prepring, 2015

word2vec

Reinforcement learning



### 1.3.1 疑难点
- dropout
    PPT里的意思好像是dropout后，在test阶段weight要乘上(1-p%)，其中p%是dropout概率？
- momentum
- rnn
    从slot filling引入，可以看看
- lstm
    讲解很丰富，值得反复看，毕竟也是现阶段最常用的经典模型了
    然后延申到了many to one和many to many
- gradient explode和gradient vanishing
    这个必须得会

## 1.4、视频学习（按部就班的来就好，别跳了）


### ~~1.4.1、Computational Graph for Backpropagation~~

地址（总览PPT和视频里的每小节略有不同，所以又给出一个地址）
```
http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS17.html
Computational Graph for Backpropagation
```


### ~~1.4.2 Deep learning for language modeling~~

language model: estimated the probability of word sequence

都是给出一些词，去预测下一个词出现的概率（这些word词来源于seq句子），多个词的概率整合起来就是一个seq出现的概率


#### 疑难点
Matrix Factorization 这个在讲PPT的时候是怎么引入的，解决什么问题，引入了一个loss很突兀


#### ~~传统做法~~
N-gram（也叫N-gram language model）

注意：这里讲解的传统的N-gram，是以词为单位的，没有到character的级别，所以李老师在讲的时候，会说这其实是一种非常暴力的方法，N-gram要学习的参数个数远比NN-based language model要大（因为是一个很大的vocabulary里面简单的计算多个类似HMM的概率，用统计的方法，NN-based langualge model则是用到了向量空间，这么一分析N-gram确实很暴力）

**2-gram的做法**
N-gram language model: P(w1, w2, w3, …., wn) = P(w1|START)P(w2|w1) …... P(wn|wn-1)
P(beach|nice) = C(nice beach) / C(nice)
C(nice beach)是说training data里面```nice beach```出现的概率

**Challenge of N-gram** : the estimated probability is not accurate
- N-gram中N很大的时候
- training data具有稀疏性


#### ~~NN-based LM~~
P(“wreck a nice beach”)
=P(wreck|START)P(a|wreck)P(nice|a)P(beach|nice)

P(b|a): the probability of NN predicting the next word.
这个机率的计算，对trainning data中不是用统计的方法来，而是把每个word变为token来处理


### ~~1.4.3、spatial transformer~~

教授的上课PPT首页里有个参考论文可以读一下
> Ref: Max Jaderberg, Karen Simonyan, Andrew Zisserman,Koray Kavukcuoglu, “Spatial Transformer Networks”, NIPS, 2015

术语也叫**localisation net**

#### 疑难点（有必要看论文来解决）
运用interpolation的时候，使得
index of layer l-1 和 index of layer l 之间的gradient不为0，能任意加在CNN网络任意一个convolition layer的后面，才能做back propagation训练网络

具体到back propagation，这一步求导是怎么算的，也就是说这里的weight是随机初始化然后求解的吗？那么预测的时候要不要摘掉

### 1.4.4、 Highway Network && Grid LSTM
