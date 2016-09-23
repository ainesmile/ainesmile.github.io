---
layout: post
title:  "两个定理"
categories: Programming theorems
---

本书第五章要推导两个定理，一个是关于\\( if-fi \\)结构的，一个是关于\\( do-od \\)结构的。令

$$ BB = (Ej:1 \leq j \leq n : B_{j}) $$

### 有关选择结构的基本定理

令，有选择结构IF和一对谓词\\(Q\\)和\\( R \\)，使得

$$ Q \Rightarrow BB $$

对所有的状态都有：

$$ (Aj: 1 \leq j \leq n: (Q \ and \ B_{j}) \Rightarrow wp(SL_{j}, R)) $$

那么，对所有的状态均有

$$ Q \Rightarrow wp(IF, R) $$




### 有关重复结构的基本定理

令选择结构 \\( IF \\) 和一个谓词 \\( P \\) 在所有的状态下都有

$$ (P \ and \ BB) \Rightarrow wp(IF, R) $$

那么，对于相应的重复结构 \\( DO \\)，我们有

$$ (P \ and \ wp(DO, T)) \Rightarrow wp(DO, P \ and \ non \ BB) $$

这个定理也被称为“循环的基本不变式定理”（Fundamental Invariance Theorem for Loops），其直观意义可以理解为：

如果初始状态使得 \\( P \\) 成立且存在 guard \\( B_{j} \\)为真，
那么在执行相应的 statement lists 之后，这些初始状态将使系统终止于一个满足 \\( P \\) 的最终状态。如果上述条件成立，那么将有，
使得 \\( P \\) 成立且使得循环完美终止的初始状态，在整个循环结构完成时，将使系统终止于一个满足 \\( P \\) 且不存在 guard \\( B_{j} \\)为真的最终状态。






### 证明过程



#### 选择结构基本定理证明

要证

$$ Q \Rightarrow wp(IF, R) $$


已知：

$$wp(IF, R) = BB \ and \ (Aj: 1 \leq j \leq n: B_{j} \Rightarrow wp(SL_{j}, R))$$

记

$$A_{1} = (Aj: 1 \leq j \leq n: B_{j} \Rightarrow wp(SL_{j}, R))$$

以及

$$A_{2} = (Aj: 1 \leq j \leq n: (Q \ and \ B_{j}) \Rightarrow wp(SL_{j}, R))$$


因为 \\( Q \Rightarrow BB \\)，所以要证明定理成立，只需证明 \\( Q \Rightarrow A_{1} \\)即可。

将状态空间分为两个互不相交的集合，分别记为 \\(Q_{f}\\) 和 \\(Q_{t}\\)，分别表示使 \\(Q\\) 为 false 的初始状态的集合和使 \\( Q \\) 为 true 的初始状态集合，其中 \\(Q_{t} = Q\\)。


\\(Q_{f}\\) 中的初始状态，因为 \\(Q\\) 为 false，所以 \\( Q \Rightarrow A_{1} \\)恒成立。

\\( \forall p \in Q\\)，使得 \\( B_{j} \\) 为 true 或者 false。对于在 \\( p \\) 上为 false 的 \\( B_{j} \\)，有

$$  B_{j} \Rightarrow wp(SL_{j}, R) $$

即 \\( p \\) 满足条件  \\( A_{1} \\)。

对于在 \\( p \\) 上为 true 的 \\( B_{j} \\)，有 \\( p \in B_{j} \\)，结合条件 \\( A_{2} \\)，有 \\( p \in wp(SL_{j}, R) \\)。因此，\\( \forall p  \\) 使得  \\( B_{j} \\) 为真，都有 \\( wp(SL_{j}, R) \\) 为真，即

$$ B_{j} \Rightarrow wp(SL_{j}, R) $$

即 \\( p \\) 满足条件 \\( A_{1} \\)。

综上，\\( \forall p \in Q\\)，\\( p \\) 满足条件 \\( A_{1} \\)，即

$$ Q \Rightarrow A_{1} $$

因此，对所有的状态均有

$$ Q \Rightarrow wp(IF, R) $$




#### 重复结构基本定理证明

要证明

\begin{align}
(P \ and \ wp(DO, T)) \Rightarrow wp(DO, P \ and \ non \ BB)
\label{origin}
\end{align}

已知：

\begin{split}
P \ and \ wp(DO, T) & = P \ and \ (Ek: 1 \leq k \leq n : H_{k}(T)) \\\\ & = 
(Ek: 1 \leq k \leq n : P \ and \ H_{k}(T))
\end{split}

以及：

$$ wp(DO, P \ and \ non \ BB) = (Ek: 1 \leq k \leq n : H_{k}(P \ and \ non \ BB))$$

因此，式子 \eqref{origin} 等价于

$$ (Ek: 1 \leq k \leq n : P \ and \ H_{k}(T)) \Rightarrow  (Ek: 1 \leq k \leq n : H_{k}(P \ and \ non \ BB)) $$

因此，要证明式子 \eqref{origin}，只需证明以下式子即可：

\begin{align}
P \ and \ H_{k}(T) \Rightarrow H_{k}(P \ and \ non \ BB)
\label{key}
\end{align}

当 \\( k = 0 \\) 时，因为

\begin{split}
P \ and \ H_{0}(T) = P \ and \ non \ BB = H_{0}(P \ and \ non \ BB)
\end{split}

得到

\begin{align}
 P \ and \ H_{0}(T) \Rightarrow H_{0}(P \ and \ non \ BB)
\label{zero}
\end{align}

当 \\( k > 0 \\) 时，因为

$$wp(IF, H_{k-1}(T)) = BB \  and \ (Aj: 1 \leq j \leq n: B_{j} \Rightarrow wp(SL_{j}, H_{k-1}(T)))$$

得到

$$wp(IF, H_{k-1}(T)) \Rightarrow BB$$

又因为

$$(P \ and \ BB) \Rightarrow wp(IF, P)$$

因此，\eqref{key} 式左边有


\begin{equation}
\begin{split}
 P \ and \ H_{k}(T) & = P \ and \ [wp(IF, H_{k-1}(T)) \ or \ non \ BB]  \\\\ & = 
 [P \ and \ wp(IF, H_{k-1}(T))] \ or \ [P \ and \ non \ BB] \\\\ & \Rightarrow 
 [P \ and \ BB \ and \  wp(IF, H_{k-1}(T))]  \ or \ [P \ and \ non \ BB]
 \\\\ & \Rightarrow [wp(IF, P)  \ and \  wp(IF, H_{k-1}(T))]  \ or \ [P \ and \ non \ BB] 
 \\\\ & = [wp(IF, P \ and \  H_{k-1}(T))]  \ or \ [P \ and \ non \ BB]
\end{split}
\label{key-leftside}
\end{equation}


\eqref{key} 式右边有

\begin{align}
H_{k}(P \ and \ non \ BB) = wp(IF, H_{k-1}(P \ and \ non \ BB)) \ or (P \ and \ non \ BB)
\label{key-rightside}
\end{align}

综合式子 \eqref{key-leftside} 以及式子 \eqref{key-rightside}，要证明式子 \eqref{key}，只需证明：

\begin{align}
P \ and \  H_{k-1}(T) \Rightarrow  H_{k-1}(P \ and \ non \ BB)
\label{k-1}
\end{align}


用数学归纳法证明。由上述论证可知：当 \\( k = 0 \\) 时，\eqref{zero} 成立；\\( \forall k > 0 \\)，当 \eqref{k-1} 成立时，有 \eqref{key} 成立。

综上，当 \\( k \geq 0 \\) 时， \eqref{key} 成立。由此，\eqref{origin} 得证。

### 笔记来源
[A Discipline of Programming](https://book.douban.com/subject/1762127/)