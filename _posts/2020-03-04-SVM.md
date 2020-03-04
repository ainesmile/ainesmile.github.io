---
layout: post
title:  "A Super Boring Guide To Support Vector Machines"
date:   2020-03-04 +1500
keywords: SVM Machine-Learning
categories: Machine-Learning
---


## 1. Overview

A support vector machine(SVM) is a supervised Machine Learning algorithm used for <span style="color:#1886a8">classification</span>, <span style="color:#1886a8">regression</span> and <span style="color:#1886a8">outliers detection</span>. The basic idea of SVM is constructing a <span style="color:#1886a8">generalized optimal hyperplane</span> that seperates the data with the maximal margin while makes the smallest number of errors.

For a particular hyperplane, the nearest instances are <span style="color:#1886a8">support vectors</span>. The distance between support vectors and the hyperplane is <span style="color:#1886a8">margin</span>. Basically, SVM Classification is a maximum-margin classifier.

<br>

## 2. Notations

We denote $$ \mathbb{S} $$ as a finite training set, with $$ X $$ as training features, and $$ y $$ as training labels, where

$$\begin{equation}
    X = \begin{bmatrix}
    x_{11} & x_{12} & \cdots & x_{1m}  \\
    \vdots & \vdots & \vdots  & \vdots \\
    x_{n1} &x_{n2} & \cdots & x_{nm} \end{bmatrix}
\end{equation}$$

and

$$\begin{equation}
    y = \begin{bmatrix}
            y_{1} \\
            y_{2} \\
            \cdots \\
            y_{n}
        \end{bmatrix}
\end{equation}$$

$$ n $$ is the sample number, $$ m $$ is the feature number. The label of instance is

$$\begin{equation}
    y_{i} \in \{ -1, 1 \}, \quad i = 1, 2, ..., n
\end{equation}$$

and the features of an instance is

$$\begin{equation}
    x_{i} = \begin{bmatrix}
                x_{i1} & x_{i2} & \cdots & x_{im}
            \end{bmatrix},
    \quad i = 1, 2, ..., n
\end{equation}$$

A hyperplane is denoted as

$$\begin{equation} \phi(x)w+b = 0 \end{equation}$$

where

$$\begin{equation}
    w = \begin{bmatrix} w_{1} \\
            w_{2} \\
            \vdots \\
            w_{m}
        \end{bmatrix}
\end{equation}$$

$$ \phi(x) $$ is a feature mapping function, and $$ b $$ is a constant.

<br>

## 3. SVM: hard margin

To make things easy, we'll consider a binary hard margin classification first. Here, we try to find a hyperplane that maximize the minimum distance between the samples and the hyperplane. The distance bewteen $$ \forall x_{i} $$ and the hyperplane is

$$\begin{equation} d_i = \frac{y_{i}(\phi(x_{i})w+b)}{||w||} \end{equation}$$

So the margin is

$$\begin{equation}
    \min \limits_{i} \ d_{i}, \quad i=1, 2, ... n
\end{equation}$$

Assume instance $$(x^*, y^*)$$ is the nearest one to the hyperplane, therefore the hard margin optimal problem $$\mathcal{H}$$ becomes

$$\begin{equation}
\begin{aligned}
    \max \limits_{w, b} \quad & \frac{y^*(\phi(x^*)w+b)}{||w||} \\
    \text{s.t.} \quad & \frac{y^*(\phi(x^*)w+b)}{||w||} \leq d_i, \quad i=1, 2, ... n
\end{aligned}
\end{equation}$$

Since for all constant $$ a \neq 0 $$, we have

$$\begin{equation}
    y^* \left( \phi(x^*)(aw)+ab \right) = a*y^* \left( \phi(x^*)w+b \right)
\end{equation}$$

so, there must exists a constant $$ c \neq 0 $$, such that

$$ c* y^*(\phi(x^*)w+b) = 1$$

therefore, $$\mathcal{H}$$ becomes

$$\begin{equation}
\begin{aligned}
    \max \limits_{w, b} \quad & \frac{1}{c||w||} \\
    \text{s.t.} \quad & 1 - y_{i}(\phi(x_{i})w+b) \leq 0, \quad i=1, 2, ... n
\end{aligned}
\end{equation}$$

Equals to

$$\begin{equation}
\begin{aligned}
    \min\limits_{w} \quad & \frac{1}{2}w^Tw \\

    \text{s.t.} \quad & 1 - y_{i}(\phi(x_{i})w+b) \leq 0, \quad i=1, 2, ... n
\end{aligned}
\end{equation}$$


Combine with __Lagrangian dual problem__[(see section Lagrangian dual problem)](#sectionDual), $$\mathcal{H}$$ equals to $$\eqref{dual}$$



$$\begin{equation}
\begin{aligned}
    \max \limits_{\lambda, \nu} \min \limits_{w}  L(w, b, \lambda, \nu) =
    \frac{1}{2}w^Tw + \sum_{i=1}^{m}\lambda_{i}(1 - y_{i}(\phi(x_{i})w+b))
\end{aligned}
\end{equation}$$

$$L(w, b, \lambda, \nu) $$ is convex about $$w$$, and $$b$$, therefore $$w$$, and $$b$$ can be drived from $$\lambda$$

$$\begin{equation}
\begin{aligned}
    & \begin{cases}
        \partial L(w, \lambda) / \partial w^* = 0 \\
        \partial L(w, \lambda) / \partial b^* = 0
    \end{cases} \\
    \Rightarrow 
    & \begin{cases}
        w_k^* = \sum_{i=1}^n \lambda_i y_i \phi(x_{ik}), \quad k=1,...m  \\
        \sum_{i=1}^n \lambda_i y_i = 0
    \end{cases} \\
\end{aligned}
\end{equation}$$

$$L(w, b, \lambda, \nu) $$ becomes

$$\begin{equation}
\begin{aligned}
    L(\lambda)
        & = \sum_{i=1}^n \lambda_i -
            \frac{1}{2}\sum_{k=1}^m \left( \sum_{i=1}^n \lambda_i y_i \phi(x_{ik}) \right) ^2 \\
        & = \sum_{i=1}^n \lambda_i -
            \frac{1}{2} \sum_{i,j=1}^n \lambda_i y_i \lambda_j y_j \phi(x_i) \phi(x_j)^T \\
        & = \sum_{i=1}^n \lambda_i -
            \frac{1}{2} \sum_{i,j=1}^n \lambda_i y_i \lambda_j y_j K(x_i, x_j)
\end{aligned}
\end{equation}$$

where $$K(x_i, x_j)$$ is the __kernel function__. Therefore, the problem equals to

$$\begin{equation}
\begin{aligned}
    & \max \limits_\lambda L(\lambda) =
        \sum_{i=1}^n \lambda_i - \frac{1}{2} \sum_{i,j=1}^n \lambda_i \lambda_j y_i  y_j K(x_i, x_j)\\

    & \text{s.t.} \quad \sum_{i=1}^n \lambda_i y_i = 0, 
                \quad \lambda_i \geq 0, \quad i=1, 2, ... n
\end{aligned}
\label{SMOProblem}
\end{equation}$$


<br>





## 4. SVM: soft margin

However, in real situation, a hard margin hyperplane may not exists or generalizes well. So we introduce soft margin hyperplane to find a good balance between maximal margin and smallest number of errors. The optimal problem becomes $$\mathcal{H_S}$$


$$\begin{equation}
\begin{aligned}
    \min\limits_{w} \quad & \frac{1}{2}w^Tw + C\sum_{i=1}^n \zeta_i \\
    \text{s.t.} \quad & 1 - y_{i}(\phi(x_{i})w+b) \leq \zeta_i, \quad i = 1, 2, ... n
\end{aligned}
\end{equation}$$

where $$C$$ controls the trade-off and $$\zeta_i$$  is the hinge loss function

$$\begin{equation}
 \zeta_i = \max(0, 1 - y_{i}(\phi(x_{i})w+b))
\end{equation}$$


The dual becomes

$$\begin{equation}
\begin{aligned}
    & \max \limits_\lambda L(\lambda) =
        \sum_{i=1}^n \lambda_i - \frac{1}{2} \sum_{i,j=1}^n \lambda_i \lambda_j y_i  y_j K(x_i, x_j)\\

    & \text{s.t.} \quad \sum_{i=1}^n \lambda_i y_i = 0, 
                \quad 0 \leq \lambda_i \leq C, \quad i=1, 2, ... n
\end{aligned}
\label{SoftDualProblem}
\end{equation}$$



## 5. Kernels

Kernels are what make SVM magical and it is not limited to SVMs. A linear model can become a non-linear model by kernel functions. Through kernel functions, we can increase feature dimensions without even actually computing the feature mapping or even knowing the feature mapping function. The most common kernel functions are:

* Gaussian kernel: $$K(x_i, x_j) = exp(- \gamma {\| x_i - x_j \|}^2)$$, $$\gamma \ geq 0$$.

* Polynomial kernel: $$K(x_i, x_j) = (<x_i, x_j> + c)^d$$, d means degree.

* Expoenetial kernel: $$K(x_i, x_j) = exp(- \gamma \| x_i - x_j \|)$$.

* Sigmoid kernel: $$K(x_i, x_j) = tanh(\gamma<x_i, x_j> + c)$$.



## 6. Iterative Methods

The traditional iterative method is __sequential minimal optimization__(SMO). [SMO](#refSMO) is an extreme case of the decomposition method used in __LIBSVM__. LIBSVM is a popular open source library for SVMs. __Decomposition methods__ translate a problem into a new one that is easier to solve, by grouping variables into sets, and solving a subproblem for each set. SMO updates only two $$\lambda$$ per iteration, other lagrange multipliers considered as constants, thus problem becomes two variables QP optimization per iteration. With the constraint as $$\eqref{SoftDualProblem}$$, we have

$$\begin{equation}
\begin{aligned}
    & \sum_{i=1}^n \lambda_i y_i = 0 \\
    \Rightarrow \quad & \lambda_i y_i + \lambda_j y_j = - \sum_{k \neq i, j}^n \lambda_k y_k \\
    \Rightarrow \quad & \lambda_j = (- \sum_{k \neq i, j}^n \lambda_k y_k - \lambda_i y_i)y_j \\

    \Rightarrow \quad & \lambda_j = (a - \lambda_i y_i)y_j


\end{aligned}
\end{equation}$$

where $$a = - \sum_{k \neq i, j}^n \lambda_k y_k$$. The Langrange in $$\eqref{SoftDualProblem}$$ becomes


$$\begin{equation}
\begin{aligned}
    L(\lambda) & =
    \sum_{k=1}^n \lambda_k - \frac{1}{2} \sum_{k,h=1}^n \lambda_k \lambda_h y_k  y_h K_{kh} \\

    & = \lambda_i(1 - y_i y_j) - \frac{1}{2} [\, \lambda_i^2K_{ii} + (a - \lambda_i y_i)^2K_{jj} \\
        & \quad + 2\lambda_i y_i (a - \lambda_i y_i)K_{ij} \\
        & \quad + 
        \lambda_i y_i \sum_{k \neq i, j}^n \lambda_k y_k (K_{ik} - K_{jk}) ]\, + constant
    
\end{aligned}
\end{equation}$$

where $$K_{ij} = K(x_i, x_j)$$. Therefore, we have

$$\begin{equation}
\begin{aligned}

    \partial L / \partial \lambda_i & = 1 - y_i y_j - (K_{ii} + K_{jj} - 2K_{ij})\lambda_i \\
    & \quad - a y_i (K_{ij} - K_{jj}) - \frac{1}{2} y_i \sum_{k \neq i, j}^n \lambda_k y_k (K_{ik} - K_{jk})

\end{aligned}
\end{equation}$$

and 

$$\begin{equation}
\begin{aligned}

    \partial^2 L / \partial \lambda_i^2 
        & = - (K_{ii} + K_{jj} - 2K_{ij}) \\
        & = - \sum_{h=1}^m(\phi^2(x_{ih})-2\phi(x_{ih})\phi(x_{jh})+\phi^2(x_{jh})) \\
        & = - \sum_{h=1}^m(\phi(x_{ih}) - \phi(x_{jh}))^2 \leq 0

\end{aligned}
\end{equation}$$

Therefore,

$$\begin{equation}
\begin{aligned}

    (\lambda_i^{u})^{*} = \frac{1- y_i y_j + \frac{1}{2}y_i (\, \sum_{k \neq i, j}^n \lambda_k y_k(2K_{ij} - 2K_{jj} + K_{jk} - K_{ik}) )\, }
    {K_{ii} + K_{jj} - 2K_{ij}} \\

\end{aligned}
\end{equation}$$


With the constraint $$\eqref{SoftDualProblem}$$, we have


$$\begin{equation}
\begin{aligned}

    & \begin{cases}
        0 \leq \lambda_i \leq C \\
        0 \leq \lambda_j \leq C \\
        \lambda_i^t y_i + \lambda_j^t y_j + \sum_{k \neq i, j}^n \lambda_k^t y_k = 0 \\
        \lambda_i^{t-1} y_i + \lambda_j^{t-1} y_j + \sum_{k \neq i, j}^n \lambda_k^{t-1} y_k = 0 \\
        \lambda_k^t = \lambda_k^{t-1}, k \neq i, j
    \end{cases} \\

    \Rightarrow
    & \begin{cases}
        L = \max \{0, \lambda_i^{t-1} + \lambda_j^{t-1} - C \}, \ H = \min \{ C, \lambda_i^{t-1} + \lambda_j^{t-1} \}, \quad if \ y_i y_j = 1\\

        L = \max \{ 0, \lambda_i^{t-1} - \lambda_j^{t-1} \}, \ H = \min \{ C, C + \lambda_i^{t-1} - \lambda_j^{t-1} \}, \quad if \ y_i y_j = -1

    \end{cases}
    
\end{aligned}
\end{equation}$$

where $$L$$ is the lower bounder, $$H$$ is the high bounder, and $$t$$ means step $$t$$. Therefore,

$$\begin{equation}
\begin{aligned}

    (\lambda_i^{u,t})^{*} = \frac{1- y_i y_j + \frac{1}{2}y_i (\, \sum_{k \neq i, j}^n \lambda_k^{t-1} y_k(2K_{ij} - 2K_{jj} + K_{jk} - K_{ik}) )\, }
    {K_{ii} + K_{jj} - 2K_{ij}} \\

\end{aligned}
\end{equation}$$


$$\begin{equation}
\begin{aligned}
    (\lambda_i^{t})^{*} =
        \begin{cases}
            L, \qquad if \ (\lambda_i^{u,t})^{*} < L \\
            
            \lambda_i^{*, u}, \qquad if \ L \leq (\lambda_i^{u,t})^{*} \leq H \\

            H, \qquad if \ (\lambda_i^{u,t})^{*} > H \\


        \end{cases}

\end{aligned}
\end{equation}$$


After solving the optimal Lagrange multipliers per iteration, the next problem is how to choose Lagrange multipliers at each iteration. In LIBSVM, it is called __working set selection__. There are several methods working with it. The current one used in LIBSVM is via the "maximal violating pair". For more details, please reference [Chang and Lin(2001)](#refLibsvm) or [Fan, Chen, and Lin(2005)](#refWSS).

Other iterative methods includes __subgradient descent__. Subgradient methods are iterative methods for solving convex minimization problems.(via wiki) Basically, it is very similar to gradient descent. However, subgradient descent can apply to non-differentiable objective function.
















## <a name="sectionDual"></a>7. Lagrangian dual problem

Given the primal problem $$\mathcal{P_0}$$

$$\begin{equation}
\begin{aligned}
    \min\limits_{x \in \mathbb{D}} \quad & f(x) \\
    \text{s.t.} \quad & g_{i}(x) \leq 0, \quad i=1, 2, ... m \\
    \quad & h_{j}(x) = 0, \quad j = 1, 2, ... p
\end{aligned}
\label{originalPrimal}
\end{equation}$$

with the domain $$\mathbb{D} \subset \mathbb{R^n}$$, the $$\textit{Lagrangian function}$$ $$\mathbb{L} : \mathbb{R^n} \times \mathbb{R^m} \times \mathbb{R^p} \to \mathbb{R}$$ is defined as

$$\begin{equation}
    L(x, \lambda, \nu) = f(x) + \sum_{i=1}^{m}\lambda_{i}g_i(x)
        + \sum_{j=1}^{p}\nu_{j}h_j(x) \label{lagrangian}
\end{equation}$$

where 

$$\begin{equation}
    \lambda = \begin{bmatrix}
                \lambda_{1} \\
                \lambda_{2} \\
                \vdots \\
                \lambda_{m}
    \end{bmatrix}, \quad

    \nu = \begin{bmatrix}
            \nu_{1} \\
            \nu_{2} \\
            \vdots \\
            \nu_{p}
    \end{bmatrix}
\end{equation}$$

are $$\textit{Lagrange muptiplier vectors}$$, $$f(x)$$ and $$g_i(x)$$ are convex. With $$\textit{Lagrangian function} \eqref{lagrangian}$$ , we denoted the $$\textit{primal problem}$$ as

$$\begin{equation}
\mathcal{P}: \quad \min \limits_{x} \max \limits_{\lambda, \nu } L(x, \lambda, \nu)
\label{primal}
\end{equation}$$

and the $$\textit{dual problem}$$ as

$$\begin{equation}
\mathcal{D}: \quad \max \limits_{\lambda, \nu} \min \limits_{x}  L(x, \lambda, \nu)
\label{dual}
\end{equation}$$

$$p^*$$ and $$d^*$$ are the optimal values.


***THEOREM 1.*** If 
$$\lambda_i \geq 0$$,  $$g_i(x) \leq 0$$ and $$h_j(x) = 0$$,
we have $$ \max \limits_{\lambda, \nu}L(x, \lambda, \nu) = f(x) $$ and
$$p^*$$ is the optimal value for primal problem $$\ref{originalPrimal}$$.

***PROOF 1:***

$$\begin{equation}
\begin{aligned}

& \partial L / \partial \lambda_i = g_i(x) \leq 0 \\

\Rightarrow \quad & 
    \begin{cases}
        \lambda_i^* = 0, \quad g_i(x) < 0 \\
        \lambda_i^* \geq 0, \quad g_i(x) = 0
    \end{cases} \\


\Rightarrow \quad & \lambda_i^* g_i(x) = 0  \\

\Rightarrow \quad & \max \limits_{\lambda, \nu} L(x, \lambda, \nu) =
    L(x, \lambda^*, \nu^*)
    = f(x)
\end{aligned}\end{equation}$$



***THEOREM 2.*** If $$f(x)$$ and $$g_i(x)$$ are convex and continuously differentiable at a point $$x^{*}$$, 
$$\lambda_i \geq 0$$,  $$g_i(x) \leq 0$$ and $$h_j(x) = 0$$, therefore

$$\begin{equation}
p^* = d^*
\end{equation}$$

***PROOF 2:*** Suppose $$(x_p^*, \lambda_p^*, \nu_p^*)$$ are the solutions to primal problem $$\mathcal{P}$$ $$\eqref{primal}$$, same as ***Proof 1***, we have

$$\begin{equation}
\begin{aligned}

    & \partial \mathcal{P} / \partial \lambda_i = g_i(x) \leq 0 \\
    \Rightarrow & \quad \lambda_{p,i}^* g_i(x) = 0 \\
    \Rightarrow & \quad \mathcal{P} = \min \limits_{x} f(x) \\
    \Rightarrow & \quad \partial f(x) / \partial x_p^* = 0
    
\end{aligned}
\label{proof2.1}
\end{equation}
$$

Similarly, suppose $$(x_d^*, \lambda_d^*, \nu_d^*)$$ are the solutions to primal problem $$\mathcal{D}$$ $$\eqref{dual}$$, we have

$$\begin{equation}
\begin{aligned}

    & \mathcal{D} = \max \limits_{\lambda} 
        f(x_d^*) + \sum_{i=1}^{m}\lambda_{i}g_i(x_d^*) \\
    \Rightarrow & \quad \partial \mathcal{D} / \partial \lambda_i =
        g_i(x_d^*) \leq 0 \\
    \Rightarrow & \quad \lambda_{d,i}^* g_i(x_d^*) = 0 \\
    \Rightarrow & \quad \mathcal{D} = f(x_d^*) \\
    \Rightarrow & \quad \partial f(x) / \partial x_d^* = 0

\end{aligned}
\label{proof2.2}
\end{equation}$$

According to \eqref{proof2.1} and \eqref{proof2.2}, given f(x) is convex, we have

$$\begin{equation}
\begin{aligned}
     f(x_p^*) = f(x_d^*) \quad \Leftrightarrow  \quad p^* = d^*
\end{aligned}
\end{equation}$$


***THEOREM 3.*** If $$f(x)$$ and $$g_i(x)$$ are convex and continuously differentiable at a point $$x^{*}$$, 
$$\lambda_i \geq 0$$,  $$g_i(x) \leq 0$$ and $$h_j(x) = 0$$, therefore

$$\begin{equation}
\mathcal{P_0} \ \Leftrightarrow  \ \mathcal{P} \ \Leftrightarrow  \ \mathcal{D}

\end{equation}$$



## 8. References

<a name="refLibsvm"></a>[1] Chih-Chung Chang and Chih-Jen Lin(2001). [LIBSVM: a library for support vector machines](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf).

<a name="refSMO"></a> [2] 
John C. Platt(1998). [Fast Training of Support Vector Machines Using Sequential Minimal Optimization](https://pdfs.semanticscholar.org/d1fa/8485ad749d51e7470d801bc1931706597601.pdf).

<a name="refWSS"></a>[3] Rong-En Fan, Pai-Hsuen Chen, and Chih-Jen Lin(2005). [Working Set Selection Using Second Order Information for Training Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/quadworkset.pdf).

