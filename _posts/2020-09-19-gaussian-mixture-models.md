---
layout: post
title:  "A Simple Note for Gaussian Mixture Models"
date:   2020-09-19 +0800
keywords: GMM cluster
categories: Machine-Learning
---

## 1. Gaussian Mixture Model

> A Gaussian mixture is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.
> - [scikit-learn.org](https://scikit-learn.org/stable/modules/mixture.html#:~:text=A%20Gaussian%20mixture%20model%20is,Gaussian%20distributions%20with%20unknown%20parameters)


Gaussian mixture model(GMM) is an clustering algorithm similar to [k-means](/machine-learning/2020/09/14/kmeans). Unlike k-means, GMM is a soft clustering, each instance associates with a cluster with a probability. Formally, given a set of observations $$X = (x_1, x_2, ..., x_n)$$, $$x_i \in R^{d}$$, and $$i=1, 2, ..., n$$, Gaussian mixture model assumes:

1. there are $$K$$ mixture components, component value denoted as $$k \in \{1, 2, ..., K\}$$

2. $$x_i$$ comes from component $$z_i$$, $$Z = (z_1, z_2, ..., z_n)$$ is the latent variables

3. $$\begin{equation}
    P(z_i=k) = \pi_k
\end{equation}$$, s.t. $$\sum_{k=1}^K \pi_k = 1$$, $$\pi = (\pi_1, \pi_2, ...\pi_K)$$ is the mixing proportions

4. $$\begin{equation}
    P(x_i=x | z_i=k) \sim N(\mu_k, \Sigma_k)
    \end{equation}$$, $$N(\mu_k, \Sigma_k)$$ is a multivariate Gaussian distribution, the desity function is

$$\begin{equation}
    f(x_i) = \frac{1}{\sqrt{(2 \pi)^d \det \Sigma_k}}
               \exp\left( -\frac{1}{2} (x_i - \mu_k)^T \Sigma_k^{-1} (x_i - \mu_k) \right),
\end{equation}$$

where $$\Sigma_k$$ is the $$d \times d$$ covariance matrix for component $$k$$. The objective of GMM is to estimate the probabilities that each instance belongs to each component, that is 

$$\begin{equation}
    P(z_i=k | x_i),  k = 1, 2, ... K, i=1, 2, ... n
\label{obj}
\end{equation}$$



## 2. Bayes and Maximum Likelihood Estimation

With Bayes rule, we have

$$\begin{equation}
\begin{aligned}
    P(z_i=k|x_i) & = \frac{P(x_i|z_i)P(z_i=k)}{P(x_i)} \\
    & = \frac{P(x_i|z_i)P(z_i=k)}{\sum_{k=1}^KP(x_i|z_i)P(z_i=k)} \\
    & = \frac{\pi_k*N(\mu_k, \Sigma_k)}{\sum_{k=1}^K\pi_k*N(\mu_k, \Sigma_k)} \\
    & = w_{ik}

\end{aligned}
\label{weights}
\end{equation}$$

If we know parameters $$\theta = (\pi_1, \pi_2, ..., \pi_K, \mu_1, \mu_2, ..., \mu_K, \Sigma_1, \Sigma_2, ..., \Sigma_K)$$, we will get the objective. Therefore, the target becomes to estimate the unknown parameter $$\theta$$. To estimate $$\theta$$, GMM uses maximum likelihood estimation(MLE),

$$\begin{equation}
\begin{aligned}
    L(\theta|X, Z) & = P(X, Z|\theta) \\
    & = \prod_{i=1}^n \sum_{k=1}^K P(x_i=x|z_i=k)P(z_i=k) \\
    & = \prod_{i=1}^n \sum_{k=1}^K \pi_k N(\mu_k, \Sigma_k)
\end{aligned}
\end{equation}$$

log-likelihood becomes

$$\begin{equation}
\begin{aligned}
    l(\theta|X, Z) & = \sum_{i=1}^n log(\,\sum_{k=1}^K \pi_k N(\mu_k, \Sigma_k))\,
\end{aligned}
\end{equation}$$

Normally, we can get

$$\begin{equation}
    \theta^{*} = \underset{\theta}{\operatorname{argmax}} l(\theta|X, Z)
\end{equation}$$

by set

$$\begin{equation} \frac{\partial l}{\partial \theta} = 0
\end{equation}$$

However, it's hard to solve $$\theta$$ with closed-form expressions now. So GMM uses the Expectation—maximization algorithm(EM).

## 3. Expectation—Maximization Algorithm


> In statistics, an expectation–maximization (EM) algorithm is an iterative method to find (local) maximum likelihood or maximum a posteriori (MAP) estimates of parameters in statistical models, where the model depends on unobserved latent variables.
> - [wiki](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)

EM iterates the expectation step(E-step) and the maximization step(M-step) until converge:

1. initializes $$\theta = (\pi_1, \pi_2, ..., \pi_K, \mu_1, \mu_2, ..., \mu_K, \Sigma_1, \Sigma_2, ..., \Sigma_K)$$ to $$\theta_0$$
2. start iteration until convergence or reach the max iterate number, at step $$t\geq0$$:

    2.1 E-step: with $$\theta_{t}$$, calculates $$\begin{equation}
    P(z_i=k|x_i)
    \end{equation}$$ through equation $$\eqref{weights}$$ for $$k=1, 2, ..., K$$ and $$i=1, 2, ..., n$$

    2.2 M-step: estimates $$\theta_{t+1}$$ by Maximum Likelihood Estimation



In **E-step**, EM defines the expected value of the log-likelihood function of $$\theta$$ as

$$\begin{equation}
    \mathbf{Q}(\theta|\theta_{t}) = \mathbf{E}_{Z|X, \theta^t}(logL(\theta;X,Z))
\end{equation}$$

In **M-step**, EM estimates next $$\theta$$ as

$$\begin{equation}
    \theta^{t+1} = \underset{\theta}{\operatorname{argmax}} \mathbf{Q}(\theta|\theta_{t}) \quad s.t. \ \sum_{k=1}^K \pi_k = 1
\end{equation}$$


## 4. GMM with EM


In GMM situation, we have

$$\begin{equation}
\begin{aligned}
    L(\theta;x_i, z_i)
    & = \prod_{k=1}^K I(z_i=k) P(x_i, z_i|\theta) \\
    & = \prod_{k=1}^K I(z_i=k) P(x_i|z_i, \theta) P(z_i | \theta) \\
    & = \prod_{k=1}^K I(z_i=k) N(x_i|\mu_k, \Sigma_k) \pi_k
\end{aligned}
\end{equation}$$

where $$I(z_i=k)$$ is a indicator function. And

$$\begin{equation}
    L(\theta|X, Z) = \prod_{i=1}^n L(\theta;x_i, z_i)
\end{equation}$$


In **E-step**, the expected value of the log-likelihood function becomes


$$\begin{equation}
\begin{aligned}
    \mathbf{Q}(\theta|\theta_{t})
    & = \mathbf{E}_{Z|X, \theta^t}(logL(\theta;X,Z)) \\
    & = \mathbf{E}_{Z|X, \theta^t}\sum_{i=1}^nlog(L(\theta;x_i, z_i)) \\
    & = \sum_{i=1}^n \mathbf{E}_{z_i|x_i, \theta^t} log(L(\theta;x_i, z_i)) \\
    & = \sum_{i=1}^n \mathbf{E}_{z_i|x_i, \theta^t}  \sum_{k=1}^K I(z_i=k) *(log\pi_k + logN(x_i|\mu_k, \Sigma_k)) \\
    & = \sum_{i=1}^n \sum_{k=1}^K P(z_i=k|x_i) * (log\pi_k + logN(x_i|\mu_k, \Sigma_k))
\end{aligned}
\end{equation}$$

Combine the current $$\theta_{t}$$ and equation $$\eqref{weights}$$, we have 

$$\begin{equation}
    w_{ik} = P(z_i=k|x_i)
\end{equation}$$

where $$w_{ik}$$ is a constant. Thus

$$\begin{equation}
    \mathbf{Q}(\theta|\theta_{t}) = \sum_{i=1}^n \sum_{k=1}^K w_{ik} * (log\pi_k + logN(x_i|\mu_k, \Sigma_k))
\end{equation}$$

In **M-step**, the optimal problem becomes

$$\begin{equation}
\begin{aligned}
    \underset{\theta}{\operatorname{max}} \quad & \mathbf{Q}(\theta|\theta_{t}) \\
    \text{s.t.} \quad & \sum_{k=1}^K \pi_k = 1
\end{aligned}
\end{equation}$$


which is a Lagrangian problem. Therefore,


$$\begin{equation}
\begin{aligned}
    H(\theta, \lambda) = \mathbf{Q}(\theta|\theta_{t}) + \lambda(\sum_{k=1}^K \pi_k - 1)
\end{aligned}
\end{equation}$$


Take partial derivatives, we have 


$$\begin{equation}
\begin{aligned}
    \frac{\partial H}{\partial \mu_k}
    & = \sum_{i=1}^n w_{ik} * \frac{\partial logN(x_i|\mu_k, \Sigma_k)}{\partial \mu_k} \\
    & = \sum_{i=1}^n w_{ik} * \frac{(x_i - \mu_k)}{\Sigma_k^2}
\end{aligned}
\label{partialmu}
\end{equation}$$

and 

$$\begin{equation}
    \frac{\partial H}{\partial \Sigma_k}
     = \Sigma_k^{-3} * \sum_{i=1}^n w_{ik} * [(x_i-\mu_k)(x_i-\mu_k)^{T} -\Sigma_{k}^{2}]
\label{partialSigma}
\end{equation}$$

and

$$\begin{equation}
    \frac{\partial H}{\partial \pi_k}
     = \sum_{i=1}^n w_{ik} / \pi_k + \lambda
\label{pi_k}
\end{equation}$$

and 

$$\begin{equation}
    \frac{\partial H}{\partial \lambda}
     = \sum_{k=1}^K \pi_k - 1
\label{lambda}
\end{equation}$$


Set $$\eqref{partialmu}$$ equal to zero, we have

$$\begin{equation}
    \mu_k^* = \frac{\sum_{i=1}^n w_{ik}x_i}{\sum_{i=1}^n w_{ik}}
\end{equation}$$

Set $$\eqref{partialSigma}$$ equal to zero, and combine with $$\mu_k^*$$, we have

$$\begin{equation}
    \Sigma_k^* = \frac{\sum_{i=1}^n w_{ik}(x_i-\mu_k)(x_i-\mu_k)^{T} }{\sum_{i=1}^n w_{ik}}
\end{equation}$$

Set $$\eqref{pi_k}$$ and $$\eqref{lambda}$$ equal to zero, we have

$$\begin{equation}
    \pi_k^* = \frac{\sum_{i=1}^n w_{ik}}{\sum_{k=1}^K \sum_{i=1}^n w_{ik}} 
\end{equation}$$

Therefore, we got $$\theta_{t+1} = (\mu_1^*, \mu_2^*, ... \mu_K^*, \Sigma_1^*, \Sigma_2^*, ... \Sigma_K^*, \pi_1^*, \pi_2^*, ... \pi_K^*)$$.




## 5. A simple python implement