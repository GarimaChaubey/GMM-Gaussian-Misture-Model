# Gaussian Mixture Models (GMM) — Full Guide with Intuition, Math, and Code

This repository provides a complete, beginner-friendly yet mathematically clear explanation of **Gaussian Mixture Models (GMM)** along with the **EM Algorithm**, **visualizations**, and **comparison with K-Means**.  
Perfect for students, beginners in Machine Learning, and anyone trying to understand clustering beyond K-Means.

---

# 📘 Table of Contents

1. [Introduction](#introduction)  
2. [Why GMM?](#why-gmm)  
3. [Understanding a Single Gaussian](#understanding-a-single-gaussian)  
4. [Gaussian Mixture Model](#gaussian-mixture-model)  
5. [Latent Variables & Generative View](#latent-variables--generative-view)  
6. [EM Algorithm — Complete Explanation](#em-algorithm--complete-explanation)  
7. [GMM Pseudocode](#gmm-pseudocode)  
8. [GMM vs K-Means](#gmm-vs-k-means)  
9. [Python Code for GMM](#python-code-for-gmm)  
10. [Python Code for K-Means Comparison](#python-code-for-k-means-comparison)  
11. [When Should You Use GMM?](#when-should-you-use-gmm)  
12. [Project Structure](#project-structure)  
13. [References](#references)

---

# 🧠 Introduction

A **Gaussian Mixture Model (GMM)** is a **probabilistic clustering model** that assumes data is generated from a mixture of multiple Gaussian distributions.

Unlike K-Means, GMM provides:

- **Soft clustering**  
- **Elliptical cluster shapes**  
- **Density estimation**

The mixture model is:

$$
p(x)=\sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

GMM learns these parameters using the **Expectation–Maximization (EM)** algorithm.

---

# ❓ Why GMM?

K-Means assumes:

- clusters are spherical  
- clusters have equal size  
- each point belongs to only one cluster  

But real-world data rarely behaves like this.

GMM fixes it because:

✔ Clusters can overlap  
✔ Covariance can be different  
✔ Membership is probabilistic  

---

# 📐 Understanding a Single Gaussian

A $d$-dimensional Gaussian distribution:

$$
\mathcal{N}(x\mid\mu,\Sigma)=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(
-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)
\right)
$$

- $\mu$ = mean  
- $\Sigma$ = covariance  

---

# 🧩 Gaussian Mixture Model

A GMM assumes:

$$
p(x)=\sum_{k=1}^{K}\pi_k \, \mathcal{N}(x\mid\mu_k,\Sigma_k)
$$

Where:

| Parameter | Meaning |
|----------|----------|
| $\pi_k$ | mixing weight |
| $\mu_k$ | cluster mean |
| $\Sigma_k$ | covariance |

---

# 🎭 Latent Variables & Generative View

For each sample:

1. Choose component:

$$
P(z_i = k) = \pi_k
$$

2. Sample data:

$$
x_i \sim \mathcal{N} (\mu_k, \Sigma_k)
$$

These hidden labels $z_i$ are **latent**.

---

# 🔁 EM Algorithm — Complete Explanation

Goal:

> Maximize likelihood of data when assignments are unknown.

---

## 🟦 E-Step: Compute Responsibilities

Probability that point $x_i$ belongs to cluster $k$:

$$
\gamma_{ik}=
\frac{
\pi_k \, \mathcal{N}(x_i\mid\mu_k,\Sigma_k)
}{
\sum_{j=1}^{K}
\pi_j \mathcal{N}(x_i\mid\mu_j,\Sigma_j)
}
$$

Interpretation:

- $\gamma_{ik} \in [0,1]$
- soft assignment  
- $\sum_k \gamma_{ik}=1$

---

## 🟧 M-Step: Update Parameters

Let:

$$
N_k = \sum_{i=1}^{N} \gamma_{ik}
$$

### Update mixing weights:

$$
\pi_k = \frac{N_k}{N}
$$

### Update means:

$$
\mu_k = \frac{1}{N_k}\sum_{i=1}^N \gamma_{ik} x_i
$$

### Update covariances:

$$
\Sigma_k =
\frac{1}{N_k}
\sum_{i=1}^N
\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^T
$$

---

## 🟩 Log-Likelihood & Convergence

Compute log-likelihood:

$$
\log L=
\sum_{i=1}^N
\log
\left(
\sum_{k=1}^{K}
\pi_k \mathcal{N}(x_i\mid\mu_k,\Sigma_k)
\right)
$$

Stop when:

$$
|\log L_{\text{new}} - \log L_{\text{old}}| < \epsilon
$$

---

# 🧾 GMM Pseudocode

```text
Initialize μk, Σk, πk

repeat:
    # E-step
    compute responsibilities γik

    # M-step
    update μk using weighted mean
    update Σk using weighted covariance
    update πk using effective counts Nk

    compute log-likelihood

until convergence
