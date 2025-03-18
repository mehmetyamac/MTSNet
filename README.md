# MTS Layers
Multiscale Tensor Summation (MTS) Factorization is a novel neural network operator that realizes tensor summation through multiple scales via Tucker decomposition-like mode products. Distinct from other tensor decomposition methods in the literature that mainly focus on network compression, MTS Layers can serve as a novel backbone neural layer alternative to current ones such as MLP, CNN, etc. 

![MTS Layer (1)](https://github.com/user-attachments/assets/f7181efd-055d-48a4-ad11-29761f84103f)

## Linear Operators in machine learning
Any linear system can be represented in matrix-vector form given by:

$$
\mathbf{y} = \mathbf{Ds},
$$

where:
- $\mathbf{y} \in \mathbb{R}^m$ is the measurement vector or feature map (output of the layer in machine learning literature),
- $\mathbf{D}$ is an $m \times n$ transformation matrix, i.e., in machine learning it is learned via training
- $\mathbf{y}  \in \mathbb{R}^n$ is the input vector (input of this layer in deep neural networks).

# Tucker Decomposition-based Factorization

$$
\mathbf{\mathcal{Y}} = \mathbf{\mathcal{S}} \times_1 \mathbf{D_1} \times_2 \mathbf{D_2} \ldots \times_{J-1} \mathbf{D_{J-1}} \times_J \mathbf{D_J}
$$

The separable multilinear operation can be written in the form of conventional matrix-vector multiplication via:

$$
\mathbf{y} = \left( \mathbf{D_1} \otimes \mathbf{D_2} \otimes \ldots \otimes \mathbf{D_J} \right) \mathbf{s}
$$

where $\otimes$ denotes the Kronecker product, and $\mathbf{y}$ is the vectorized version of the tensor $Y$
