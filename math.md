---
layout: post
title: "My First Math Blog:"
date: 2025-12-04 08:00:00 +0700
categories: math tutorial
tags: [latex, mathjax, algebra]
---
$$
\begin{eqnarray}
    Q = X_{input} W_{Q} \\
    K = X_{input} W_{K} \\
    V = X_{input} W_{V}
\end{eqnarray}
$$

\begin{itemize}
    \item Q (query): what am I looking for (current word)?
    \item K (keys): what does each item offer (other words including current word)?
    \item V (values): aggregated information based on Q, K (content created by these words)
\end{itemize}

Attention guided step by step. For instance, we are using a dataset of $n$ samples, $t$ time values, and $d$ features:
\begin{itemize}
    \item Input embedding: map each input $X_i$ $\in$ space $ \mathbb{R}^{t \times d}$ to $X_{i-embed}$ $\in$ $\mathbb{R}^{t \times d_{embed}}$ by a linear projection in which $d_{embed}$ represents the dimension of embedding space 
    \begin{eqnarray}
        X_{i-embed} = W_{linear} \times X_{i} + \mathrm{bias}  \quad (optional)
    \end{eqnarray}
    
    \item Positional embedding takes into account sequence order 
    \begin{eqnarray}
        X_{input} = X_{i-embed-position} = X_{i-embed} + \mathrm{POS_ENC}
    \end{eqnarray}

    \item Q, K, V: For each of $num_heads$ attention heads, we perform
        \begin{eqnarray}
            Q = X_{input} W_{Q} \\
            K = X_{input} W_{K} \\
            V = X_{input} W_{V}
        \end{eqnarray}
        where $W_Q, W_K, W_V$ are learnable weight matrix $\in$ $\mathbb{R}^{d_{embed} \times d_{h}}$ with $d_h = d_{embed} // h$. As a result, Q,K,V $\in$ $\mathrm{R}^{t \times d_{h}}$.

    \item Scaled dot-product (not matrix multiplication) attention:
        \begin{itemize}
            \item Attention weights (scores): 
            \begin{eqnarray}
                \alpha_{i,j} = softmax_{j} (\dfrac{Q_{i} K_{j}}{\sqrt(d_h}) \in \mathrm{R}^{t \times t} 
            \end{eqnarray}
\href{https://crlannister.medium.com/the-attention-mechanism-in-transformers-understanding-the-basics-f1f9cd2fbaf9}{
Noted that we apply softmax along last dimension (j) which corresponds to the keys} 

            \item Attention output per head: $head_{k}$ = $\alpha V$ $\in$ $\mathrm{R}^{t \times d_h}$
        \end{itemize}

    \item Concatenate multi-heads:
    \begin{eqnarray}
        \mathbf{Concat}(\mbox{head}_1, ..., \mbox{head}_h) \xrightarrow[\mbox{learnable} W_0]{\mbox{a linear layer (d-embed, d-embed)}}  \mathbf{MSHA}(X_i)
    \end{eqnarray}
    where $W_0 \in$ $\mathbf{R}^{h * d_{h} \times d_{embed}}$is learnable matrix => $\mathbf{MSHA}(X_i) \in \mathrm{R}^{t \times d_{embed}}$
\end{itemize}

We should not confuse between : Attention weights $\alpha_{ij}$, attention output (prdoduct of attention weights with $V$) and learnable weights matrix $W_Q, W_K, W_V$
