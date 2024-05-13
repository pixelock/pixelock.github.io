---
author: Hugo Authors
title: Math Typesetting
date: 2019-03-08
description: A brief guide to setup KaTeX
math: true
---

Mathematical notation in a Hugo project can be enabled by using third party JavaScript libraries.
<!--more-->

In this example we will be using [KaTeX](https://katex.org/)

- Create a partial under `/layouts/partials/math.html`
- Within this partial reference the [Auto-render Extension](https://katex.org/docs/autorender.html) or host these scripts locally.
- Include the partial in your templates like so:  

```bash
{{ if or .Params.math .Site.Params.math }}
{{ partial "math.html" . }}
{{ end }}
```

- To enable KaTeX globally set the parameter `math` to `true` in a project's configuration
- To enable KaTeX on a per page basis include the parameter `math: true` in content files

**Note:** Use the online reference of [Supported TeX Functions](https://katex.org/docs/supported.html)

{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}
<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
{{ end }}
{{</ math.inline >}}

### Examples

Inline math: $\varphi = \dfrac{1+\sqrt5}{2}= 1.6180339887…$

Block math:
$$
 \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } } 
$$

Transformer 标准的 MHA(Multi-Head Attention) 结构中, $n_h$ 为 attention heads 数量, $d_h$ 为每个 head 内部的维度, $\mathbf{h}_t \in \mathbb{R}^d$ 代表了当前 attention layer 层中第 $t$ 个 token 的输入. 标准的 MHA 通过三个不同的参数矩阵 $W_Q,W_K,W_V\in\mathbb{R}^{d_hn_h \times d}$ 得到 $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t \in \mathbb{R}^{d_h n_h}$, 对应于:

$$
\begin{aligned}
& \mathbf{q}_t = W_Q \mathbf{h}_t \\\
& \mathbf{k}_t = W_K \mathbf{h}_t \\\
& \mathbf{v}_t = W_V \mathbf{h}_t
\end{aligned}
$$

在 MHA 中, $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ 被拆分为 $n_h$ 个 heads, 然后每个 head 中进行 attention 计算:

$$
\mathbf{q}_t + \mathbf{q}_{1}
$$

每个 head 计算得到的结果再拼接起来, 经过参数矩阵 $W_O \in \mathbb{R}^{d \times d_h n_h}$ 得到 MHA 的输出张量.

$$
\begin{aligned}
& {\left[\mathbf{q}_{t, 1} ; \mathbf{q}_{t, 2} ; \ldots ; \mathbf{q}_{t, n_h}\right]=\mathbf{q}_t,} \\
& {\left[\mathbf{k}_{t, 1} ; \mathbf{k}_{t, 2} ; \ldots ; \mathbf{k}_{t, n_h}\right]=\mathbf{k}_t,} \\
& {\left[\mathbf{v}_{t, 1} ; \mathbf{v}_{t, 2} ; \ldots ; \mathbf{v}_{t, n_h}\right]=\mathbf{v}_t,} \\
& \mathbf{o}_{t, i}=\sum_{j=1}^t \operatorname{Softmax}_j\left(\frac{\mathbf{q}_{t, i}^T \mathbf{k}_{j, i}}{\sqrt{d_h}}\right) \mathbf{v}_{j, i}, \\
& \mathbf{u}_t=W^O\left[\mathbf{o}_{t, 1} ; \mathbf{o}_{t, 2} ; \ldots ; \mathbf{o}_{t, n_h}\right]
\end{aligned}
$$

