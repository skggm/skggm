---
layout: page
title: Gaussian graphical models with skggm
permalink: /how_to
redirect_from: /
---

<script src='https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'></script>


Given $$n$$ independently drawn, $$p$$-dimensional Gaussian random samples $$S \in \mathbb{R}^{n \times p}$$, the maximum likelihood estimate of the inverse covariance matrix $$\Theta$$ can be computed via the _graphical lasso_, i.e., the program

$$(\Sigma^{*})^{-1} = \Theta^{*} = \underset{\Theta \succ 0}{\mathrm{arg\,min}}~ -\mathrm{log\,det}\Theta + \mathrm{tr}(S\Theta) + \|\Theta\|_{1, \Lambda}$$

where $$\Lambda \in \mathbb{R}^{pxp}$$ is a symmetric non-negative weight matrix and

$$\|\Theta\|_{1, \Lambda} = \sum_{i,j=1}^{p} \lambda_{ij}|\Theta_{ij}|$$

is a regularization term that promotes sparsity \[[Hsieh et al.](http://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf)\]. This is a generalization of the scalar $$\lambda$$ formulation found in \[[Friedman et al.](http://statweb.stanford.edu/~tibs/ftp/glasso-bio.pdf)\] and implemented [here](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphLassoCV.html).

In [skggm](https://github.com/jasonlaska/skggm) we provide a [scikit-learn](http://scikit-learn.org)-compatible implementation of the program above and a collection of modern best practices for working with the graphical lasso.   

# Methods and tradeoffs 
Math version (vs. github readme version).  Can include some of the same images from the estimator suite.

{% highlight python %}
from inverse_covariance import QuicGraphLassoCV

model = QuicGraphLassoCV()
model.fit(X)  # X is data matrix of shape (n_samples, n_features) 

# see: model.covariance_, model.precision_, model.lam_
{% endhighlight %}

# Example: Study Forrest data set

---

This is an ongoing effort. We'd love your feedback on which algorithms we should provide bindings for next and how you're using the package. We also welcome contributions. 

[@jasonlaska](https://github.com/jasonlaska) and [@mnarayn](https://github.com/mnarayan)

---

# Example of raw escaping

{% raw %}
  $$a^2 + b^2 = c^2$$ --> note that all equations between these tags will not need escaping! 
{% endraw %}
