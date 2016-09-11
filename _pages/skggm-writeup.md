---
layout: page
title: Gaussian graphical models with skggm
permalink: /how_to
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

<!--
Introduce Gaussian Graphical Models and what sparse inverse covariance estimation looks like.  This should highlight that there are trade offs to be made in how these graphs are fit -- 

we can optimize for true network to be a subset of the estimated network i.e minimize predictive loss and permit extra edges, or 

we can optimize for network to be a subset of the true network i.e. penalize false positives harshly via EBIC or Thresholding ensemble model.

Depending on how far we get, we can have the model recovery/statistical power examples on some simulated graph types for the different estimators we have (naive, two stage, ensemble model). 

Show off how to use some of our features using the studyforrest data
Initial: Fit a GGM to one subject. Get ensemble model. Do the same for all 19 subjects. Discuss what things look like.  Edges are biased and cannot be interpreted. 
Make it adaptive: 
Create a group wide weight matrix.  Eg. if edges have a low stability proportion across all subjects, make the penalty weight high, otherwise reduce the penalty to something low or near zero.  
Now do an adaptive fit on each of the subjects. 
Do some discussion of the results and what we see. 
This kind of shows off how to leverage the naive, adaptive and ensemble components. We can end with discussing that toolbox will be expanded to account for non-independence of data, more sophisticated regularizers that do sparsity/group-sparsity, etc... This makes it nice to revisit the dataset with improvements. 
-->

<!--
# Example of raw escaping

{% raw %}
  $$a^2 + b^2 = c^2$$ 
  note that all equations between these tags will not need escaping! 
{% endraw %}
-->