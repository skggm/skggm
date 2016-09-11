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
The core estimator provided in [skggm](https://github.com/jasonlaska/skggm) is `QuicGraphLasso` which is a scikit-learn compatible interface to an implementation of the [QUIC](http://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf) algorithm.

{% highlight python %}
from inverse_covariance import QuicGraphLasso

model = QuicGraphLasso(
    lam=,               # Graph lasso penalty $$\Lambda$$ (scalar or matrix) 
    mode=,              # If 'default': single estimate, If 'path': estimates over sequence of scaled penalties 
    path=,              # Sequence of penalty scales (scalars) mode='path'
    init_method=,       # Inital covariance estimate: 'corrcoef' or 'cov'
    auto_scale=True,    # If True, scales penalty by max off-diagonal entry of the sample covariance
)
model.fit(X)  # X is data matrix of shape (n_samples, n_features) 

# see: model.covariance_, model.precision_, model.lam_
{% endhighlight %}

## True network as as subset of the estimated network

## Estimate network as a subset of the true network

## Refining coefficient support and values via adaptive methods



# Example: Study Forrest data set

---

This is an ongoing effort. We'd love your feedback on which algorithms we should provide bindings for next and how you're using the package. We also welcome contributions. 

[@jasonlaska](https://github.com/jasonlaska) and [@mnarayn](https://github.com/mnarayan)

---

<!--
Introduce Gaussian Graphical Models and what sparse inverse covariance estimation looks like.  This should highlight that there are trade offs to be made in how these graphs are fit 

we can optimize for true network to be a subset of the estimated network i.e minimize predictive loss and permit extra edges, or 

we can optimize for network to be a subset of the true network i.e. penalize false positives harshly via EBIC or Thresholding ensemble model.

Depending on how far we get, we can have the model recovery/statistical power examples on some simulated graph types for the different estimators we have (naive, two stage, ensemble model). 


# Study Forest

Show off how to use some of our features using the studyforrest data

Initial: Fit a GGM to one subject. Get ensemble model. Do the same for all 19 subjects. Discuss what things look like.  Edges are biased and cannot be interpreted. 

Make it adaptive: 
Create a group wide weight matrix.  Eg. if edges have a low stability proportion across all subjects, make the penalty weight high, otherwise reduce the penalty to something low or near zero.  

Now do an adaptive fit on each of the subjects. 

Do some discussion of the results and what we see. 

This kind of shows off how to leverage the naive, adaptive and ensemble components. We can end with discussing that toolbox will be expanded to account for non-independence of data, more sophisticated regularizers that do sparsity/group-sparsity, etc... This makes it nice to revisit the dataset with improvements. 

Earlier notes on this data 

Goal: Explore how brain networks differ by 

emotional arousal (high versus low arousal)
emotional valence (positive or negative valence). 

- Forrest gump movie has 10 second segments annotated by emotion
- 2 hour fMRI data throughout for ~ 20 subjects

Really rich dataset of subjects listening to forrest gump movie. 
http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4416536/
http://studyforrest.org/

- Exceedingly well organized data and totally open; making the analysis fully automated will be easy 
- Preprocessed data available, or I wouldn't have time to do this
- Opportunity to have cool pictures/visualization/novel results

I think we will need to have the sparse + group-sparse version of QUIC available to make this analysis really work. Sample sizes are small and subjects are all healthy so we can borrow strength across subjects. 

Recent paper used the dataset to think about brain networks involved in interoception, so we both have some basis for interpreting our results
http://www.sciencedirect.com/science/article/pii/S1053811915008174
-->

<!--
# Example of raw escaping

{% raw %}
  $$a^2 + b^2 = c^2$$ 
  note that all equations between these tags will not need escaping! 
{% endraw %}
-->