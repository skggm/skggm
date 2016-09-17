---
layout: page
title: Gaussian graphical models with skggm
permalink: /how_to
---

<script src='https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'></script>


Given $$n$$ independently drawn, $$p$$-dimensional Gaussian random samples $$S \in \mathbb{R}^{n \times p}$$, the maximum likelihood estimate of the inverse covariance matrix $$\Theta$$ can be computed via the _graphical lasso_, i.e., the program

$$
\begin{align}
(\Sigma^{*})^{-1} = \Theta^{*} = \underset{\Theta \succ 0}{\mathrm{arg\,min}}~ -\mathrm{log\,det}\Theta + \mathrm{tr}(S\Theta) + \|\Theta\|_{1, \Lambda}
\label{eqn:graphlasso}\tag{1}
\end{align}
$$

where $$\Lambda \in \mathbb{R}^{p\times p}$$ is a symmetric non-negative weight matrix and

$$\|\Theta\|_{1, \Lambda} = \sum_{i,j=1}^{p} \lambda_{ij}|\Theta_{ij}|$$

is a regularization term that promotes sparsity \[[Hsieh et al.](http://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf)\]. This is a generalization of the scalar $$\lambda$$ formulation found in \[[Friedman et al.](http://statweb.stanford.edu/~tibs/ftp/glasso-bio.pdf)\] and implemented [here](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphLassoCV.html).

The suport of the sparse precision matrix can be interpreted as the adjency matrix of an undirected graph (with non-zero elements as edges) for correlated components in a system (from which we obtain samples). **Thus, inverse covariance estimation finds numerous applications in X, Y, and Z.**

In [skggm](https://github.com/jasonlaska/skggm) we provide a [scikit-learn](http://scikit-learn.org)-compatible implementation of the program (\ref{eqn:graphlasso}) and a collection of modern best practices for working with the graphical lasso.   

## Methods and tradeoffs 
The core estimator provided in [skggm](https://github.com/jasonlaska/skggm) is `QuicGraphLasso` which is a scikit-learn compatible interface to an implementation of the [QUIC](http://jmlr.org/papers/volume15/hsieh14a/hsieh14a.pdf) algorithm for (\ref{eqn:graphlasso}).  It's example usage (and common parameters) is:

{% highlight python %}
from inverse_covariance import QuicGraphLasso

model = QuicGraphLasso(
    lam=int or np.ndarray,  # Graph lasso penalty (scalar or matrix) 
    mode=str,               # 'default': single estimate
                            # 'path': use sequence of scaled penalties 
    path=list,              # Sequence of penalty scales mode='path'
    init_method=str,        # Inital covariance estimate: 'corrcoef' or 'cov'
    auto_scale=bool,        # If True, scales penalty by 
                            # max off-diagonal entry of the sample covariance
)
model.fit(X)                # X is data matrix (n_samples, n_features) 
{% endhighlight %}

If the input penalty `lam` is a scalar, it will be converted to a matrix with zeros along the diagonal and `lam` for all other entries. A matrix `lam` is used as-is, although it may be scaled.

After the model is fit, the estimator object will contain the covariance estimate `model.covariance_`$$\in \mathbb{R}^{p\times p}$$, the sparse inverse covariance estimate `model.precision_`$$\in \mathbb{R}^{p\times p}$$, and the penalty `model.lam_` used to obtain these estimates.  When `auto_scale=False`, the output pentalty will be identical to the input penalty, however, by default the penalty is scaled for best performance for the given data. If `mode='path'` is used, then the `path` parameter must be provided and both `model.covariance_` and `model.precision_` will be a list of $$p\times p$$ matrices of length `len(path)` and `lam_` remains a scalar. More details can be found via `help(QuicGraphLasso)`. In general, the estimators introduced here will follow this interface unless otherwise noted.  

The choice of the penalty $$\Lambda$$ can have a large impact on the kind of result obtained.  If a good $$\Lambda$$ is known _a priori_, e.g., when reproducing existing results from the literature, then look no further than this estimator (with `auto_scale='False'`).  

For a new data or new problems, we provide several methods for selecting an appropriate $$\Lambda$$. Selection methods roughly fall into two categories of performance: a) biased away from sparsity, resulting in estimates with false positive edges and where the true underlying graph is a subset of the estimate; or b) biased toward sparsity, resulting in estimates with missing edges and where the estimate is a subset of the true underlying graph.

# Model selection via cross-validation (less sparse)
One common way to find $$\Lambda$$ is via cross-validation.  Specifically, for a given grid of penalties, we fit the model on $$K$$ subsets of the data (folds) and measure the estimator performance.  We aggregate the score across the folds to determine a score for each $$\Lambda$$.

In this technique, estimator performance is measured against the sample covariance, i.e., 
$$
\Sigma_{\mathrm{S}} = \frac{1}{n - 1} \sum_{i=1}^{n} (s_{i} - \bar{s}) (s_{i} - \bar{s})^{T}
$$
for samples $$s_{i} \in S$$ and the mean of the observations $$\bar{s}$$. We provide several metrics $$d(\Sigma_{\mathrm{S}}, \Theta^{*})$$ that can be used in with cross-validation via the `score_metric` parameter:

$$
\begin{align}
- \mathrm{tr}(\Sigma_{\mathrm{S}} \cdot \Theta^{*}) + \mathrm{log\,det}~\Theta^{*} - p \cdot \mathrm{log}2\pi
\label{eqn:log_likelihood}\tag{log-likelihood}
\end{align}
$$

$$
\begin{align}
\frac{1}{2}\left( 
\mathrm{tr}(\Sigma_{\mathrm{S}} \cdot \Theta^{*})  - \mathrm{log\,det}(\Sigma_{\mathrm{S}} \cdot \Theta^{*}) - p
\right)
\label{eqn:kl_loss}\tag{kl-loss}
\end{align}
$$

$$
\begin{align}
\sum_{ij}\left(
\Sigma_{\mathrm{S}} - \Sigma^{*}
\right)^{2}
\label{eqn:frobenius}\tag{Frobenius}
\end{align}
$$

$$
\begin{align}
\mathrm{tr}~\left(\Sigma_{\mathrm{S}} \cdot \Theta^{*} - I_p\right)^{2}
\label{eqn:quadratic}\tag{quadratic}
\end{align}
$$

Cross validation can be performed with _QuicGraphLasso_ and [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html), however, we provide an optimized convenience class `QuicGraphLassoCV` that takes advantage of _path mode_ to adaptively estimate the search grid.  This implementation is closely modeled after scikit-learn's [GraphLassoCV](http://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphLassoCV.html), but additionally supports matrix penalties.

{% highlight python %}
from inverse_covariance import QuicGraphLassoCV

model = QuicGraphLassoCV(
    lam=int or np.ndarray,  # (optional) Initial penalty (scalar or matrix) 
    lams=int or list,       # If int, determines the number of grid points 
                            # per refinement
    n_refinements=int,      # Number of times the grid is refined
    cv=int,                 # Number of folds or sklearn CV object
    score_metric=str,       # One of 'log_likelihood' (default), 'frobenius', 
                            #        'kl', or 'quadratic'
    init_method=str,        # Inital covariance estimate: 'corrcoef' or 'cov'
)
model.fit(X)                # X is data matrix (n_samples, n_features) 
{% endhighlight %}

In addition to covariance and precision estimates, this class returns the best penalty in `model.lam_` and the penalty grid `model.cv_lams_` as well as the cross-validation scores for each penalty `model.grid_scores`.

<img style="margin: 0 auto;display: block;" src="assets/graph_lasso_cv.png" width="650" />
<div style="margin: 0 auto;display: block; width:600px;">
<center><i><small>
Figure 1.  Inverse covariance estimates (less sparse).  From left to right:  the original inverse covariance (precision) matrix, the sample covariance, a QuicGraphLassoCV estimate with log-likelihood for scoring (error = 0.6, support error = 4), and a QuicGraphLassoCV estimate with the Frobenius norm for scoring (error = 0.64, support error = 2).
</small></i></center>
</div>
<br>
An example is shown in Figure 1 (code for these toy examples can be found in [skggm](https://github.com/jasonlaska/skggm) in [examples/estimator_suite.py](https://github.com/jasonlaska/skggm/blob/master/examples/estimator_suite.py)). The `QuicGraphLassoCV` estimates are much sparser than the empirical covariance, but contain a superset of the true precision support. Further, the coefficient values on the true support are not particularly accurate.  In this trial of this example, the Frobenius scoring function performed better than log-likelihood and kl-divergence, however, this does not reflect how they might compare in general.  As the dimension of the samples `n_features` grows, we find that this model selection method tends to bias toward more non-zero coefficients.

# Model selection via EBIC (more sparse)
An alternative to cross-validation is the _Extended Bayesian Information Criteria_ (EBIC) \[[Foygel et al.](https://papers.nips.cc/paper/4087-extended-bayesian-information-criteria-for-gaussian-graphical-models)\],

$$
\begin{align}
EBIC_{\gamma} := -2 \cdot l(\Sigma_{\mathrm{S}}, \Theta^{*}) + |\Theta^{*}| \cdot \mathrm{log}(n) + 4 \cdot |\Theta^{*}| \cdot \mathrm{log}(p) \cdot \gamma,
\label{eqn:ebic}\tag{2}
\end{align}
$$

where $$l(\Sigma_{\mathrm{S}}, \Theta^{*})$$ is the log-likelihood between the estimate and the sample covariance and $$\mid\Theta^{*}\mid$$ is sparsity of the inverse covariance estimate.  The parameter $$\gamma$$ penalizes larger graphs.  When $$\gamma = 0$$, (\ref{eqn:ebic}) reduces to the conventional Bayesian information crieteria (BIC).

`QuicGraphLassoEBIC` is provided as a convenience class to use _EBIC_ for model selection.  This class computes a path of estimates and selects the model that minimizes the _EBIC_ criteria.  We omit showing the interface here as it is similar to the classes described above with the addition of `gamma`.

<img style="margin: 0 auto;display: block;" src="assets/ebic.png" width="500" />
<div style="margin: 0 auto;display: block; width:600px;">
<center><i><small>
Figure 2. Inverse covariance estimates (more sparse).  From left to right:  the original inverse covariance (precision) matrix, a QuicGraphLassoEBIC estimate with gamma = 0 (BIC) (error = 0.68, support error = 0), and a QuicGraphLassoEBIC estimate with gamma = 0.1 (error = 1.36, support error = 6).
</small></i></center>
</div>
<br>
An example is shown in Figure 2. The `QuicGraphLassoEBIC` estimates are much sparser than `QuicGraphLassoCV` estimates, and often contain a subset of the true precision support.  In this small dimensional example, `BIC` (gamma = 0) performed best as `EBIC` with gamma = 0.1 selected only the diagonal coefficients. As the dimension of the samples `n_features` grows, `BIC` will produce a less-sparse result and thus an increasing gamma parameter serves to obtain sparser solutions.

# Randomized model averaging
For some problems, the support of the sparse precision matrix is of primary interest.  In these cases, the support can be estimated robustly via the _random lasso_ or _stability selection_ [[Wang et al.](https://arxiv.org/abs/1104.3398), [Meinhausen et al.](https://arxiv.org/pdf/0809.2932v2.pdf)]. The skggm `ModelAverage` class implements a meta-estimator to do this. (We note this is a similar facility to scikit-learn's [_RandomizedLasso_](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RandomizedLasso.html), but for the graph lasso.)

This technique estimates the precision over an ensemble of estimators with random penalties.  Specifically, in each trial we

1. produce boostrap samples by randomly subsampling $$S$$
2. choose a random matrix penalty

A final _proportion matrix_ is then estimated by summing or averaging the precision estimates from each trial.  The precision support can be estimated by thresholding the proportion matrix.

The random penalty can be chosen in a variety of ways.  We initially offer the following methods:

- `subsampling` (no penalty): This option does not modify the penalty and only draws bootstrapped samples from $$S$$.

- `random`: This option generates a randomly perturbed penalty `lam` weight matrix.  Specifically, the off-diagonal entries take the values $$\{\lambda\eta,~ \frac{\lambda}{\eta}\}$$ with probability $$1/2$$ for one triangle ($$\eta$$ is the class parameter `lam_perturb`). 

- `fully_random`: This option generates a symmetric matrix with Gaussian off-diagonal entries (for a single triangle).  The penalty matrix is scaled appropriately for $$S$$.

`ModelAverage` takes the parameter `estimator` as a base graph lasso estimator to average.  By default, it will use `QuicGraphLassoCV` as the base estimator.  An explicit example is as follows.

{% highlight python %}
from inverse_covariance import QuicGraphLassoCV, ModelAverage

model = ModelAverage(
    estimator=estimator,    # Graph lasso estimator instance 
                            # e.g., QuicGraphLassoCV(), QuicGraphLassoEBIC()
    n_trials=int,           # Number of trials to average over
    penalization=str,       # One of 'subsampling', 'random' (default), 
                            #        or 'fully_random'
    lam=float,              # Used if penalization='random'          
    lam_perturb=float,      # Used if penalization='random'
    support_thresh=float,   # Estimate support via proportions (default=0.5)
)
model.fit(X)                # X is data matrix (n_samples, n_features)             
{% endhighlight %}

This class will contain the matrix of support probabilities `model.proportion_`$$\in \mathbb{R}^{p\times p}$$, an estimate of the support `model.support_`$$\in \mathbb{R}^{p\times p}$$, the penalties used in each trial `model.lams_`, and the indeices for selecting the subset of data in each trial `model.subsets_`.  

<img style="margin: 0 auto;display: block;" src="assets/model_average.png" width="500" />
<div style="margin: 0 auto;display: block; width:600px;">
<center><i><small>
Figure 3. Random model averaging support estimates.  From left to right:  the original inverse covariance (precision) matrix, the ModelAverage proportions matrix, and the thresholded proportions matrix (support estimate).  The threshold used in this estimate was 0.5 and the support error is 0.
</small></i></center>
</div>
<br>
An example is shown in Figure 3. The dense `model.proportions_` matrix contains the sample probability of each element containing a nonzero.  Thresholding this matrix by the default value of 0.5 resulted in a correct estimate of the support.  While this will not be the case in general, this technique generally provides a more robust support estimate than the previous methods.

# Refining coefficients via adaptive methods
Given an initial sparse estimate, we can compute a new penalty based on the estimate and refit the graph lasso with this adaptive penalty [[Zhou et al.](http://www.jmlr.org/papers/volume12/zhou11a/zhou11a.pdf), [Meinhausen et al.](http://stat.ethz.ch/~nicolai/relaxo.pdf)]. Currently refitting is always done with `QuicGraphLassoCV`.  We provide three ways of computing new weights before refitting:

- `binary`: Generates a matrix that has zeros where where the estimator was non-zero and ones elsewhere.  This is sometimes called the _relaxed lasso_ or _gelato_ [[Meinhausen et al.](http://stat.ethz.ch/~nicolai/relaxo.pdf)].

- `inverse`: For an element $$\theta_{i,j}$$ of $$\Theta$$, computes $$\frac{1}{\theta_{i,j}}$$ for non-zero coefficients and $$\mathrm{max}\{\frac{1}{\theta_{i,j}}\}$$ for the zero valued coefficients.  This is what is used in the _glasso_ R package.

- `inverse_squared`: Computes $$\frac{1}{\theta_{i,j}^{2}}$$ for non-zero coefficients and $$\mathrm{max}\{\frac{1}{\theta_{i,j}^{2}}\}$$ for the zero valued coefficients.

Since the `ModelAverage` meta-estimator produces a good support estimate, this can be used in conjunction with the `binary` option to estimate coefficient values. 

{% highlight python %}
from inverse_covariance import QuicGraphLassoCV, AdaptiveGraphLasso

model = AdaptiveGraphLasso(
    estimator=estimator,    # Graph lasso estimator instance
                            # e.g., QuicGraphLassoCV() or ModelAverage instance
    method=str,             # One of 'binary', 'inverse', or 'inverse_squared'
)
model.fit(X)                # X is data matrix (n_samples, n_features)            
{% endhighlight %}

The resulting model will contain `model.estimator_` which is a final `QuicGraphLassoCV` instance fit with the adaptive penalty `model.lam_`. 

<img style="margin: 0 auto;display: block;" src="assets/adaptive.png" width="650" />
<div style="margin: 0 auto;display: block; width:600px;">
<center><i><small>
Figure 4. Adaptive inverse covariance estimates.  From left to right:  the original inverse covariance (precision) matrix, adaptive estimate with QuicGraphLassoCV base estimator and 'inverse' method (error = 0.32, support error = 2), adaptive estimate with QuicGraphLassoEBIC (gamma = 0) base estimator and 'inverse' method (error = 0.38, support error = 10), adaptive estimate with ModelAverage base estimator and 'binary' method (error = 0.08, support error = 0).
</small></i></center>
</div>
<br>
An example is shown in Figure 4. The adaptive estimator will not only refine the estimated coefficients but also produce a new support estimate. This can be seen in the example as the adaptive cross-validation estimator produces a smaller suppor than the adaptive BIC estimator, the opposite of what we found in the non-adaptive examples. It is clear that the estimated values of the true support are much more accurate for each method combination.  


## Example: Study Forrest data set

**TODO**

---

This is an ongoing effort. We'd love your feedback on which algorithms we should provide bindings for next and how you're using the package. We also welcome contributions. 

[@jasonlaska](https://github.com/jasonlaska) and [@mnarayn](https://github.com/mnarayan)


<!--
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