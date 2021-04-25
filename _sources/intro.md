# Basic Concepts of Probability Theory for AI and ML


* Author: Prof. Dr. Johannes Maucher
* Institution: Stuttgart Media University
* Version: 0.1 - Draft!!!
* Last Update: 23.04.2021

In this document the basic notions, rules and theorems of Probability Theory are introduced. The goal is to provide the skills from this subject, which are fundamental for Artificial Intelligence and Machine Learning. This means that this introduction does not claim to present the field in its entirety. 

The topics of this document are summarized in the picture below.

<figure align="center">
<img width="500" src="https://maucher.home.hdm-stuttgart.de/Pics/probBookJupyterBookOverview.png">
<figcaption>Overview on contents in this jupyterbook</figcaption>
</figure>

In [Probability Univariate](ProbabilityUnivariate) only a single random variable, either discrete or continuous, is considered. These concepts are then extended to the multivariate case in [Probability Multivariate](ProbabilityMultivariate). Here, joint- and conditional probabilitis as well as the corresponding fundamental laws such as chain-rule, marginalisation and Bayes Theorem are introduced.

In [Estimate Probability](estimateProbability) it is shown how Probability Distributions can be estimated from data. In [Example Estimate Probability](exampleProbEst) it is shown how the Python package [Pandas](https://pandas.pydata.org) can be applied to estimate joint- and conditional probabilities from discrete datasets.

Section [Bayes Classifier](parametricClassification1D) builds the bridge to Machine Learning. Concrete, is shown how the Bayes Rule is applied for Bayesian Inference and how a corresponding classification algorithm can be learned from data.

Bayesian Networks constitute an important framework to model uncertainty in Artifical Intelligence. They allow to represent uncertain knowledge and provide means for causal- and diagnostic- probabilistic inference. Section [Bayesian Networks](BayesNetAsia) demonstrates how Bayesian Networks can be constructed and applied with the Python package [pyAgrum](https://pyagrum.readthedocs.io/en/0.20.1/). In contrast to Machine Learning algorithms, Bayesian Networks provide the important capability to integrate knowledge from data with expert-knwoledge. In [Learn Bayesian Networks](BayesNetLearningWithPandas) it is shown how the parameters of a Bayesian Network can be learned from data. 

