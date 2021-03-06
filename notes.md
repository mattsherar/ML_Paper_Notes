---
title: ML Paper Notes
subtitle: >
  Notes on specific ML Papers
---

## Time Series Data Augmentation for Deep Learning: A Survey
___

- systematically review different data augmentation methods for time series
- emperically compare different methods on various tasks such as anomaly detection, classification and forecasting
- basic idea of augmentation is to generate synthetic datasets conering unexplored input space while maintaining correct labels
- challanges include temporal dependancy and multivariate structure of timeseries
- decomposition methods: 
    + time series decomposed into trend, seasonality and irregular components
    + different transofmations applied to the individual components
- model based methods:
    + statistical model learned from the data and then pertubation performed on the parameter space
    + GAN's 
- Time domain:
    + most straightforward method
    + manipulate the original time series directly
    + inject guassian noise or more complicated noise patterns
- window warping:
    + similar to DTW, select a window then compress (downsample) or extend (upsample) then crop
- frequency domain:
    + perturbations in both amplitutde spectrum and phase spectrum
- on various forecasting tasks using deep models (DeepAR, MQRNN and Transformer) (tasks: electricty, traffic, M4) data augmentation provided significant average improvement in Mean absolute scaled error but did significantly worsen performance for specific model-data combos


## Deep learning of dynamical attractors from time series measurements 
___

- the goal is to develop a general purpose method for reconstructing the d-dimensional attractor of an unknown dynamical system given only a univariate measurement time series
- introduce a custom loss function and regularizer for autoencoders (false nearest neighbour loss)
- discover underlying attractors in datasets from climate science, neuroscience and electrophysiology
- 

## Unsupervised Scalable Representation Learning for Multivariate Time Series
___

- their method can handle varying and long-length multivariate time series
- They use exponentially dilated convolutions in their encoders
- Triplet loss employing time-based negative sampling
- test how representations can be used for classification tasks
- triplet loss works as follows:
    + select a segment x_ref of a given time series
    + select a segment x_pos that is contained within x_ref
    + choose another segment x_neg at random from a different series (or the same one if it is long enough)
    + the loss pushes the computed representations to distinguish between x_ref and x_neg and assimilate x_ref and x_pos
- 
- 


## Unsupervised Feature Extraction by Time Contastive Learning and Nonlinear ICA
___

- propose a new principal for unsupervised deep learning from time series which uses the non-stationarity structure of the data
- representation allows for optimal discrimination of the time segments
- they show that their method (TCL) can be related to a non-linear ICA model, when ICA includes temporal non-stationarities
- temporal coherence or slow feature analysis try to find features which change as slowly as possible, originally proposed for learning invariant features
- learned features should enable discrimination between different time windows
- formulate a generative model in which independant components have different distributions in different time windows
- Basic Algo:
    + divide multivariate time series into segment windows
    + associate each data point with the corresponding segment window in which it is contained
    + learn a feature extractor h() together with linear regression to classify all data points with the corresponding segment labels

- method is applied to MEG (brain) data
    + used segments of 12.5 seconds or 625 data points

## Representation Learning with Contrastive Predictive Coding
___

- Key insight of the proposed model is that it learns representations by predicting the future latent space with autoregressive methods
- the model is able to learn useful representations in four distinct domains:
    + speech
    + images
    + text 
    + RL in 3D envs
- common strategies for representation learning have been predicting future, missing or contextual information
- in neuroscience, predictive coding theory suggests that the brain predicts observations at various levels of abstraction
- the paper proposes the following:
    + compress high-D data into compact latent embedding
    + use powerful autoregressive models in this latent space
    + rely on Noise Contrastive Estimation for th loss function

![alt text][CPC_Model_1]

[CPC_Model_1]: images/CPC_Model_1.png "CPC"


- use infoNCE loss
    + based on one positive sample and N negative samples
    + categorical cross entropy loss of classifying the positive sample correctly
- Experiments
    + Audio - use a 100 hour LibriSpeech dataset 
        * Used strided convolution network that runs on 16KHz waveform
        * 256 hidden unit GRU
        * use the hidden state of the GRU at each timestep to predict 12 timesteps into the future
        * train on audio samples length 20480 (slightly longer than a second)



## Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models

___

- Paper claims to address the problem of forecasting non-stationary time series for multiple prediction steps
- Distortion Loss including Shape and Time
- Majority of methods use MSE, but has shortcomings
- derive a differentialable loss function for both shape and temporal terms as well as an efficent custom backprop implementation for speeding up optimization
- Training DNN's with Dynamic Time Warping enables focus on the shape error between two signals
- DTW ignores elastic distortions
- Use soft-dtw for the shape term
- Temporal loss is derived from the DTW matrix and computing the Time Distortion Index
    + basically consists in computing the deviation between the optimal DTW path A* and the first diagonal


## Estimating Individual Treatment Effects: Generalized bounds and algorithms
___

- new theoretical analysis on predicting ITE with observational data
- propose new algorithms that acheive SOTA on real and synthetic datasets
- main contribution is to give the first generalization error bound for estimating ITE 
 - from the bounds they derive, they build models that learn outcome functions (m_0, m_1) under a constraint that encourages better generalization across the treated and control populations
 - Builds on pervious paper with balancing neural networks
     + they previously proposed learning a representation then peforming ridge regression on the representation
     + new method overcomes the limitations:
         * linear hypothesis of the representation
         * treatment indicator might get lost if the learned representation is high dimensional
- call the general framework they propose 'counterfactual regression'

![alt text][ite_bounds]

[ite_bounds]: images/ite_bounds.png "ITE"

- TARnet is CFR without weighting their IPM penalty

![alt text][ite_results]

[ite_results]: images/ite_results.png "ITE_res"


## Adapting Neural Networks for the Estimation of Treatment Effects
___

- estimation of treatment effects from observational data usually done in two steps:
    + fit models for expected outcome and treatment propenstiy
    + plug these fitted models into some downstream estimator
- NN's are a good choice for the first step
    + question is, how do we adapt them to improve the quality of the final estimate
- propose new architecture they call dragonnet exploits sufficency of propensity score for estimation adjustment
- use targeted regularization, induces a bias towards models that have asymptotic properties 'out of the box'
- Sufficiency of propensity scores implies that only the information in X that is relevant for estimating the propensity score is necassary for the estimation of the causal effect
- using this, if we build a NN to predict treatment, then remove the last layer, we should be left with represenation that is most relevant for predicting the treatment
- Dragonnet uses this idea in creating a three-headed NN
- they find empircal evidence that the dargonnet trades off prediction quality for a good representation for propensity score

![alt text][dragonnet]

[dragonnet]: images/dragonnet.png "dragonnet"

- Test on ACIC 2018 data and IHDP data
![alt text][dragonnet_results]

[dragonnet_results]: images/dragonnet_results.png "dragonnet_results"

## The M4 Competition: Results, findings, conclusion and way forward 
___
- Three goals of M4 Competition
    + increase number of series
    + include ML methods
    + evaluate both point forecasts and prediction intervals
- Of the 17 most successful methods, 12 were hybrid stats+ML
- Best algo at both point and intervals was hybrid (almost 10% better)
- Second best combined 7 stats models and 1 ML
    + Weights for how to average the stats models calculated by the ML model
- COMB benchmark
    + simple average of three exponential smoothing methods
        * Simple
        * Holt
        * Damped
- Knowledge or prediction intervals is limited
- 95% prediction intervals largely underestimated reality 
    + meaning that only 80% of the data fell within the 95% PI's of a given forecast
- Randomness is the most critical factor determining forecast accuracy
    + followed by linearity
    + seasonal time series tend to be easier to predict
        * they are likely to be less noisy


## Fast ES-RNN 
___
[https://arxiv.org/pdf/1907.03329.pdf]
[https://github.com/damitkwr/ESRNN-GPU]

- time component breaks i.i.d. assumption
- Slawek Smyls is the uber engineer credited with ES-RNN
- This paper ports his c++ code to pytorch
- The model local linear trend component of the ES method is replaced by an RNN
    + The rnn is an LSTM with skip connections to form the dilated LSTM network
        * allows the algorithm to remember information at greater distances
- Training stesp:
    + estimate the level and seasonal coefficents with classical Holt-Winters equations
        * If we have N time series, we have to store N(2+S) parameters, where S is the length of the seasonality
    + to generate the output, pass the output of the LSTM through a non-linear layer with tanh activation and finally through a linear layer
- Notes
    + Syml's implementation uses mulitple seasonality (such as hourly and weekly)
Architecture: 
![alt text][ES_RNN]

[ES_RNN]: images/es_rnn_arch.png "ES_RNN"

## DeepGLO (Think Globally, Act Locally)
___
[https://arxiv.org/pdf/1905.03806.pdf]

- Hybrid model that combines matrix factorization (regularized by a temporal convolution network) and a local temporal network that captures patterns of the individual time series
- Introduce LeveledInit
    + initilization method that removes the need for scaling the data before modellign
- Use TCN-MF 
    + unlike TRMF, TCN-MF can capture non-linear dependancies

## Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
___
[https://arxiv.org/pdf/1912.09363v1.pdf]

- An attention based architecture that combines high performance multi-horizon forecasting with interpretable insights
- Specifically incorporating
    + static covariate encoders which encode context vectors for use in other parts of the network
    + gating mechanism throughout and sample dependant variable selection to minimize the contributions of irrelevant inputs 
    + a sequence to sequence layer to locally process known and observed inputs 
    + a temporal self-attention decoder to learn any long range dependancies present within the dataset
- Intepretability: 
    + globally important variables 
    + persistent temporal patterns 
    + significant events

Overview: 
![alt text][tft_overview]

[tft_overview]: images/tft_overview.png "TFT"
- known inputs include things like day of week or if a promotion will be on
- observed could be other time varying covraiates 

Architecture: 
![alt text][tft_arch]

[tft_arch]: images/tft_arch.png "TFT_arch"

## Detecting and quantifying causal associations in large nonlinear time series datasets
___

- Data driven causal inference in complex dynamical systems (such as Earth systems) is challenging due to high dimensionalality, non-linear and limited sample size
    + Motivating examples:
        * El Nino
        * how are human body processes coupled?
- The method introduced combines linear or non-linear conditional independance tests with a causal discovery algorithm
- The goal in time sereis causal discovery from complex dynamical systems is to estimate causal links including their time lags
- a current major approach in not only Earth data analysis but also neuroscience is to esimate the time-lagged causal associations using autoregressive models in the framework of Granger causality
- They present a causal network discovery method based on the graphical causal model
- Full Coniditional Independance testings' main drawback is that as the dimensionality increases, the detection strenght of causal links decreases
- Causal discovery theory tells us that the parents of a variable are a sufficient conditioning set that alllows establishing conditional independance
    + in contrast to conditioning on the whole past of all processes as in FullCI, conditioning only on a set that at least includes parents of variable X suffices to identify spurious links
    + Markov discovery algorithms (such as the PC algorithm) allows us to detect these parents
        * PC algorithm can be implemented with different kinds of conditional independance tests that can accomodate nonlinear functional dependancies
- Name their method PCMCI, it consists of two stages
    + PC condition selection to find parents for all variables
        * PC algo removes irrelevant parents for each of the N variables by iterative independance testing
            - First iteration run unconditional independance tests and remove all parents for which the null hypothesis cannot be regected 
            - Next iteration, sort the parents by their (absolute) test statistic value the conduct conditional independance tests
                + After each iteration, independant parents are removed
        * Selecting a high significance threshold allows PC to converge and include all parents of each variable as well as some false-positives
        * The MCI deals with the false positives
    + momentary conditional independance tests (MCI)
        * test for conditional dependance for each variable and a potential parent by conditioning on all other potential parents of that variable
- A follow up question is to quantify the causal effect
    + can be done with structural causal models or potential outcomes
- PCMCI is not well suited for highly deterministic systems since it strongly conditions on the past of the driver system and hence removes most of the information that could be measured in the response systen
- Estimate causal effects by running full multivariate regression for each variable using all the parents identified using the PCMCI method
- Numerical experiments show that PCMCI has significantly higher detection power than established methods such as Lasso, the PC algorithm or Granger causality and its nonlinear extensions for time series data sets on the order of dozens to hundreds of variables
- Method focuses on time-lagged dependancies and assumes stationarity of data

## Recommendations as treatments: Debaising Learning and Evaluation
___

- Most data for evaluating and training recommender systems is subject to selection biases, either through self-selection of the use or the system
- provide a method to handle selection bais by adapting models from casual inference
- having observations be conditioned on the effect we would like to optimize leads to data that is missing not at random
- four main contributions
    + evaluation of recommender systems with propensity-weighting - deriving unbaised estimators for a wide range of performance measures
        * in particular the self-normalized inverse propensity scoring is used in calculating accuracy metrics on rating prediction
    + emprical risk minimization framework for learning recommender systems under selection bais
    + ERM framework that allows for matrix factorization method that can account for selection bais
        * using ERM with propensity score adjustment - adding propsensities to the training objective of standard incomplete matrix factorization 
        * the propensities act like weights for each loss term
        * conventional incomplete matrix factorization is a special case of missing completely at random data (ie the propensities are all equal)
    + explore methods to estimate propensities in observational settings where selection bias is due to self-selection
- ML100K dataset provides 100K MNAR ratings for 1683 movies by 944 users (grouplens/movielens)
- collected a new dataset - coat shopping
- MF-IPS performs robustly and efficently on real-world data

## Invariant Risk Minimization
___

- Minimizing training error leads machines into 'recklessly' absorbing all the correlations found in training data
    + a correlation is spurious when we do neot expect it to hold in the future in the same manner it held in the past
- when shuffling data, we destroy information about how the data distribution changes when one varies the data source or collection specifics
- assume the training data is collectin into distinct, seperate enviornments
- the seamless intergration of causation tools into machine learning pipelines remains cumbersome, disallowing what we believe to be a powerful synergy
- propose IRM, a learning paradigm that estimates nonlinear, invariant, causal predictors from multiple training envoirnments to enable out of distribution generalization
- Invariance is a statistically testable quantity that we can measure to discover causation
- Key Idea: to learn invariances across envoirnments, find a data representation such that the optimal classfier on top of that representation matches for all envoirnments
    + uses linear predictor on top of non-linear representation
    + mathmatically, this is a constrained optimization probplem 
    + constraint to balance between predictive power and the invariance of the predictor
    + includes proofs for derivation of constraint 

## Inferring causation from time series in Earth System sciences
___

- in large scale, complex dynamical systems such as the earth systems, real experiments are rarely feasible
- Discuss methods to address four key generic problems
    + causal hypothesis testing
    + causal network analysis
    + exploratory causal driver detection
    + causal evaluation of physical models
- Overview of causal inference methods:
    + Granger Causality 
        * time series based approaches that determines X is GC of Y if removing X from the forecasting of Y increases prediction error
        * Based on linear autoregressive modelling
    - Nonlinear state-space methods
        + Convergent cross mapping
        + For more stochastic series, CCM is less well suited
    - Casual network learning 
        + reconstruction of large scale causal graphical models
        + PC Algorithm
            * Start with a fully connected graph
                - test for the removal of a link between two variables iteratively based on conditioning sets of growing cardinality 
        - PC MCI addresses challanges of autocorrelated high-dimensional and nonlinear time series 
            + PC -> condition selection step
            + MCI (momentary conditional independance) 

Summary:
![alt text] [ci_methods]

- Key task in earth systems sciences is to evaluate which model better simulates the real system
- Some key challanges: 
Architecture: 
![alt text][earth_cause]

[earth_cause]: images/earth_causation_challanges.png "Challanges"
[ci_methods]: images/CI_methods.png "methods"

## Can we learn individual-level treatment policies from clinical data?
___ 

- two main challanges
    + Causal one - unmeasured confounding making it hard to measure causal effects
    + Statistical one - estimate effect in noisy data dominated by other influences
- a host of work has shown how experimental data can be analyzed to learn individualized treatment rules
- sample size required to learn ITE is much larger than what is required to learn ATE
- use of observational data will be key
    + need better methods to take into account unmeasured confounding
- New methods for jointly analyzing experimental and observational data
- new designs of trials also will be beneficial

## Recurrent NN for Multivariate Time Series with Missing Values
___

- missing values and their missing patterns are often correlated with target labels
    + called 'informative missingness'
- paper states there have not been RNN structures incorporating the patterns of missingness for time series classifcation problems 
- develop RNN called GRU-D to exploit two representations of informative missingness patterns
    + masking - informs the model which inputs are missing
    + time-interval - encapsulates the input observation patterns
- show that their model outperforms GRU baselines that use imputation
- GRU-D -> the D stands for decay, as a temporal variable is usually only useful (in medical contexts) for a certain time period after its observation
    + Decay mechanism is used for input variables and hidden states
- PhysioNet Challange 2012 dataset
    + 8000 ICU visits each with 33 variables
- MIMC-III 
    + 58,000 hospital admission records 
    + 99 time series from 19,714 visits 


## RETAIN - An interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism
___

- Accurate and interpretable model for EHR data
- attends to EHR data in reverse time order
- applied on dataset with 14M visits completed by 263K patients
- data is comprised of sequences of patient visits
    + each visit is comprised of a varying number of medical codes
    + in disease progression modelling the goal is to predict the codes occuring at the next visit 
- uses two sets of weights, one for visit level attention, and one for variable level attention
    + alphas for visit level, betas for variable level
    + use two RNNs to generate alphas and betas seperatly

![alt_text][retain_arch]
[retain_arch]: images/retain_arch.png "retain"

- Overall, the RETAIN attention mechanism can be viewed as the inverted architecture of the standard attention mechanism for NLP
    + MLP to embed visit information and use RNN to generate attention weights
- Finding the visits that contribute the most to prediction are derived by finding the largest alpha
    - finding influential variables more complicated
    - propose a method that assumes the x_{i,j} that yeilds the biggest change in y will be the input variable with highest contribution
- apply to predicting heart failure 

## Attentive State-Space Modeling of Disease Progression
___

- Models of disease progression (DPM) are key for two main reasons
    + Predicting patient outcomes
    + understanding disease dynamics
- present a probabalistic model that learns an accurate and interpretable structured representation for disease trajectories
- unlikey markovian models - this one uses attention to creat memoryful dynamics
- use data from UK cystic fibrosis registry
- model uses a state-space representation to segement a disease into stages of progression
- learns disease states in unsupervised fashion
- implement dynamic attention mechanism with seq to seq RNN
- since model is non-markovian, inference of posterior disease states is intractable and cannot be conducted using standard forward-backward routines
    + use a structured inference network trained to predict posterior state distributions by mimicking the attentive structure of the model
- the model applies attetnio to the latent state-space
    + in RETAIN it is applied to the sample space
    + in this model attention interprets disease dynamics
- HMM are traditionally used for DPM problems
- attention weights in the model determine the influence of past state realizations on future state transitions via linear dynamics

-architecture:

![a_ss_arch] [a_ss_arch]
[a_ss_arch]: images/a_ss_arch.png "a_ss_arch"

- use a variational learning algorithm that jointly learns model parameters and a structured inference network that approximates the posterior p(z|x)

-the inference network is below:

![a_ss_inference_network] [a_ss_inference_network]
[a_ss_inference_network]: images/a_ss_inference_network.png "a_ss_inference_network"

- training algorithm:
    + Sample (z_1, ..., z_t) ~ q(z_T, x_T) for i = 1,..., N
    + estimate ELBO L(x_T, z_1, ..., z_t)
    + gradients theta and phi
    + update two model parameters
- model effectively models disease state transitions
- also effectively models 1-year risk of comordbitities more accurately than retain and other baselines


### List of papers read:
- <b> Detecting causal associations in large nonlinear time series datasets </b>
- <b> Learning Representations for Counterfactual Inference </b>
- <b> ARMA Time Series Modeling with Graphical Models </b>
- <b> Granger Causal Structure Reconstruction from hetrogreneous multivariate timeseries  </b>
- <b> Deep AR </b>
- <b> Learning neural causal models from unknown interventions </b>
- <b> WaveNet: A generative Model for Raw Audio </b>
- <b> Forecasting at scale (Prophet) </b>
- <b> Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning </b>
- <b> Causal Discovery and Forecasting in Nonstatinoary Envoirnments with State-Space Models </b>
- <b> Consistent individualized feature attribution for tree ensembles </b>
- <b> The mythos of model interpretability </b>
- <b> A unified approach to interpreting model predictions </b>
- <b> Why should I trust you: explaining the predictions of any classifier </b>
- <b> Machine Learning: The high-interest credit card of technical debt </b>
- <b> Estimation of causal effects with multiple treatments: a review and new ideas </b>
- <b> Estimating Heterogeneous treatment effects with obeservational data </b>
- <b> Estimation and inference of heterogeneous treatment effects using random forests </b>
- <b> N-BEATS: Neural basis expansion analysis for interpretable time series forecasting </b>
- <b> Automated versus do-it-yourself methods for causal inference: Lessons learned from a data analysis competition </b>
- *Improve user retention with Causal Learning*
- *Learning connections in financial time series*
- *Temporal difference variational autoencoder*
- *Macroeconomic forecasting for Australia using a large number of predictors*
-  *Off to the races: a comparison of machine learning and alternative data for predicting economic indicators*
-  *Nowcasting of the local economy: using Yelp data to measure economic activity*
-  *Robustly Disentangled Causal Mechanisms - Validating Deep Representations for Interventional Robustness*
-  *Efficent and robust approximate nearest neighbor search using Hierachical Navigable Small World graphs*
-  *Latent Translation: crossing modalities by bridging generative models*
-  *Diagnosing and enhancing VAE models*
-  *Casual embeddings for Reccomendation*
-  *Discrete Autoencoders for sequence models*
-  *Limits of estimating heterogeneous treatment effects: guidelines for practical algorithm design*
-  *Some methods for heterogenous treatment effect estimation in high-dimesions*
-  *Neural representation of sketch drawings*
-  Real-time personalization using embeddings for search ranking at Airbnb
-  Swoosh: a generic approach to entity resolution
-  Dynamic churn prediction framework with more effective use of rare event data: the case of private banking
-  A nonparametric Bayesian analysis of hetrogeneous treatment effects in digital experimentation
-  Multi-Relational Record Linkage 
-  A join Model of Usage and Churn in Contractual Settings
-  Dynamic allocation of pharmaceutical detailing and sampling for long-term profitability
-  New statistical learning methods for estimation optimal dynamic treatment regimes
-  A compairson of blocking methods for record linkage
-  Anchors: High-Precision Model Agnostic Explanations
-  Deep learning based recommender systems: A survey and new perspectives
-  The promise and peril of human evaluation for model interpretability
-  The intriguing properties of model explanations
-  Patient2Vec: A personalized interpretable deep representation of the longitudinal electronic health record
-  Multi-Task Learning for Email Search Ranking with Auxiliary Query Clustering
-  Towards a rigorous science of interpretable Machine Learning
-  Challanges for transparency
-  A second chance to get causal inference right: A classification of data science tasks
-  Wasserstein dependancy measure for representation learning
-  An engagement based customer lifetime value system for ecommerce
-  Two decades of recommender systems at Amazon.com
-  Data efficent hierarchical reinforcement learning
-  Interpretability beyond feature attribution: Quantitative Testing with Concept Activation Vectors
-  Cross domain regularization for neural ranking models using adversarial learning
- Challanging common assumptions in the unsupervised learning of disentangled representations
- Estimating Individualized Treatment Rules Using Outcome weighted learning
- How to fine tune BERT for Text Classification
- The adaptive markets hypothesis: Market Efficiency from an evolutionary perspective
- Proximal Policy Optimization Algorithms
- Using text embeddings for causal inference
- Spreading vectors for similarity search
- Applying deep learning to Airbnb search
- 
-  WAE GAN
- Conditional time series forecasting with convolution neural networks

## To Read
- The deconfounded recommender: a causal inference approach to reccomendation
- Disentangling State Space Representations

















