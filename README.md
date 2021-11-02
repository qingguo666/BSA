# Association-based Optimal Subpopulation Selection forMultivariate Data

This is the official code repository for the paper `Association-based Optimal Subpopulation Selection for Multivariate Data`.

In  the  analysis  of  multivariate  data,  a  useful  problem  is  to  identify  a  subset  of observations for which the variables are strongly associated.  One example is in driving safety analytics, where we may wish to identify a subset of drivers with a strong association  among  their  driving  behavior  characteristics.   Other  interesting  domains include  finance,  healthcare,  marketing,  etc.   Existing  approaches,  such  as  the  Top-k method or the tau-path approach primarily relate to bivariate data and/or invoke the normality  assumption.   Directly  adapting  these  methods  to  the  multivariate  framework is cumbersome.  In this work, we propose a semiparametric statistical approach for  the  optimal  selection  of  subpopulations  based  on  the  patterns  of  associations  in  multivariate data.  The proposed method leverages the concept of general correlation coefficients to enable the optimal selection of subpopulation for a variety of association patterns.   We  develop  efficient  algorithms  consisting  of  sequential  inclusion  of  cases into the subpopulation.  We illustrate the performance of the proposed method using simulated data and an interesting real data.

You can clone this repository by running:

```
git clone https://github.com/qingguo666/FastBSA
```
## Author

* Qing Guo, Department of Statistics, Virginia Tech, e-mail: qguo0701@vt.edu
* Xinwei Deng, Department of Statistics, Virginia Tech, e-mail: xdeng@vt.edu
* Nalini Ravishanker, Department of Statistics, University of Connecticut, e-mail: nalini.ravishanker@uconn.edu (correspondence)

## Citation

If you reference or use our method, code or results in your work, please consider citing the paper:

```
To be updated
```

## Contents

This repository contains the following contents. 

## Related Resources

* code for BSA method https://github.com/acaloiaro/topk-taupath/tree/master/R

#### - Experiment codes
Python codes used for our experiments. 

#### - Results and visualization
Python codes used for the visualization of our results. 

## Prerequisites

The algorithm is built with:

* Python (version 3.7 or higher)


## Installing third-party packages
To be updated...

## Datasets
COVID-19 data in VA
