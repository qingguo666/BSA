# Association-based Optimal Subpopulation Selection forMultivariate Data

This is the official code repository for the paper `Association-based Optimal Subpopulation Selection for Multivariate Data`.

In  the  analysis  of  multivariate  data,  a  useful  problem  is  to  identify  a  subset  ofobservations for which the variables are strongly associated.  One example is in driv-ing safety analytics, where we may wish to identify a subset of drivers with a strongassociation  among  their  driving  behavior  characteristics.   Other  interesting  domainsinclude  finance,  healthcare,  marketing,  etc.   Existing  approaches,  such  as  the  Top-kmethod or the tau-path approach primarily relate to bivariate data and/or invoke thenormality  assumption.   Directly  adapting  these  methods  to  the  multivariate  frame-work is cumbersome.  In this work, we propose a semiparametric statistical approachfor  the  optimal  selection  of  subpopulations  based  on  the  patterns  of  associations  inmultivariate data.  The proposed method leverages the concept of general correlationcoefficients to enable the optimal selection of subpopulation for a variety of associationpatterns.   We  develop  efficient  algorithms  consisting  of  sequential  inclusion  of  casesinto the subpopulation.  We illustrate the performance of the proposed method usingsimulated data and interesting real data.

You can clone this repository by running:

```
git clone https://github.com/qingguo666/FLO
```
## Author

* Qing Guo Department of Statistics, Virginia Tech, e-mail: qguo0701@vt.edu
* Xinwei DengDepartment of Statistics, Virginia Tech, e-mail: xdeng@vt.edu
* Nalini RavishankerDepartment of Statistics, University of Connecticut, e-mail: nalini.ravishanker@uconn.edu (correspondence)

## Citation

If you reference or use our method, code or results in your work, please consider citing the paper:

```
To be updated
```

## Contents

This repository contains the following contents. 

#### - Jupyter notebooks
Jupter notebook examples of our FLO model and various baselines (Nguyen-Wainwright-Jordan (NWJ), Donsker-Varadhan (DV), Barber-Agakov (BA), Contrastive Predivtive Coding (CPC/InfoNCE), etc.). 

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
