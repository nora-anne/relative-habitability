# Relative Habitability Project
Some code, figures, and information behind the paper ["Relative Habitability of Exoplanet Systems with Two Giant Planets"](https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.4765B/abstract).

# Introduction

The primary intent of this project was to use a large dataset of simulation results to investigate how the orbits of giant planets within a planetary system affect the potential habitability of an Earth-like companion planet. In conducting this research, I completed close to 12 million individual simulations to create a dataset spanning 10 dimensions. The simulations were run on the high performance computing cluster at the University of Chicago. The entire dataset is [published and available for use](https://zenodo.org/record/6324216#.ZF27pXbMJPY). Here, I document a few examples of the work I completed during this project. For a more comprehensive description and results, please see the paper.

# Understanding the Dataset

The zipped archive of NumPy arrays `param_m2_single39.npz` contains the results of a single set of simulations. For a single set of simulations, there are 2 giant planets with fixed properties and an Earthlike planet whose location changes in each of 80 iterations. This is an example of a single set; in all, there were 148,510 files of this nature. The arrays included are:
- `outcomes` which documents the outcome of the simulation; this is a prediction of the longterm stability of the system, where 0 indicates an instability occurred during the simulation
- `outcome_codes` which uses a single character to document the reason that a system went unstable during the simulation for all "0" outcomes
- `max_eccs_E` which documents the maximum eccentricity of the Earthlike planet during the duration of the simulation

The code file `plotting_single_relative_habitability.py` loads and analyzes a single set of simulations. In this case, it is set up for `param_m2_single39.npz`. Evaluating a single instance in this manner helps understand the data products to refine the analysis process so that it could be implemented in an automated manner for the entire dataset. It is also helpful for diving in to understand the results of the automated process for any given set of simulations.

First, the outcomes are plotted against the location of the Earthlike planet for the set of simulations. The outcome codes are used to shade the plot in different colors for each of the reasons for instability, which helps to understand the physical mechanisms at work.

Next, the habitability of the Earthlike planet in each iteration is estimated. The habitability is taken from a model created for a single Earthlike planet and adjusted based on the maximum eccentricity of the Earthlike planet. This relative habitability curve is then plotted along with the basic model of habitability used to help compare the differences.

Finally, the eccentricity-adjusted habitability and the stability outcomes are combined, so that unstable systems have no habitability and systems that are more likely to become unstable have lower habitability. Using a spline, the area under this adjusted habitability curve is integrated. Comparing the total habitability of this configuration of giant planets to the total habitability of the basic model of habitability gives a relative habitability for a potential Earthlike planet in a system with this configuration of giant planets. This single value is the datapoint used for further analysis of the entire dataset.

The figure `example_plot_single.png` shows the figure created from this analysis.

# Visualization

Given such a large and multidimensional dataset, a key part of my work was visualizing this data in a way that was understandable without being so simplified as to lose the relevant details.

# Open Access Data

For purposes of validation and continued research, it was important to publish the entire dataset in a machine readable format for anyone else to make use of.
