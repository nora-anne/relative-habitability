# Relative Habitability Project
Some code, figures, and information behind the paper ["Relative Habitability of Exoplanet Systems with Two Giant Planets"](https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.4765B/abstract).

# Introduction

The primary intent of this project was to use a large dataset of simulation results to investigate how the orbits of giant planets within a planetary system affect the potential habitability of an Earth-like companion planet. In conducting this research, I completed close to 12 million individual simulations to create a dataset spanning 10 dimensions. The simulations were run on the high performance computing cluster at the University of Chicago. The entire dataset is [published and available for use](https://zenodo.org/record/6324216#.ZF27pXbMJPY). Here, I document a few examples of the work I completed during this project. For a more comprehensive description and results, please see the paper.

# Understanding the Dataset

The zipped archive of NumPy arrays `param_m2_single39.npz` contains the results of a single set of simulations. For a single set of simulations, there are 2 giant planets with fixed properties and an Earthlike planet whose location changes in each of 80 iterations. This is an example of a single set; in all, there were 148,510 files of this nature.

The code file `plotting_single_relative_habitability.py` loads and analyzes a single set of simulations. In this case, it is set up for `param_m2_single39.npz`. Evaluating a single instance in this manner helps understand the data products to refine the analysis process so that it could be implemented in an automated manner for the entire dataset. It is also helpful for diving in to understand the results of the automated process for any given set of simulations.

The figure [`example_plot_single.png`](example_plot_single.png) shows the figure created from this analysis. The result of the analysis is a single value, the relative habitability. For the entire dataset, the relative habitabilities are collected and saved in the NumPy array `rel_hab_all.npy`.

# Visualization

Given such a large and multidimensional dataset, a key part of my work was visualizing this data in a way that was understandable without being so simplified as to lose the relevant details.

The initial analysis uses a fiducial case and varies the parameters one at a time, lending itself to a simple visualization of the relative habitability versus each parameter. The value of each parameter for the fiducial case is shown with a vertical line. The file `plotting_relative_habitability_all_inner.py` loads and analyzes each output file for the fiducial systems, calculating the relative habitability, and then plots and labels the results. To improve the efficiency of the plot, similar parameters are combined within a single plot. In addition, the final plot contains various features to aid in understanding, including shaded areas to indicate the locations of the giant planets within the habitable zone and various key resonances of the giant planets. The figure `fiducial_inner_results.png` shows the outcome of this script.

The larger analysis is the full dataset where the parameters vary along 8 dimensions in every possible combination. The resulting dataset is much more difficult to visualize. The final method decided on was the use of double-sided violin plots to illustrate the distribution of outcomes, as depicted in figure [`1D_violin_plot.png`](1D_violin_plot.png). The relevant code is included in `plotting_relative_habitability_all_1D_splitviolins.py`. A lot of customization of the Matplotlib violin plot function was required to obtain a readable result, including normalizing the violin plot widths based on a kernel density estimation for each of the distributions (this was saved as `violin_plot_width.npy` to speed up trial and error of the plotting results).

There are two dimensions with only two values, and those are used to split the results into 4 groups, indicated in the plot by color and by left/right side of the distribution. By making the colors slightly transparent and using complementary colors, it is easy to see where the plots overlap (e.g., red + blue = purple). The remaining 6 dimensions are then flattened into the vertical distribution. For example, the red violin plot at 0.1 for m1 indicates the distribution of all the relative habitability results where m1 = 0.1 from the subset of results where the giant planets are coplanar and aligned (red color, left side). This method of visualization allowed for key trends and outcomes to be noticed across the entire dimensional space and highlighted the small impact of the two colored dimensions, as the violin plots appear to overlap almost exactly.

Because the results of the violin plots so closely aligned for the inclination and pericenter parameters, I used an additional visualization to exploit that similarity to further explore and understand the results. By comparing the results between, for example, the coplanar and inclined case and plotting that difference in a two-dimensional heatmap for each pair of additional parameters (a corner plot), the areas where the inclination does have a significant effect are immediately obvious and allowed for further investigation of the relevant results. The code for this is included in `plotting_difference_2D_coplanar_vs_inclined.py` and the resulting figure is [`2D_diff_inclination.png`](2D_diff_inclination.png).

# Open Access Data

For purposes of validation and continued research, it was important to publish the entire dataset in a machine readable format for anyone else to make use of.
