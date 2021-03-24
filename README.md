# ML-Error
Code to generate plots for research on scaling error bars from machine-learning models.

# Instructions for use:
To recreate the figures from the paper "Calibrated Bootstrap for Uncertainty Quantification in Regression Models", use the format specified below.  

```
usage: <python> main.py <action> <model> <dataset> <optional_save_path>

<python> is your python command.
<action> is one of the available actions:
         - run
         - plot
<model>  is one of the available models:
         - RF
         - LR
         - GPR
         - GPR_Bayesian
<datset> is one of the available datasets:
         - Diffusion
         - Perovskite
         - Friedman
         - Friedman_0.1_Noise
         - Friedman_0.2_Noise
         - Friedman_0.3_Noise
         - Friedman_0.4_Noise
         - Friedman_0.5_Noise
         - Friedman_1.0_Noise
         - Friedman_2.0_Noise
<optional_save_path> is an optional location to save the resulting data and plots.
         If no path is given here, data and plots will be saved in the current directory.
```

Example use:

```
python3 main.py plot GPR Diffusion
```
In the example above, r-statistic and RMS residual vs. uncertainty estimate plots are made from the residuals, uncertainty estimates, and calibration factors found for Gaussian process regression on the Diffusion data set. These plots should match the corresponding plots from the paper.

```
python3 main.py run RF Friedman
```
In the example above, 5-fold cross-validation splits are randomly generated for the synthetic dataset used for the paper to obtain residuals, uncertainty estimates, and calibration factors for predictions by a random forest model. These are then used to make r-statistic and RMS residual vs. uncertainty estimate plots. These plots will likely differ slightly from the ones in the paper due to random variation in the cross-validation.
