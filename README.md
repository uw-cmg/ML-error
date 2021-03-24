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
