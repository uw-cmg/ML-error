# ML-Error
Code to generate plots and/or re-run calculations from the paper "Calibrated Bootstrap for Uncertainty Quantification in Regression Models".

# Instructions for use:
Use the command line format specified below. Code should be run with Python 3. 

Note that the 'plot' command simply replots the figures from the paper using computations that have already been done, while the 'run' command re-runs our cross-validation method with random splits. Accordingly, the 'run' command may take (substantially) longer to finish, and will result in plots that may differ slightly from the ones in the paper.

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
         - GPR_Both
<dataset> is one of the available datasets:
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

## Note on different GPR models
Note the three different options for running GPR: ```GPR```, ```GPR_Bayesian```, and ```GPR_Both```. Each of these performs a different action. 
* The ```GPR``` model makes predictions with a bootstrap ensemble of GPR models, with UQ given by the standard deviation of the ensemble predictions.
* The ```GPR_Bayesian``` model makes predictions with a single GPR model, with standard Bayesian UQ as implemented in sklearn.
* The ```GPR_Both``` model makes predictions with a single GPR model, but uses two methods of UQ so the user can compare them. The first is the standard Bayesian error bars, as in ```GPR_Bayesian``` above. The second is by fitting a bootstrap ensemble of GPR models and finding the standard deviation of those predictions, as in ```GPR``` above (except that here, the ensemble is not used to make the actual predictions, only for UQ). (See Figure 3 in the paper.)
