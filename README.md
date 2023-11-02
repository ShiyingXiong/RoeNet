# RoeNets: Predicting Discontinuity of Hyperbolic Systems from Continuous Data
* Paper: https://arxiv.org/abs/2006.04180
## Summary
We introduce Roe Neural Networks (RoeNets) that can predict the discontinuity of the hyperbolic conservation laws (HCLs) based on short-term discontinuous and even continuous training data. Our methodology is inspired by Roe approximate Riemann solver (P. L. Roe, J. Comput. Phys., vol. 43, 1981, pp. 357--372), which is one of the most fundamental HCLs numerical solvers. In order to accurately solve the HCLs, Roe argues the need to construct a Roe matrix that fulfills "Property U", including diagonalizable with real eigenvalues, consistent with the exact Jacobian, and preserving conserved quantities. However, the construction of such matrix cannot be achieved by any general numerical method. Our model made a breakthrough improvement in solving the HCLs by applying Roe solver under a neural network perspective. To enhance the expressiveness of our model, we incorporate pseudoinverses into a novel context to enable a hidden dimension so that we are flexible with the number of parameters. The ability of our model to predict long-term discontinuity from a short window of continuous training data is in general considered impossible using traditional machine learning approaches. We demonstrate that our model can generate highly accurate predictions of evolution of convection without dissipation and the discontinuity of hyperbolic systems from smooth training data.

![](https://github.com/ShiyingXiong/RoeNet/blob/main/Figure/Roenet1.png)

![](https://github.com/ShiyingXiong/RoeNet/blob/main/Figure/Roenet2.png)


## Usage
We demonstrated the efficacy of our Roenet in predicting linear and nonlinear, one-dimensional and multidimensional, one-component and multi-component equations.
In order to train a Roenet:
* 1C Linear: `python3 trivial_1c/train_net.py`
* 3C Linear: `python3 trivial_3c/train_net.py`
* Burgers: `python3 nontrivial_1c/train_net.py`
* Sod Tube: `python3 nontrivial_3c/train_net.py`
* 2D Linear/Nonlinear: `python3 2d_wave/train_net.py`
* Ablation Test on Computational Cost: `python3 trivial_1c_grid`

Please note: Each experimental folder is designed to function independently. Every folder contains a file for training and a file for data generation. Since the file structure is similar across folders, detailed commentary has been provided in the trivial_1c folder. This should help users understand the file functionalities, which are largely consistent throughout. The differences in the codes for each experiment primarily lie in the formulas used to generate data and the specific configurations of the neural network matrices.

## Problem setups

![](https://github.com/ShiyingXiong/RoeNet/blob/main/Figure/Setup.png)



## Results
Results can be found in Paper: https://arxiv.org/abs/2006.04180.

## Dependencies
* PyTorch
* NumPy
* h5py
* Matplotlib
