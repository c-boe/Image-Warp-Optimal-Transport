# Optimal transport for image warping

Implemenation of optimal mass transport algorithms for image warping.


## General information

Currently implemented:
* [An efficient approach for the numerical solution of Monge-Ampère equation by Sulman, et al.](https://www.sciencedirect.com/science/article/abs/pii/S0168927410001819) for square densities
* 1D integration method for separable distributions


## Installation

Install packages in `requirements.txt` by

```
pip install -r requirements.txt
```

## Examples

Execute `examples.py` to redistribute a uniform density into a grayscale "Lena" density:

![N|Solid](/Images/Lena.jpg)

The resulting optimal transport map represented by a meshgrid:

![N|Solid](/Images/Lena_map.png)



