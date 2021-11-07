# -*- coding: utf-8 -*-
"""
Examples for image warping with optimal transport
"""
import logging

import numpy as np 
from numpy.matlib import repmat
import PIL.Image as Image
from skimage.color import rgb2gray

from ot import imgwarp

imgwarp.logger.setLevel(logging.INFO)


def main():

    ##### Example 1: Uniform to "Lena"
    path_out = "./Images/Lena.jpg"
    img = Image.open(path_out)
    img = img.resize((100,100),Image.BILINEAR)

    img_array = np.array(img)
    density_out = rgb2gray(img_array)  

    density_in= np.ones(density_out.shape)

    pmae = imgwarp.SolveParabolicMAE(density_in, density_out, absTol = 10, 
                                     step_size= 0.00005, smooth_map = True, 
                                     smooth_kind = "interp")
   
    pmae.plot_densities()
    pmae.plot_map(step=1)
    pmae.plot_density_ratio()
    
    ##### Example 2: separable densities
    x_in = np.linspace(-1, 1, 300)
    y_in = np.linspace(-1, 1, 160)
    Y_in, X_in = np.meshgrid(x_in,y_in)

    density_in = np.exp(X_in)*np.exp(Y_in)
    
    x_out = np.linspace(-1, 1, 200)
    out_x = np.sin(x_out*2*np.pi) + 2
    
    density_out = repmat(out_x, 400, 1)
    

    sepd = imgwarp.SeparableDensities(density_in, density_out)
  
    sepd.plot_densities()
    sepd.plot_map(step = 2)
    sepd.plot_density_ratio()
    
    return


if __name__ == "__main__":
    
    main()