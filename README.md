# pois_comparisons
The MATLAB code for comparing the POIS model with others

This code is based on the original code by Inouye et. al. avaiable at ... .

To run the code first install the POIS package from here and the XRMF package from CRAN. Then run `new_demo.m`

The code relies heavily on the original code, but the organization is different and new functionality has been implemented. 
We borrow the implemntation of PoiMix, CopulaPoi, IndNegBin, T-PGM and PoiSQR from the original code. We add POIS, Bootsrap, CopulaMult, IndMult. 
Current code is not backward compatibile with Inouye et. al.'s, but there is enough similairy that you can hopefully port other methods that they implement to our code. 




