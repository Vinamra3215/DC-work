import pywt
import numpy as np

def dwt2(img, wavelet="haar"):
    LL, (LH, HL, HH) = pywt.dwt2(img, wavelet)
    return LL, LH, HL, HH

def idwt2(LL, LH, HL, HH, wavelet="haar"):
    return pywt.idwt2((LL, (LH, HL, HH)), wavelet)
