# AvaSpec Python

This is a Python wrapper for the Avantes AvaSpec spectrometer library. 

It can be used to connect to Avantes spectrometers via Python. The functions of this wrapper include live and single measurements of a spectrum, measurement of a dark spectrum to calibrate the spectrometer to its light surroundings, and a least-squares fitting algorithm to fit a Planck curve to the recorded spectrum, if possible.

# Important

**Note:** This wrapper does not correct the spectrum. Instead, it measures the raw spectrum directly from the spectrometer.

