"""
Final coursework project in Numerical Recipes course, taken during fall semester 2018, at University of Edinburgh

Aleksandra Sobieska

Data generation using a 2-dimensional probability density function (PDF) for
radioactive decay and parameter fitting with maximum likelihood.

This programme randomly generates data for a probability density function (PDF) representing a radioactive decay
using box method and plots a histogram of the whole distribution or just its components. It also performs
a maximum likelihood fit on the PDF parameters for the complete data set with both time and angle data
and its partial version with time data only available. In addition, it calculates simple and proper errors
for the best-fit values and plots NLL (the negative of the log of the joint likelihood) against a parameter.
"""
from NRclasses import Generate, ML

def main():

    #Part 1 - Data generation
    n_events = int(input("For how many events? "))
    #specify if data for the whole PDF or its distribution component have to be generated
    mode = int(input("Do you want to generate random events for PDF (0), P1 (1) or P2 (2)? "))
    distr = Generate(n_events, mode) #create initial settings
    theta, time = distr.generate_decay() #generate data

    #plot angle and time distributions
    distr.histo_plotting(theta, "theta [radians]", "number of occurences", "theta distribution")
    distr.histo_plotting(time, "time [microseconds]", "number of occurences", "time distribution")

    #Part 2 - Comparing parameters fitting using only time data to using both time and angle data
    data_type = int(input("For time only data (1) or both angle and time data (2)? "))
    ml = ML(distr, n_events, mode, data_type) #create initial settings
    ml.data2array() #convert provided time and angle data in a text file to an array
    ml.minimise() #find best-fit values for parameters using maximum likelihood
    ml.findPamError() #determine simple errors on the best-fit values

    #Part 3 - Calculating proper errors on fitted parameters
    ml.properError() #determine proper errors on the best-fit values

if __name__ == "__main__":
    main()
