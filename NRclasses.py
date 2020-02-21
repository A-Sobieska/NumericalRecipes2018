"""
Classes utilised in NRmain.py

There are two classes: one for data generation following the radioactive decay PDF
and the other one is for parameter fitting using maximum likelihood.
"""
import numpy as np
import random
from scipy import integrate
import pylab as pl
import scipy.optimize as opt
import matplotlib.pyplot as plt

class Generate(object): #generates data following the radioactive decay PDF
    def __init__(self, n_events, mode): #constructor
        #decay time range
        self.t_bounds = [0, 10]
        #angular limit intervals
        self.theta_bounds = [0, 2*np.pi]
        #number of random events
        self.n_events = n_events
        #specifies if data for the whole PDF or its compononent have to be generated
        self.mode = mode

    def N(self, tau): #normalisation function for the PDF
        return (3*np.pi*tau*(1 - np.exp(-10/tau)))

    def P1(self, theta, t, tau): #the first component of PDF
        return ((1+(np.cos(theta)**2))*np.exp(-t/tau))/self.N(tau)

    def P2(self, theta, t, tau): #the second component of PDF
        return (3*((np.sin(theta))**2)*np.exp(-t/tau))/self.N(tau)

    def PDF(self, theta, t): #probability density function of radioactive decay with F = 0.5
        return (0.5*self.P1(theta, t, 1) + 0.5*self.P2(theta, t, 2))

    #generating arbitrary decay distribution using box method
    def generate_decay(self):
        #intiailising arrays for angle theta and time t distributions
        self.theta_distr = np.zeros(self.n_events)
        self.t_distr = np.zeros(self.n_events)

        k = 0 #set counter for how many random events were generated
        y_max = 1.1 #value slightly higher than the maximum y-value of PDF
        while k < self.n_events: #loop until all events are collected

            #randomly generate time and angle values within their intervals
            t1 = self.t_bounds[0] + (self.t_bounds[1] - self.t_bounds[0])*np.random.uniform()
            theta1 = self.theta_bounds[0] + (self.theta_bounds[1] - self.theta_bounds[0])*np.random.uniform()
            #find probability for these time and angle values, which creates a test boundary to check
            #if randomly generated (withing the range) y2 falls within the distribution
            if self.mode == 0:
                y1 = self.PDF(theta1, t1)
            elif self.mode == 1:
                y1 = self.P1(theta1, t1, 1)
            elif self.mode == 2:
                y1 = self.P2(theta1, t1, 2)

            y2 = y_max*np.random.uniform()

            #if y2 falls within the distribution, include the event in the data lists
            #if not, discard it and look again
            if y2 < y1:
                self.theta_distr[k] = theta1
                self.t_distr[k] = t1
                k += 1

        return self.theta_distr, self.t_distr

    def histo_plotting(self, array, x_label, y_label, title): #plot a histogram of the distribution
        plt.rcParams.update({'font.size': 15}) #increase font size of all the labels
        pl.hist(array, bins=50)
        pl.xlabel(x_label)
        pl.ylabel(y_label)
        pl.title(title)
        pl.show()

#finds best-fit values and simple/proper erros for parameters using maximum likelihood
class ML(object):
    def __init__(self, gen, n_events, function, data_type): #constructor
        self.gen = Generate(n_events, function) #allows to use methods from the other class
        #initial guess of parameter values for the minimiser
        self.param_guess = np.array([0.5, 1, 2]) #(F, tau1, tau2)
        #angular and time limit intervals taken from the previous class
        self.theta_bounds = gen.theta_bounds
        self.t_bounds = gen.t_bounds
        #parameter bounds for the minimiser
        self.pam_bounds = [(0, 0.9999), (0.05, None), (0.05, None)] #[F, tau1, tau2]
        self.mode = data_type

    #probability density function of radioactive decay for both time and angle data
    def PDF(self, theta, t, F, tau1, tau2):
        return (F*self.gen.P1(theta, t, tau1) + (1 - F)*self.gen.P2(theta, t, tau2))

    def data2array(self):
        #convert provided data to an array matrix with floats and transpose,
        #so that the columns (time, theta) are rows
        self.t_and_theta = np.loadtxt("decay.txt", dtype = float, unpack = True)

    def marginal_P1(self, t, tau): #marginalised first component of PDF for time only data
        return (3*np.pi*np.exp(-t/tau))/self.gen.N(tau)

    def marginal_P2(self, t, tau): #marginalised second component of PDF for time only data
        return (3*np.pi*np.exp(-t/tau))/self.gen.N(tau)

    def marginal_PDF(self, t, F, tau1, tau2): #marginalised PDF for time only data
        return (F*self.marginal_P1(t, tau1) + (1 - F)*self.marginal_P2(t, tau2))

    def NLL(self, *params): #the negative of the log of the joint likelihood
        #open the tuple and assign its elements to specific parameters
        F, tau1, tau2 = params[0]
        if self.mode == 1: #for time only data
            decay = self.marginal_PDF(self.t_and_theta[0], F, tau1, tau2) #decay probability density function
        else: #for both time and angle data
            decay = self.PDF(self.t_and_theta[1], self.t_and_theta[0], F, tau1, tau2)

        sum = -np.sum(np.log(decay)) #calculate NLL
        return sum

    def NLL_F(self, *params): #NLL for finding a proper error on F
        tau1, tau2 = params[0]
        if self.mode == 1: #for time only data
            decay = self.marginal_PDF(self.t_and_theta[0], self.F_pam, tau1, tau2) #decay probability density function
        else: #for both time and angle data
            decay = self.PDF(self.t_and_theta[1], self.t_and_theta[0], self.F_pam, tau1, tau2)
        sum = -np.sum(np.log(decay))
        return sum

    def NLL_tau1(self, *params): #NLL for finding a proper error on tau1
        F, tau2 = params[0]
        if self.mode == 1: #for time only data
            decay = self.marginal_PDF(self.t_and_theta[0], F, self.tau1_pam, tau2) #decay probability density function
        else: #for both time and angle data
            decay = self.PDF(self.t_and_theta[1], self.t_and_theta[0], F, self.tau1_pam, tau2)

        sum = -np.sum(np.log(decay))
        return sum

    def NLL_tau2(self, *params): #NLL for finding a proper error on tau2
        F, tau1 = params[0]

        if self.mode == 1: #for time only data
            decay = self.marginal_PDF(self.t_and_theta[0], F, tau1, self.tau2_pam) #decay probability density function
        else: #for both time and angle data
            decay = self.PDF(self.t_and_theta[1], self.t_and_theta[0], F, tau1, self.tau2_pam)

        sum = -np.sum(np.log(decay))
        return sum

    def minimise(self): #finds parameter by minimising NLL

        if self.mode == 1: #for time only data
            minim = opt.minimize(self.NLL, self.param_guess, bounds = self.pam_bounds, method = 'L-BFGS-B')
            self.minNLL1 = self.NLL(minim.x)
        else: #for both time and angle data
            minim = opt.minimize(self.NLL, self.param_guess, bounds = self.pam_bounds, method = 'L-BFGS-B')
            self.minNLL2 = self.NLL(minim.x)

        #return fitted parameters
        self.F = minim.x[0]
        self.tau1 = minim.x[1]
        self.tau2 = minim.x[2]
        print("Parameter F is " + str(self.F) + ".")
        print("Parameter tau1 is " + str(self.tau1) + ".")
        print("Parameter tau2 is " + str(self.tau2) + ".")

    def cross_error(self): #finds parameter value at which NLL = NLL minimum + 0.5
        for k in range(0, self.F_array.size):
            #if the threshold have been passed
            if self.NLL_array[k] > self.error_NLL and self.NLL_array[k-1] < self.error_NLL:
                #create a list with differences between the minimised NLL and its neighbouring NLL value points
                NLL_diff = [abs(self.error_NLL - self.NLL_array[k]), abs(self.error_NLL - self.NLL_array[k-1])]
                #mark the index with a value closer to the minimised NLL
                if NLL_diff.index(min(NLL_diff)) == 0:
                    self.error_index = k
                else:
                    self.error_index = k - 1
                break #the value with an error has been found


    def findPamError(self): #estimate simple error for best-fit values
        #NLL threshold at which a parameter has its error, i.e. minimised NLL + 0.5
        if self.mode == 1: #for time only data
            self.error_NLL = self.minNLL1 + 0.5
        else: #for both time and angle data
            self.error_NLL = self.minNLL2 + 0.5
        #creates a parameter array with set boundaries around its fit parameter
        self.F_array = np.linspace(self.F, self.F+0.1, 300)
        self.tau1_array = np.linspace(self.tau1, self.tau1+0.1, 300)
        self.tau2_array = np.linspace(self.tau2, self.tau2+0.1, 300)
        #initiate a NLL array corresponding to the parameter array
        self.NLL_array = np.zeros((self.tau1_array.size))

        #find simple error for F
        for k in range(0, self.F_array.size): #populate the NLL array
            self.NLL_array[k] = self.NLL([self.F_array[k], self.tau1, self.tau2])

        self.cross_error() #find parameter value at which NLL = NLL minimum + 0.5
        #subtract a parameter value at minimised NLL from a parameter value at NLL minimum  + 0.5
        F_error = abs(self.F_array[self.error_index] - self.F)

        #find simple error for tau1
        for k in range(0, self.F_array.size):
            self.NLL_array[k] = self.NLL([self.F, self.tau1_array[k], self.tau2])

        self.cross_error()

        tau1_error = abs(self.tau1_array[self.error_index] - self.tau1)

        #find simple error for tau2
        for k in range(0, self.F_array.size):
                self.NLL_array[k] = self.NLL([self.F, self.tau1, self.tau2_array[k]])

        self.cross_error()

        tau2_error = abs(self.tau2_array[self.error_index] - self.tau2)

        print("Simple error on parameter F is " + str(F_error) + ".")
        print("Simple error on parameter tau1 is " + str(tau1_error) + ".")
        print("Simple error on parameter tau2 is " + str(tau2_error) + ".")

    def properError(self): #estimate proper error for best-fit values
        #NLL threshold at which a parameter has its error, i.e. minimised NLL + 0.5
        if self.mode == 1: #for time only data
            self.error_NLL = self.minNLL1 + 0.5
        else: #for both time and angle data
            self.error_NLL = self.minNLL2 + 0.5

        #create arrays with values varying around its best-fit parameter
        self.F_array = np.linspace(self.F, self.F+0.1, 300)
        self.tau1_array = np.linspace(self.tau1, self.tau1+0.1, 300)
        self.tau2_array = np.linspace(self.tau2, self.tau2+0.1, 300)
        #initiate an NLL array corresponding to the parameter array
        self.NLL_array = np.zeros((self.F_array.size))

        #finding proper error for F
        #initial guess of parameter values with best-fits for the minimiser
        param_guess = np.array([self.tau1, self.tau2])
        pam_bounds = self.pam_bounds[1:] #bounds for (tau1, tau2) for the minimiser

        for k in range(0, self.F_array.size): #populate the NLL array
            self.F_pam = self.F_array[k] #assign F value for NLL method in the minimiser
            #find best-fit values of other parameters for a specific F value
            minim = opt.minimize(self.NLL_F, param_guess, bounds = pam_bounds, method = 'L-BFGS-B')
            self.NLL_array[k] = self.NLL_F([minim.x[0], minim.x[1]])

        self.cross_error() #finds F value at which NLL = NLL minimum + 0.5

        #subtract a parameter value at minimised NLL from a parameter value at NLL minimum  + 0.5
        F_error = abs(self.F_array[self.error_index] - self.F)
        print("Proper error on parameter F is " + str(F_error) + ".")

        #finding proper error for tau1
        param_guess = np.array([self.F, self.tau2])
        pam_bounds = self.pam_bounds[:2] #parameter bounds for (F, tau2)

        for k in range(0, self.tau1_array.size):
            self.tau1_pam = self.tau1_array[k]
            minim = opt.minimize(self.NLL_tau1, param_guess, bounds = pam_bounds, method = 'L-BFGS-B')
            self.NLL_array[k] = self.NLL_tau1([minim.x[0], minim.x[1]])

        self.cross_error()
        tau1_error = abs(self.tau1_array[self.error_index] - self.tau1)
        print("Proper error on parameter tau1 is " + str(tau1_error) + ".")

        #finding proper error for tau2
        param_guess = np.array([self.F, self.tau1])
        pam_bounds = [self.pam_bounds[0], self.pam_bounds[2]]

        for k in range(0, self.tau2_array.size):
            self.tau2_pam = self.tau2_array[k]
            minim = opt.minimize(self.NLL_tau2, param_guess, bounds = pam_bounds, method = 'L-BFGS-B')
            self.NLL_array[k] = self.NLL_tau2([minim.x[0], minim.x[1]])

        self.cross_error()
        tau2_error = abs(self.tau2_array[self.error_index] - self.tau2)
        print("Proper error on parameter tau2 is " + str(tau2_error) + ".")
