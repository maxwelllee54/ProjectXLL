import numpy as np
from scipy import stats
from scipy.optimize import brentq, brenth, ridder, newton
import re


class ImpliedVolatility():
    def __init__(self, S, K, T, r, sigma, cStar, optionType, maxIter = 10000, tolerance = 1e-10):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.cStar = cStar
        self.optionType = optionType.lower()
        self.maxIter = int(maxIter)
        self.tolerance = tolerance

    def bsmValue(self, sigma = None):
        if sigma == None:
            sigma = self.sigma

        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)

        if self.optionType == 'call':

            return self.S * stats.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2)

        elif self.optionType == 'put':

            return self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2) - self.S * stats.norm.cdf(-d1)

        else:
            raise TypeError('the option_type argument must be either "call" or "put"')

    def bsmVega(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        vega = self.S * stats.norm.pdf(d1) * np.sqrt(self.T)
        return vega

    def bsmVomma(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        return self.bsmVega() * d1 * d2 / self.sigma

    def f(self, sigma = None):
        return self.bsmValue(sigma) - self.cStar

    def bsmMuller(self, x0, x1, x2):

        f0, f1, f2 = self.f(x0), self.f(x1), self.f(x2)
        h0, h1 = x0 - x2, x1 - x2
        e0, e1 = f0 - f2, f1 - f2
        det = h0 * h1 * (h0 - h1)
        A = (e0 * h1 - h0 * e1) / det
        B = (h0 ** 2 * e1 - h1 ** 2 * e0) / det
        C = f2

        if B < 0:
            x3 = x2 - 2 * C / (B - np.sqrt(B ** 2 - 4 * A * C))
        else:
            x3 = x2 - 2 * C / (B + np.sqrt(B ** 2 - 4 * A * C))

        return x3

    def bsmBisectionVol(self, upper = 3):

        lower = -0.1
        middle = (lower + upper)/2

        if self.f(lower) * self.f(upper) > 0:
            print('No root! Change the lower and upper value!\n')
            return self.sigma, np.nan

        for i in range(self.maxIter):

            if self.f(lower) * self.f(upper) <= 0:

                if self.f(lower) * self.f(middle) < 0:
                    upper = middle

                if self.f(upper) * self.f(middle) < 0:
                    lower = middle

                middle = (lower + upper) / 2

                if np.fabs(self.f(middle)) < self.tolerance:

                    return middle, i+1

        #print('Max iteration reached! This is Bisection method.\n')
        return middle, self.maxIter

    def bsmNewtonVol(self):

        for i in range(self.maxIter):

            sigma = self.sigma
            self.sigma = sigma - self.f(sigma)/self.bsmVega()

            if np.fabs(self.f(self.sigma)) < self.tolerance:
                return self.sigma, i + 1

            # avoid the sigma diverge and become too big
            # if not (0.0 <= self.sigma <= 3):
            #    print('Newton Diverge!\n')
            #    return sigma, self.maxIter

            if np.isnan(self.sigma):
                break

            if np.fabs(sigma - self.sigma) < self.tolerance:
                return self.sigma, i + 1

            #if i > (self.maxIter - 20):
            #    print('Steps: {}, {}'.format(sigma, self.sigma))

        #print('Max iteration reached! This is Newton method.\n')
        return self.sigma, self.maxIter

    def bsmMullerBisectionVol(self, upper = 3):
        lower = -0.1
        middle = (lower + upper) / 2

        if self.f(lower) * self.f(upper) > 0:
            print('No root! Change the lower and upper value!\n')
            return self.sigma, np.nan

        for i in range(self.maxIter):
            if self.f(lower) * self.f(upper) <= 0:

                muller = self.bsmMuller(lower, upper, middle)

                if self.f(lower) * self.f(middle) < 0:
                    upper = middle

                if self.f(upper) * self.f(middle) < 0:
                    lower = middle

                if muller < lower or muller > upper:

                    middle = (lower + upper) / 2

                else:
                    middle = muller

                if np.fabs(self.f(middle)) < self.tolerance:
                    return middle, i+1

        #print('Max iteration reached! This is MullerBisection method.')
        return middle, self.maxIter

    def bsmHalley(self):
        for i in range(self.maxIter):
            oldSigma = self.sigma
            self.sigma = oldSigma + (-self.bsmVega() + np.sqrt(self.bsmVega() ** 2 - 2 * self.f(oldSigma) * self.bsmVomma())) / self.bsmVomma()

            if np.fabs(self.f(self.sigma)) < self.tolerance:
                return self.sigma, i+1

            # diverged
            #if not (0.0 <= self.sigma <= 3):
            #    print('Halley Diverge!\n')
            #    return oldSigma, self.maxIter
            if np.isnan(self.sigma):
                break

            if np.fabs(oldSigma - self.sigma) < self.tolerance:
                return self.sigma, i + 1

            #if i > (self.maxIter - 20):
            #    print('Steps: {}, {}'.format(newSigma, self.sigma))

        #print('Max iteration reached! This is Halley!\n')
        return self.sigma, self.maxIter

    def bsmBrentInitial(self, initialIter=1, lower=-0.1, upper=3):

        if self.f(lower) * self.f(upper) > 0:
            print('No root! Change the lower and upper value!\n')
            return None

        self.sigma = brentq(self.f, lower, upper, args=(), xtol=2e-12, rtol=8.8817841970012523e-16, maxiter=initialIter, full_output=False, disp=False)

    def bsmMullerBisectionInitial(self, initialIter=1, lower=-0.1, upper=2):

        if self.f(lower) * self.f(upper) > 0:
            print('No root! Change the lower and upper value!\n')
            return None

        middle = (lower + upper) / 2
        for i in range(initialIter):

            muller = self.bsmMuller(lower, upper, middle)

            if self.f(lower) * self.f(middle) < 0:
                upper = middle
            else:
                lower = middle

            if muller < lower or muller > upper:
                middle = (lower + upper) / 2
            else:
                middle = muller

            if np.fabs(self.f(middle)) < self.tolerance:
                break

        self.sigma = middle

    def bsmBrentq(self, a=-0.1, b=3):
        if self.f(a) * self.f(b) > 0:
            print('No root! Change the lower and upper value!\n')
            return self.sigma, np.nan

        for i in range(self.maxIter):
            if self.f(a) * self.f(b) < 0:
                return brentq(self.f, a, b, args=(), xtol=2e-12, rtol=8.8817841970012523e-16, maxiter=100, full_output=True, disp=True)


    def bsmBrenth(self, a=-0.1, b=3):
        if self.f(a) * self.f(b) > 0:
            print('No root! Change the lower and upper value!\n')
            return self.sigma, np.nan

        for i in range(self.maxIter):
            if self.f(a) * self.f(b) < 0:
                return brenth(self.f, a, b, args=(), xtol=2e-12, rtol=8.8817841970012523e-16, maxiter=100, full_output=True, disp=True)

    def bsmRidder(self, a=-0.1, b=3):
        if self.f(a) * self.f(b) > 0:
            print('No root! Change the lower and upper value!\n')
            return self.sigma, np.nan

        for i in range(self.maxIter):
            if self.f(a) * self.f(b) < 0:
                return ridder(self.f, a, b, args=(), xtol=2e-12, rtol=8.8817841970012523e-16, maxiter=100, full_output=True, disp=True)

    def bsmScipyNewton(self, x0 = None):
        r = re.compile(r'\d?\.\d+')

        if x0 == None:
            x0 = self.sigma
        try:
            return newton(self.f, x0, fprime=None, args=(), tol=1.48e-08, maxiter=50, fprime2=None), 0
        except RuntimeError as err:
            return float(r.findall(err.args[0])[0]), 100

    def bsmHalleyMomentum(self, t=0.9):
        #newSigma = self.sigma
        for i in range(self.maxIter):
            oldSigma = self.sigma
            self.sigma = t * oldSigma - (self.f(oldSigma) / self.bsmVega()) / (1 - self.f(oldSigma) * self.bsmVomma() / (2 * self.bsmVega() * self.bsmVega()))

            if np.fabs(self.f(self.sigma)) < self.tolerance:
                return self.sigma, i+1

            # diverged
            #if not (0.0 <= self.sigma <= 3):
            #    print('HalleyMom Diverge!\n')
            #    return oldSigma, self.maxIter

            if np.isnan(self.sigma):
                break


            if np.fabs(oldSigma - self.sigma) < self.tolerance:
                return self.sigma, i + 1

            #if i > (self.maxIter - 20):
            #    print('Steps: {}, {}'.format(newSigma, self.sigma))

        #print('Max iteration reached! This is HalleyMomentum!\n')
        return self.sigma, self.maxIter

if __name__ == '__main__':
    import methods
    import pandas as pd
    import numpy as np

    data = pd.read_csv('europeanOptions_final_0510.csv')
    data = data.loc[:, ['currentDate', 'ExpDate', 'StrikePrice', 'Ticker', 'Type', 'Last', 'IV', 'StockPrice', 'T']]
    testMethods = ['brentq', 'brenth', 'ridder', 'bisection', 'mullerBisection', 'newton', 'new_newton', 'halley',
                   'new_halley', 'halleyMomentum']

    #testMethods = ['bisection']
    # testSigma = [0.33, 0.66 , 0.99, 1.33, 1.66, 1.99]
    #testSigma = np.random.uniform(0, 2, 4)
    testIter = [5]
    result = pd.DataFrame(index=testMethods,
                          columns=['log_accuracy', 'mse', 'msd', 'log_mse', 'duration', 'steps', 'efficiency', 'sigma'])
    dicResult = {}

    for iter in testIter:
        dicResult[iter] = {}
        for method in testMethods:
            print(method)
            result.loc[method, :] = methods.performance(data, method, initialIter=iter)
        dicResult[iter] = result.copy()
        print('This is the test result for initial iteration {:d} times:\n{}\n\n'.format( iter, result))