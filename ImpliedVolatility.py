import numpy as np
from scipy import stats
from datetime import date
import time


class ImpliedVolatility():
    def __init__(self, S, K, T, r, sigma, cStar, optionType, maxIter = 1e4, tolerance = 1e-10):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.cStar = cStar
        self.optionType = optionType
        self.maxIter = int(maxIter)
        self.tolerance =tolerance

    def bsmValue(self, sigma = None):
        if sigma == None:
            sigma = self.sigma

        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)

        if self.optionType in ['Call', 'call', 'CALL']:

            return self.S * stats.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2)

        elif self.optionType in ['Put', 'put', 'PUT']:

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

    def bsmBisectionVol(self, upper = 10):
        lower = 1e-15
        middle = (lower + upper)/2

        for i in range(self.maxIter):
            old_middle = (lower + upper) / 2
            if self.f(lower) * self.f(middle) < 0:
                upper = middle
            else:
                lower = middle

            middle = (lower + upper) / 2

            if np.fabs(old_middle - middle) < self.tolerance:
                return middle

        return middle

    def bsmNewtonVol(self):
        for i in range(self.maxIter):
            sigma = self.sigma
            self.sigma = sigma - self.f()/self.bsmVega()
            if np.fabs(sigma - self.sigma) < self.tolerance:
                return self.sigma

        return self.sigma

    def bsmMullerBisectionVol(self, upper = 10):
        lower = 1e-15
        middle = (lower + upper) / 2

        for i in range(self.maxIter):

            muller = self.bsmMuller(lower, upper, middle)

            old_middle = (lower + upper) / 2


            if self.f(lower) * self.f(middle) < 0:
                upper = middle
            else:
                lower = middle

            if muller < lower or muller > upper:
                middle = (lower + upper) / 2
            else:
                middle = muller

            if (np.fabs(old_middle - middle) < self.tolerance) or np.isnan(self.bsmMuller(lower, upper, middle)):
                return middle

        return middle

    def bsmHalley(self):
        newSigma = self.sigma
        for i in range(self.maxIter):

            newSigma = self.sigma + (-self.bsmVega() + np.sqrt(self.bsmVega() ** 2 - 2 * self.f() * self.bsmVomma())) / self.bsmVomma()
            self.sigma = newSigma

            if np.fabs(newSigma - self.sigma) < self.tolerance:
                return self.sigma

        return self.sigma

    def bsmMullerBisectionInitial(self, initialIter = 10, lower=1e-15, upper=10):

        middle = (lower + upper) / 2

        for i in range(initialIter):

            muller = self.bsmMuller(lower, upper, middle)

            old_middle = (lower + upper) / 2


            if self.f(lower) * self.f(middle) < 0:
                upper = middle
            else:
                lower = middle

            if muller < lower or muller > upper:
                middle = (lower + upper) / 2
            else:
                middle = muller

            if (np.fabs(old_middle - middle) < self.tolerance) or np.isnan(self.bsmMuller(lower, upper, middle)):
                break

        self.sigma = middle



if __name__ == '__main__':

    today = date(2016,11,15)
    expDay = date(2016, 12, 16)
    T = expDay - today
    sigma0 = 0.5
    r = 0.02 # 3-month T-bill rate
    currentStockPrice = 16.26


    bacOptionList = [[currentStockPrice, 16.00, T.days / 365, r , sigma0, 0.77, 'call'],
                 [currentStockPrice, 17.00, T.days / 365, r, sigma0, 0.31, 'call'],
                 [currentStockPrice, 16.00, T.days / 365, r, sigma0, 0.56, 'put'],
                 [currentStockPrice, 17.00, T.days / 365, r, sigma0, 1.12, 'put']]


    print('The underlying asset is Bank of America (BAC), current stock price is ${:.2f}, the expiration date is {:%Y-%m-%d}\n'.format(currentStockPrice, expDay))
    print('Now, let\'s begin:\n')

    time_start = time.clock()
    for option in bacOptionList:
        impvol = ImpliedVolatility(option[0], option[1], option[2], option[3], option[4], option[5], option[6]).bsmBisectionVol()
        print('Here is a {0} option.\nThe strike price is ${1:.2f} and option price is ${2:.2f}.\nThe implied volatility is {3:.16%}\n'.format(option[6], option[1], option[5], impvol))
    print('The Bisection method takes {:.4f} seconds to run\n'.format(time.clock() - time_start))

    time_start = time.clock()
    for option in bacOptionList:
        impvol = ImpliedVolatility(option[0], option[1], option[2], option[3], option[4], option[5], option[6]).bsmMullerBisectionVol()
        print('Here is a {0} option.\nThe strike price is ${1:.2f} and option price is ${2:.2f}.\nThe implied volatility is {3:.16%}\n'.format(option[6], option[1], option[5], impvol))
    print('The Muller-Bisection method takes {:.4f} seconds to run\n'.format(time.clock() - time_start))

    time_start = time.clock()
    for option in bacOptionList:
        impvol = ImpliedVolatility(option[0], option[1], option[2], option[3], option[4], option[5], option[6]).bsmNewtonVol()
        print('Here is a {0} option.\nThe strike price is ${1:.2f} and option price is ${2:.2f}.\nThe implied volatility is {3:.16%}\n'.format(option[6], option[1], option[5], impvol))
    print('The Newton method takes {:.4f} seconds to run\n'.format(time.clock() - time_start))

    time_start = time.clock()
    for option in bacOptionList:
        impvol = ImpliedVolatility(option[0], option[1], option[2], option[3], option[4], option[5], option[6]).bsmHalley()
        print('Here is a {0} option.\nThe strike price is ${1:.2f} and option price is ${2:.2f}.\nThe implied volatility is {3:.15%}\n'.format(option[6], option[1], option[5], impvol))
    print('The Halley method takes {:.4f} seconds to run\n'.format(time.clock() - time_start))