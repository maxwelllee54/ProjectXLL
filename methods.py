from ImpliedVolatility import ImpliedVolatility
import pandas as pd
import numpy as np
import datetime
def performance(df, method, r=0.05, sigma0=0.5, initialIter=1):
    time_start = datetime.datetime.now()
    dfCopy = df.copy()

    for i in df.index:
        impVol = ImpliedVolatility(df.ix[i, 'StockPrice'], df.ix[i, 'StrikePrice'], df.ix[i, 'T'], r, sigma0,
                                   df.ix[i, 'Last'], df.ix[i, 'Type'])

        if method == 'bisection':
            dfCopy.ix[i, method] = impVol.bsmBisectionVol()

        elif method == 'newton':
            dfCopy.ix[i, method] = impVol.bsmNewtonVol()

        elif method == 'mullerBisection':
            dfCopy.ix[i, method] = impVol.bsmMullerBisectionVol()

        elif method == 'halley':
            dfCopy.ix[i, method] = impVol.bsmHalley()

        elif method == 'new_newton':
            impVol.bsmMullerBisectionInitial(initialIter=initialIter)
            dfCopy.ix[i, method] = impVol.bsmNewtonVol()

        elif method == 'new_halley':
            impVol.bsmMullerBisectionInitial(initialIter=initialIter)
            dfCopy.ix[i, method] = impVol.bsmHalley()

        elif method == 'brentq':
            dfCopy.ix[i, method] = impVol.bsmBrentq()

        elif method == 'brenth':
            dfCopy.ix[i, method] = impVol.bsmBrenth()

        elif method == 'ridder':
            dfCopy.ix[i, method] = impVol.bsmRidder()

        elif method == 'scipy_newton':
            dfCopy.ix[i, method] = impVol.bsmScipyNewton()

    dfCopy.loc[:, 'log_difference'] = -np.log(np.fabs(dfCopy.IV - dfCopy.ix[:, method]))
    dfCopy.loc[:, 'MSE'] = np.power((dfCopy.IV - dfCopy.ix[:, method]), 2)

    log_accuracy = dfCopy.log_difference.mean()
    mse = dfCopy.MSE.mean()
    duration = datetime.datetime.now() - time_start

    return log_accuracy, mse, duration  # datetime.time(hour=0, minute=0, second=duration.seconds, microsecond=duration.microseconds).strftime("%M:%S.%f")[:-3]

if __name__ == '__main__':
    data = pd.read_csv('europeanOptions_.01tol.csv')
    data = data.loc[:, ['currentDate', 'ExpDate', 'StrikePrice', 'Ticker', 'Type', 'Last', 'IV', 'StockPrice', 'T']]

    testMethods = ['brentq', 'brenth', 'ridder', 'scipy_newton', 'bisection', 'mullerBisection', 'newton', 'halley',
                   'new_newton', 'new_halley']
    testSigma = [0.3, 0.5, 0.7, 1, 1.5, 1.7, 2]
    testIter = [1, 2, 3, 5]
    result = pd.DataFrame(index=testMethods, columns=['log_accuracy', 'MSE', 'Time(hh:mm:ss)'])
    dicResult = {}

    for iter in testIter:
        dicResult[iter] = {}
        for sigma in testSigma:
            for method in testMethods:
                result.loc[method, :] = performance(data, method, sigma0=sigma, initialIter=iter)
            dicResult[iter][sigma] = result.copy()
            print('This is the test result for initial point sigma {:.2f}, \
    initial iteration {:d} times:\n{}\n\n'.format(sigma, iter, result))