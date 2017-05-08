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
            dfCopy.ix[i, method] = impVol.bsmBisectionVol()[0]
            dfCopy.ix[i, method+'_steps'] = impVol.bsmBisectionVol()[1]

        elif method == 'newton':
            dfCopy.ix[i, method] = impVol.bsmNewtonVol()[0]
            dfCopy.ix[i, method + '_steps'] = impVol.bsmNewtonVol()[1]

        elif method == 'mullerBisection':
            dfCopy.ix[i, method] = impVol.bsmMullerBisectionVol()[0]
            dfCopy.ix[i, method + '_steps'] = impVol.bsmMullerBisectionVol()[1]

        elif method == 'halley':
            dfCopy.ix[i, method] = impVol.bsmHalley()[0]
            dfCopy.ix[i, method + '_steps'] = impVol.bsmHalley()[1]

        elif method == 'new_newton':
            #impVol.bsmMullerBisectionInitial(initialIter=initialIter)
            impVol.bsmBrentInitial(initialIter=initialIter)
            dfCopy.ix[i, method] = impVol.bsmNewtonVol()[0]
            dfCopy.ix[i, method + '_steps'] = impVol.bsmNewtonVol()[1]

        elif method == 'new_halley':
            #impVol.bsmMullerBisectionInitial(initialIter=initialIter)
            impVol.bsmBrentInitial(initialIter=initialIter)
            dfCopy.ix[i, method] = impVol.bsmHalley()[0]
            dfCopy.ix[i, method + '_steps'] = impVol.bsmHalley()[1]

        elif method == 'halleyMomentum':
            #impVol.bsmMullerBisectionInitial(initialIter=initialIter)
            impVol.bsmBrentInitial(initialIter=initialIter)
            dfCopy.ix[i, method] = impVol.bsmHalleyMomentum()[0]
            dfCopy.ix[i, method + '_steps'] = impVol.bsmHalleyMomentum()[1]

        elif method == 'brentq':
            dfCopy.ix[i, method] = impVol.bsmBrentq()[0]
            dfCopy.ix[i, method + '_steps'] = impVol.bsmBrentq()[1].iterations

        elif method == 'brenth':
            dfCopy.ix[i, method] = impVol.bsmBrenth()[0]
            dfCopy.ix[i, method + '_steps'] = impVol.bsmBrenth()[1].iterations

        elif method == 'ridder':
            dfCopy.ix[i, method] = impVol.bsmRidder()[0]
            dfCopy.ix[i, method + '_steps'] = impVol.bsmRidder()[1].iterations

        elif method == 'scipy_newton':
            dfCopy.ix[i, method] = impVol.bsmScipyNewton()[0]
            dfCopy.ix[i, method + '_steps'] = impVol.bsmScipyNewton()[1]

    dfCopy.loc[:, 'log_difference'] = -np.log(np.fabs(dfCopy.IV - dfCopy.ix[:, method]))
    dfCopy.loc[:, 'MSE'] = np.power((dfCopy.IV - dfCopy.ix[:, method]), 2)
    dfCopy.loc[:, 'log_mse'] = np.log(1 + dfCopy.MSE)

    log_accuracy = dfCopy.log_difference.mean()
    mse = dfCopy.MSE.mean()
    log_mse = dfCopy.log_mse.mean()
    duration = datetime.datetime.now() - time_start
    efficiency = np.power(np.e, mse) * np.log(1 + duration.seconds)
    steps = dfCopy.loc[:, method + '_steps'].mean()


    return log_accuracy, mse, log_mse, duration, steps, efficiency  # datetime.time(hour=0, minute=0, second=duration.seconds, microsecond=duration.microseconds).strftime("%M:%S.%f")[:-3]

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