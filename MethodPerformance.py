from ImpliedVolatility import ImpliedVolatility
import pandas as pd
import numpy as np
import datetime


def performance(df, method, r = 0.05, sigma0 = 0.5):
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
            impVol.bsmMullerBisectionInitial()
            dfCopy.ix[i, method] = impVol.bsmNewtonVol()

        elif method == 'new_halley':
            impVol.bsmMullerBisectionInitial()
            dfCopy.ix[i, method] = impVol.bsmHalley()

    dfCopy.loc[:, 'Difference'] = -np.log(np.fabs(dfCopy.IV - dfCopy.ix[:, method]))

    accuracy = dfCopy.Difference.mean()
    duration = datetime.datetime.now() - time_start

    return accuracy, duration


data = pd.read_csv('europeanOptions.csv')
data = data.loc[:, ['cDate', 'ExpDate', 'StrikePrice', 'Ticker', 'Type', 'Last', 'IV', 'StockPrice', 'T']]

testMethods = ['bisection', 'mullerBisection', 'newton', 'halley', 'new_newton', 'new_halley']
testSigma = [0.1, 0.3, 0.5, 0.7, 1]
result = pd.DataFrame(index=testMethods, columns=['Accuracy', 'Time(hh:mm:ss)'])
resultDic = {}


for sigma in testSigma:
    for method in testMethods:
        result.loc[method, :] = performance(data, method, sigma0=sigma)

    resultDic[sigma] = result
    print('This is the test result for initial point sigma {}:\n{}\n\n'.format(sigma, result))


