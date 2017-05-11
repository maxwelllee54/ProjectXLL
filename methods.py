from ImpliedVolatility import ImpliedVolatility
import pandas as pd
import numpy as np
import datetime
def performance(df, method, r=0.005, sigma0=None, initialIter=1, maxIter = 10000):
    time_start = datetime.datetime.now()
    dfCopy = df.copy()

    for i in df.index:
        if sigma0 == None:
            sigma0 = np.random.uniform(0, 1)

        impVol = ImpliedVolatility(df.ix[i, 'StockPrice'], df.ix[i, 'StrikePrice'], df.ix[i, 'T'], r, sigma0,
                                   df.ix[i, 'Last'], df.ix[i, 'Type'], maxIter=maxIter)

        dfCopy.ix[i, 'sigma'] = sigma0

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

    # discard the records that diverged
    dfCopy.to_csv(method + '.csv')

    #dfNew = dfCopy.loc[dfCopy.loc[:, method + '_steps'] < maxIter, :].copy()
    dfNew = dfCopy.loc[np.isfinite(dfCopy.loc[:, method + '_steps']), :].copy()

    dfNew.loc[:, 'log_difference'] = -np.log(np.fabs(dfNew.IV - dfNew.ix[:, method]))
    dfNew.loc[:, 'MSE'] = np.power(dfNew.IV - dfNew.ix[:, method], 2)
    dfNew.loc[:, 'MSD'] = np.fabs(dfNew.IV - dfNew.ix[:, method])
    dfNew.loc[:, 'log_mse'] = np.log(1 + dfNew.MSE)

    log_accuracy = dfNew.log_difference.mean()
    mse = dfNew.MSE.mean()
    msd = dfNew.MSD.mean()
    sigma = dfNew.sigma.mean()
    log_mse = dfNew.log_mse.mean()
    duration = datetime.datetime.now() - time_start
    steps = dfNew.loc[:, method + '_steps'].mean()
    efficiency = np.power(np.e, mse) / np.log2(steps+1)


    return log_accuracy, mse, msd, log_mse, duration, steps, efficiency, sigma  # datetime.time(hour=0, minute=0, second=duration.seconds, microsecond=duration.microseconds).strftime("%M:%S.%f")[:-3]

if __name__ == '__main__':
    pass