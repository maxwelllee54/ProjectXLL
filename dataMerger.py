'''
This is a program to merge all the option data into a single excel file.

'''

import os
import pandas as pd

class OptionDataMerger():

    def __init__(self, rooPath):
        self.rootPath = rooPath
        self.optionData = pd.DataFrame

    def readCSV(self):
