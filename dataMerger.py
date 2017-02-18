'''
This is a program to merge all the option data into a single excel file.

'''

import os
import pandas as pd

class OptionDataMerger():

    def __init__(self, rooPath):
        self.rootPath = rooPath
        self.optionData = pd.DataFrame

    def fileList(self):

        directory = os.walk(self.rootPath)

        fileList = []

        for root, dirs, files in directory:
            for file in files:
                if file[-3:] in ['csv', 'xls'] or file[-4:] == 'xlsx':
                    fileList.append(os.path.join(root, file))



if __name__ == '__main__':
    print(OptionDataMerger('/Users/maxwelllee54/GitHubs/ProjectXLL').fileList())




