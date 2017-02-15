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
        fileList = []

        dir_list = os.walk(self.rootPath)

        for root, dirs, files in dir_list:
            for file in files:
                if file[-3:] in ['csv', 'lsx', 'xls']:
                    fileList.append(os.path.join(root, file))


        for file in fileList:
            if file[-3:] == 'csv':
                csv = pd.read_csv()



if __name__ == '__main__':
    print(OptionDataMerger('/Users/maxwelllee54/GitHubs/ProjectXLL').fileList())




