import pandas
import numpy
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re
import csv


class FileReader(object):

    def __init__(self):
        pass

    def get_vote_data(self):
        """ for reading and preparing the data from the votes.data file"""

        column_names = ['party', 'handicapped-infants', 'water-project', 'adoption-of-budget', 'fee-freeze',
                        'el-salvador-aid', 'religious-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan',
                        'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending',
                        'right-to-sue', 'crime', 'duty-free-exports', 'export-act-south-africa']
        data_set = pandas.read_csv("votes.data.txt", header=None, names=column_names)
        # print(data_set.shape)
        # print(data_set)

        # replace the ? with NaN
        data_set = data_set.replace('?', numpy.NaN)

        # replace missing values with most frequent value in column
        data_set = data_set.apply(lambda x: x.fillna(x.value_counts().index[0]))
        # print(data_set)

        # return the pandas dataframe
        return data_set

    def test_data(self):
        """This was the test_data used for me to test if the tree was being correctly built"""
        credit_score = ['g', 'g', 'g', 'g', 'm', 'm', 'm', 'm', 'l', 'l', 'l', 'l']
        income = ['h', 'h', 'l', 'l', 'h', 'l', 'h', 'l', 'h', 'h', 'l', 'l']
        collateral = ['g', 'p', 'g', 'p', 'g', 'p', 'p', 'g', 'g', 'p', 'g', 'p']
        y = ['yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'no']
        column_names = ['credit score', 'income', 'collateral', 'Loan']

        loan_array = []
        for index in range(0, len(y)):
            row = [credit_score[index], income[index], collateral[index], y[index]]
            loan_array.append(row)

        loan_array = numpy.array(loan_array)
        # print(loan_array)
        loan_array = pandas.DataFrame(loan_array, columns=column_names)
        # print(loan_array['Loan'].values.tolist())
        return loan_array
