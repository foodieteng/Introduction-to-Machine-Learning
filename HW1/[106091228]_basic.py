#!/usr/bin/env python
# coding: utf-8

# import packages
# Note: You cannot import any other packages!
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random



# Global attributes
# Do not change anything here except TODO 1 
StudentID = '106091228' # TODO 1 : Fill your student ID here
input_dataroot = 'input.csv' # Please name your input csv file as 'input.csv'
output_dataroot = StudentID + '_basic_prediction.csv' # Output file will be named as '[StudentID]_basic_prediction.csv'

input_datalist =  [] # Initial datalist, saved as numpy array
output_datalist =  [] # Your prediction, should be 20 * 2 matrix and saved as numpy array
                      # The format of each row should be [Date, TSMC_Price_Prediction] 
                      # e.g. ['2021/10/15', 512]

# You can add your own global attributes here
mtk_train = []
tsmc_train = []
mtk_test = []
tsmc_test_actual = []
tsmc_test_predic = []
mtk_predict = []
tsmc_predict = []
tsmc_actual = []
prediction_date = []


# Read input csv to datalist
with open(input_dataroot, newline='') as csvfile:
    input_datalist = np.array(list(csv.reader(csvfile)))

# From TODO 2 to TODO 6, you can declare your own input parameters, local attributes and return parameters
    
def SplitData():
# TODO 2: Split data, 2021/10/15 ~ 2021/11/11 for testing data, and the other for training data and validation data 
    for i in range(169):
        mtk_train.append(float(input_datalist[i][1]))
        tsmc_train.append(float(input_datalist[i][2]))
    for i in range(20):
        mtk_test.append(float(input_datalist[-40 + i][1]))
        tsmc_test_actual.append(float(input_datalist[-40 + i][2]))
        mtk_predict.append(float(input_datalist[-20 + i][1]))
        prediction_date.append(input_datalist[-20 + i][0])
        tsmc_actual.append(float(input_datalist[-20 + i][2]))

def PreprocessData():
# TODO 3: Preprocess your data  e.g. split datalist to x_datalist and y_datalist
    pass


def Regression():
# TODO 4: Implement regression
    x = np.array(mtk_train)
    y = np.array(tsmc_train)
    coef = np.polyfit(x, y, 2)
    
    return coef


def CountLoss(actual, predict):
# TODO 5: Count loss of training and validation data
   
    actual = np.array(actual)
    predict = np.array(predict)
    
    return float(np.mean(np.abs((actual - predict) / actual)) * 100.0)

def MakePrediction():
# TODO 6: Make prediction of testing data 
    coef = Regression()
    predict = np.poly1d(coef)

    for i in range(20):
        tsmc_predict.append(int((predict(mtk_predict[i]))))
        tsmc_test_predic.append(float((predict(mtk_test[i]))))
            
    for i in range(20):
        output_datalist.append([prediction_date[i], tsmc_predict[i]])
    

# TODO 7: Call functions of TODO 2 to TODO 6, train the model and make prediction
SplitData()
MakePrediction()

MAPE_test = CountLoss(tsmc_test_actual, tsmc_test_predic)
MAPE_predict = CountLoss(tsmc_actual, tsmc_predict)

print('Test MAPE is: {:.2f}%'.format(MAPE_test))
print('Prediction MAPE is: {:.2f}%'.format( MAPE_predict))
output_datalist = np.array(output_datalist)
# Write prediction to output csv
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    
    for row in output_datalist:
        writer.writerow(row)

