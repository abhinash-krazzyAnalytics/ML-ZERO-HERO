#Numpy Nd-array

import numpy as np

#how to create array
#array can be created from list/tuple
#list/tuple: Datatype of python

#array create 2D from list & tuple
tup=((1,2),(4,5),(60,45))
li=[[1,2],[3,8],[78,45]]

arr_tuple=np.array(tup)
arr_li=np.array(li)

arr_tuple
arr_li

#now what is the dimension??
print(arr_li.ndim)

#number of elemnts in array
print(arr_li.size)

#lets see the shape
print(arr_li.shape)
'''
(      [[ 1,  2],
       [ 3,  8],
       [78, 45]])
'''

#array create elemnt_value=0 number of row=10,col=1
arr_zero=np.zeros((10,3))
arr_zero

#array of onces
arr_one=np.ones((5,3))
arr_one

#50 lakh
arr_50lakh=np.ones((5,2))
arr=arr_50lakh*5000000000
print(arr)

#constanct value 10
arr_10=(np.ones((5,3))*10)
arr_10

####
array=np.array([[45,23,45],[45,78,98]])
#array: 1D/2D ???

#now you are asked to make it 1D
#Deep learning : CNN(Flatten())
print(array.flatten())
print(array)

#@question
#2D array you are asked to flatten it without using
#faltten()
#IP:array=np.array([[45,23,45],[45,78,98]])
#OP:[45 23 45 45 78 98]



#array indexing slicing/extraction
import numpy as np
arr=np.array([[-1,2,3,4],
              [23,1,4,5],
              [9,8,5,4],
              [1,2,7,6]])
arr  
'''         Columns   
  Row    0   1   2   3  
[ 0    [-1,  2,  3,  4],
  1    [23,  1,  4,  5],
  2    [ 9,  8,  5,  4],
  3    [ 1,  2,  7,  6]])

'''
#data extract from the index postion
#df=arr[row,col]
df=arr[0:2,0:3:2]
df

print(arr[1:3,1:3:+1])

#basic maths operation
arr=np.array([[1,2,3],[4,5,8]])
print(arr*5)

print(arr+50)
print(arr/5)

#array act like matrix
arr=np.array([[1,2,3],
              [4,5,8],
              [7,8,9]])

#martix trace: sum of diagonal elements
print("Trace of array is:",arr.trace())

arr=np.array([[1,2],
              [4,5],
              [7,8]])
#Matrix Transpose??
trs=arr.T
print(trs)
print("shape of arr :",arr.shape)
print("shape of trs :",trs.shape)

# age:18,20,21,25,23
#salary:50lakh,1cr,75lk,1.5cr,90lkh(per-month)

#data-Transfomation use numpy

sal=np.array([20000,45000,78000,89500,154500])
log=np.log(sal)
print(sal)
print(log)


sal=[1000,1200,4500,7800]
new=np.square(sal)
new

#array
arr=np.array([451,416,1325,135148,21])
exp=np.exp(arr)
exp


import matplotlib.pyplot as plt
sal=np.array([20000,45000,78000,89500,154500])
sr=np.array([1,2,3,4,5])
plt.Figure(figsize=(10,10))
plt.plot(sr,sal)
plt.show()



import matplotlib.pyplot as plt
sal=np.array([20000,45000,78000,89500,154500])
sr=np.array([1,2,3,4,5])
log=np.log(sal)
plt.Figure(figsize=(10,10))
plt.plot(sr,log)
plt.show()


#-------------------------------------
arr=np.array([[1,2,3],
              [4,5,6],
              [78,89,15]])
#max element value ?
print(arr.max())
#min value
print(arr.min())

#row and col wise max and min
print(arr.max(axis=1))
#axis=1 row wise

#col wise
print(arr.max(axis=0))
#axis=0 col wise

#array sum
print(arr.sum())

#row wise cumulative-sum
'''
[[1,3,6],
 [4,9,15],
 [78,167,182]]
'''
print(arr.cumsum(axis=1))

#-------------------------------
#binary operation on array
a=np.array([[1,2],
            [3,4]])
b=np.array([[11,22],
            [33,44]])

print(a+b)

#sorting in array:
arr=np.array([[11,2,3],
              [4,51,6],
              [78,89,15]])

print("sorted array is:",np.sort(arr))

# row axis=1 col axis=0
print("sorted array is:",np.sort(arr,
                    axis=1))

print("sorted array is:",np.sort(arr,
                    axis=0))


#stacking: numpy 2 form stacking
#V-stack  #H-stack

a=np.array([[1,2,3],
            [6,5,4],
            [45,78,98]])
b=np.array([[1,2,45],
            [5,4,78]])

print("stacking :")
print(np.vstack((a,b)))

#V-stack: number of columns Same
#number of row diff


#H-stack:
#number of row Same
#number of diff diff

a=np.array([[1,2,3],
            [6,5,4]])
b=np.array([[2,45],
            [4,78]])
print("stacking :")
print(np.hstack((a,b)))


#array forms
name=np.array(["raja","raj",
               "diya","tina"])
math=np.array([78,50,46,89])
phy=np.array([89,78,65,45])

score=np.vstack((math,phy))

df=np.vstack((name,score))
df

#------------------------------------
Ml-LIBRARY NUMBER 2: PANDAS

PANDAS: Most import library for ML/DL/DATA/DA
#Structure data Row and colms

import pandas as pd

#pandas how may datastructure
1) Series: 1D
2) DataFrame***: 2D (tabluar data)
3) Panel Data: 3D(Rare Industry)

#creating DataFrame
import pandas as pd

dic={"NAME":["teja","goga","shakal","jivan"],
     "GRE_MARKS":[320,340,321,318]}
print(type(dic))

df=pd.DataFrame(dic)
print(df)
print(type(df))

#-----------------------------------------
#question: To import the data from Csv.
import pandas as pd
#file loaction/file path
data=pd.read_csv('DATA.csv')
#if working  dir is same 
print(data)

#if working Dir is diff
import pandas as pd
data=pd.read_csv(r'C:\Users\Abhinash\Desktop\DATA.csv')
print(data)

#-------------------------------------
#interview Excel Workbook: 4 Sheet
#4-question 
#read from excel workbook
data=pd.read_excel('data.xlsx')
data

#by sheetname data read
data_name=pd.read_excel('data.xlsx',
                        sheet_name='name')
data_name

data_age=pd.read_excel('data.xlsx',
                        sheet_name='age')
data_age

#--------------------------------------
#txt read--convert into CSv format
#Data (DataBase:query read how will you do ??)
#homework: how to read data from a database ??







