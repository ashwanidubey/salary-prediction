import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def display_and_select(Files):
    i=0
    for file_name in Files:
        print(i,"<-->",file_name)
        i+=1
    return Files[int(input("Select file no to select csv file "))]    
        
def welcome():
    print("hii,how r you\npress enter key to proceed")
    input()

def check_csv():
    csv_files=[]
    current_dir=os.getcwd()
    files=os.listdir(current_dir)
    for x in files:
        if x.split('.')[-1]=='csv':
            csv_files.append(x)
    if len(csv_files)==0:
        return "sorry ! no csv file found in the given directory"
    else :
        return csv_files

def  graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred):
    plt.scatter(X_train,Y_train,color='red',label='training data')
    plt.plot(X_train,regressionObject.predict(X_train),color='blue',label='Best')
    plt.scatter(X_test,Y_test,color='green',label='test data')
    plt.scatter(X_test,Y_pred,color='black',label='predicted test data')
    plt.title("salary vs Experience")
    plt.xlabel('years of experience')
    plt.ylabel('salary')
    plt.legend()
    plt.show()
    
def main1():
    welcome()
    try:
        csv_files=check_csv()
        if csv_files=="sorry ! no csv file found in the given directory":
            raise FileNotFoundError("sorry ! no csv file found in the given directory")
        csv_file=display_and_select(csv_files)
        print(csv_files ,"is selected")
        print("reading csv file")
        print("creating database")
        database=pd.read_csv(csv_file)
        print("database created")
        X=database.iloc[:,:-1].values
        Y=database.iloc[:,-1].values
        s=float(input("enter size(0 to 1) of data u want to test "))
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=s)
        print("model creation in progression")
        regressionObject=LinearRegression()
        regressionObject.fit(X_train,Y_train)
        print("model is created")
        print("press Enter key to predict test data in trained model ")
        input()
        Y_pred=regressionObject.predict(X_test)
        print(Y_pred,"\n",Y_test)
        i=0
        print("X_test.....Y_test....Y_pred")
        while i<len(X_test):
            print(X_test[i],'...',Y_test[i],'...',Y_pred[i])
            i+=1
        print("press enter to see above result in graphical form")
        graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred)
        r2=r2_score(Y_test,Y_pred)
        print("our model is  %2.2f%% accurate"%(r2*100))

        print("now u can predict the sallerey of employee using our model")
        print("\nenter experience in years of the candidate,separated by comma ")

        exp=list(map(float,input().split(',')))
        ex=[]
        for x in exp:
            ex.append([x])
        experience=np.array(ex)
        salaries=regressionObject.predict(experience)
        
        plt.scatter(exp,salaries,color='black')
        plt.xlabel('years of experience')
        plt.ylabel('Salaries')
        plt.show()

        d=pd.DataFrame({'Experience':exp,'salaries':salaries})
        print(d)
        
    except FileNotFoundError :
        print('no csv file in the directory\npress enter key to exit')
        input()
        exit()
if __name__=="__main__":
    main1()
    input()
