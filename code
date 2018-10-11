import pickle
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='draw annomaly score')
parser.add_argument('--dataset', type=str, default='rpt_min_network_bras_new_0811-0911-61-187-89-79_rnn_raw',
                    help='type of the dataset (ecg,bras, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--modelname', type=str, default='rpt_min_network_bras_new_0811-0911-61-187-89-79_rnn_raw-lstm-400epoch-batch72',
                    help='modelname of the dataset')

args = parser.parse_args()

#----usage:python3 draw_result.py --dataset rpt_min_network_bras_new_0811-0911-61-187-89-79_rnn_raw --modelname rpt_min_network_bras_new_0811-0911-61-187-89-79_rnn_raw-lstm-400epoch-batch72

savePath="/home/qhw/python/RNN-Time-series-Anomaly-Detection/low_quality/fig_result/"

testDataPath="/home/qhw/python/RNN-Time-series-Anomaly-Detection/dataset/bras/labeled/test/"+args.dataset
onePredicatonPath="/home/qhw/python/RNN-Time-series-Anomaly-Detection/result/bras/"+args.modelname+"/oneStep_predictions"
scorePath="/home/qhw/python/RNN-Time-series-Anomaly-Detection/result/bras/"+args.modelname+"/score"

testDataList=[]
onePredictList=[]
scoreList=[]

targetAnnomyArray=np.array([1,2])

figTitleList=['Anomaly Detection on watch user number','Anomaly Detection on low quality user number','Anomaly Detection on low quality ratio']

figNameList=['watch_num','low_num','low_ratio']
save_dir = Path(savePath).with_suffix('').joinpath(args.modelname)
save_dir.mkdir(parents=True,exist_ok=True)
            
def loadData(path):
		DataF=open(path+".pkl",'rb')
		DataList=pickle.load(DataF)
		DataF.close()
		
		return DataList

def pickTargetAnnomyArray():
    targetAnnomyList=[]
    for i in range(len(testDataList)):
        if testDataList[i][3] == 1.0:
            l3=testDataList[i][:3]
            l3.insert(0,i)
            targetAnnomyList.append(l3)
            
    return np.array(targetAnnomyList)
    		
def drawSingleFig():
    testDataArray=np.array(testDataList)
    onePredictArray=np.array(onePredictList)#.reshape(len(testDataList),-1)
    for i in range(len(testDataArray[0])-1):
        print("-----draw "+figNameList[i]+"--------")    
        plt.figure(figsize=(40,20))
        plt.plot(np.arange(1,len(testDataList)+1),testDataArray[:,i],label='Target',
                 color='black',  marker='.', linestyle='--', markersize=1, linewidth=0.5)
        plt.plot(np.arange(1,len(testDataList)+1),onePredictArray[i,:], label='1-step predictions',color='red', marker='*', linestyle='--', markersize=1, linewidth=0.5)
        plt.legend()
        plt.ylabel('Value',fontsize=30)
        plt.xlabel('Index',fontsize=30)
        plt.title(figTitleList[i], fontsize=30, fontweight='bold')
        #plt.xlim([0,len(testDataList)])
        
        plt.savefig(str(save_dir.joinpath(figNameList[i]).with_suffix('.png')))
        plt.close()    
        
def drawDoubleFigure():
    testDataArray=np.array(testDataList)
    onePredictArray=np.array(onePredictList)#.reshape(len(testDataList),-1)
    for i in range(len(testDataArray[0])-1):
        print("-----draw "+figNameList[i]+"--------")    
        plt.figure(figsize=(40,20))
        plt.plot(np.arange(1,len(testDataList)+1),testDataArray[:,i],label='Target',
                 color='black',  marker='.', linestyle='--', markersize=1, linewidth=0.5)
        plt.plot(np.arange(1,len(testDataList)+1),onePredictArray[i,:], label='1-step predictions',color='red', marker='*', linestyle='--', markersize=1, linewidth=0.5)
        plt.legend()
        plt.ylabel('Value',fontsize=30)
        plt.xlabel('Index',fontsize=30)
        plt.title(figTitleList[i], fontsize=30, fontweight='bold')
        #plt.xlim([0,len(testDataList)])
        
        plt.savefig(str(save_dir.joinpath(figNameList[i]).with_suffix('.png')))
                    
def drawFig():
    testDataArray=np.array(testDataList)
    onePredictArray=np.array(onePredictList)    
    #scoreArray=np.array(scoreList)
    for i in range(len(testDataArray[0])-1):
        print("-----draw "+figNameList[i]+"--------")
        fig, ax1 = plt.subplots(figsize=(30,20))
        ax1.plot(testDataArray[:,i],label='Target',
                 color='black',  marker='.', linestyle='--', markersize=1, linewidth=0.5)
        ax1.plot(onePredictArray[i,:], label='1-step predictions',
                 color='blue', marker='*', linestyle='--', markersize=1, linewidth=0.5)
        ax1.legend(loc='upper left')
        ax1.set_ylabel('Value',fontsize=30)
        ax1.set_xlabel('Index',fontsize=30)
        ax1.scatter(targetAnnomyArray[:,0],targetAnnomyArray[:,i+1],c = 'r',marker = 'o')  
        ax2 = ax1.twinx()
        ax2.plot(scoreList[i].numpy(), label='Anomaly scores',
                 color='red', marker='.', linestyle='--', markersize=1, linewidth=1)
        #draw annomy for target
        
        ax2.legend(loc='upper right')
        ax2.set_ylabel('anomaly score',fontsize=30)

        plt.title(figTitleList[i], fontsize=30, fontweight='bold')
        plt.tight_layout()
        plt.xlim([0,len(testDataList)])
        plt.savefig(str(save_dir.joinpath(figNameList[i]).with_suffix('.png')))
        #plt.show()
        plt.close()	
        
print('*' * 89)
print('==> load  test data......')
testDataList=loadData(testDataPath)
print('==> load  oneStep_predictions......')
onePredictList=loadData(onePredicatonPath)
print('==> load  score......')
scoreList=loadData(scorePath)
print('==> get target annomy list.....')
targetAnnomyArray=pickTargetAnnomyArray()
print('==> draw figure......')
drawFig()
print('************************complated********************************************************')
