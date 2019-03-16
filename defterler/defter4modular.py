import numpy as np
from pywt import wavedec
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import  graphviz_layout
import networkx as nx
import pandas as pd
import seaborn as sns

plt.rcParams['figure.figsize'] = [25, 20]     


def genData(param, show = False):
	a=[]
	if param[0]=="normal":
		mu, sigma, s = param[1],   param[2], param[3]
		a= np.random.normal(mu, sigma, size=s)
	elif param[0]=='uniform':
		mi, ma,s=param[1],   param[2], param[3]
		a= np.random.uniform(mi, ma, s)
	elif param[0]=="poisson":
		rate,s=param[1],   param[2]
		a = np.random.poisson(rate, s)
	if(show):
		count, bins, ignored = plt.hist(s, 14, density=True)
	return a

def genSample(signalCount):
    print("generate sample data")
    signals = [] 
    for i in range(signalCount):
        a = genData(["normal", 100,100,8])
        #print(a)
        sig = [] 
        for j in range(8):
            sig.append(int(a[j]))
        signals.append(sig)
    for i in range(signalCount):
        print(signals[i])
    return signals

def getSamplePredef():
    print("return sample data")
    signals = [[105, 220, 23, 99, 266, 190, 37, 5],
                [334, 174, 134, -7, 19, 155, 93, 89],
                [72, 96, 102, 151, -14, 171, 127, 127],
                [151, 38, 283, 204, 232, 141, 121, 47],
                [157, -60, 54, 54, 69, -27, -14, 101],
                [0, 113, 74, 176, 68, 322, 135, 367],
                [56, 114, 126, 181, 93, 41, 118, 76],
                [164, 200, 351, 51, 36, 163, 298, -5],
                [140, 124, 99, 34, -46, -5, 240, 136],
                [113, 58, 130, 123, 171, 143, 109, 17],
                [-8, 299, 65, 62, 130, 146, -43, 23],
                [-96, 212, 56, 150, -55, 150, 151, 70],
                [-22, 148, 219, 62, 108, 136, 198, 126],
                [220, 84, 165, 167, 1, 227, 15, 144],
                [0, 135, 165, 64, 100, 224, 244, 140],
                [211, 183, -161, 65, 33, 257, -16, 112]]
    return signals


def mergeList(input_data, verbose= False):
    if verbose:
        print("merge data")
    merged_list = []
    for l in input_data:
        merged_list += list(l)
    return merged_list


def listToPandasDF(input_data):
    df = pd.DataFrame(input_data)
    return df

def getWaveletCoefs(input_data):
    coefs = [] 
    for i in range(len(input_data)):
        girdi = np.array(input_data[i]) #np.array([1,2,3,4,5,6,7,8])*1
        coeff = wavedec(girdi, 'haar', level=int(np.log2(len(girdi))))
        coefs.append(mergeList(coeff))
    return coefs

def plotCorrelation(input_data_frame):
    Var_Corr = input_data_frame.corr()
    # plot the heatmap and annotation on it
    sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
    plt.show()
    # Basic correlogram
    sns_plot = sns.pairplot(input_data_frame)
    plt.show()

def generateOperations(operations_count, Test = False, verbose= False):
    energy = 0 
    all=[]
    for i in range(operations_count):
        if(verbose):print(i, end="\t")
        a = genData(["uniform", 0,10,4])    
        a= [int(x) for x in a]
        if(True):#Test):
            a[3]=0
            
        all.append(a)
        #print(int(a[0]) , int(a[1])%4, int(a[2]) ,int(a[3])%2)
        if(int(a[1])%4==0 and int(a[3])%2 == 0  ):
            if(verbose):
                print(int(a[0]) , "+", int(a[2]) ,"=", int(a[0])  + int(a[2]))
        elif(int(a[1])%4==0 and int(a[3])%2 == 1 ):
            if(verbose):print(int(a[0]) , "+", int(a[2]) ,"= ?")
        elif(int(a[1])%4==1 and int(a[3])%2 == 0  ):
            if(int(a[2])<int(a[0])):
                if(verbose):print(int(a[0]) , "/", int(a[2]) ,"=", int(int(a[0])  - int(a[2])))
            else:
                if(verbose):print(int(a[0]) , "/", int(a[2]) ,"=", int(int(a[2])  - int(a[0])))          
        elif(int(a[1])%4==1 and int(a[3])%2 == 1  ):
            if(verbose):print(int(a[0]) , "-", int(a[2]) ,"= ?")        
        
        elif(int(a[1])%4==2 and int(a[3])%2 == 0  ):
            if(verbose):print(int(a[0]) , "*", int(a[2]) ,"=", int(a[0])  * int(a[2]))
        elif(int(a[1])%4==2 and int(a[3])%2 == 1 ):
            if(verbose):print(int(a[0]) , "*", int(a[2]) ,"= ?")
        
        elif(int(a[1])%4==3 and int(a[3])%2 == 0  ):
            if(int(a[2])!=0):
                if(verbose):print(int(a[0]) , "/", int(a[2]) ,"=", int(int(a[0])  / int(a[2])))
            elif(int(a[0])!=0):
                if(verbose):print(int(a[0]) , "/", int(a[2]) ,"=", int(int(a[2])  / int(a[0])))    
        
        elif(int(a[1])%4==3 and int(a[3])%2 == 1  ):
            if(verbose):print(int(a[0]) , "/", int(a[2]) ,"= ?")
    return all


def decodeOperations(all):
    for i in range(1):
        print(i, end="\t")
        a = all[i]
        #print(int(a[0]) , int(a[1])%4, int(a[2]) ,int(a[3])%2)
        if(int(a[1])%4==0 and int(a[3])%2 == 0  ):
            print(int(a[0]) , "+", int(a[2]) ,"=", int(a[0])  + int(a[2]))
        elif(int(a[1])%4==0 and int(a[3])%2 == 1 ):
            print(int(a[0]) , "+", int(a[2]) ,"= ?")
        elif(int(a[1])%4==1 and int(a[3])%2 == 0  ):
            if(int(a[2])<int(a[0])):
                print(int(a[0]) , "/", int(a[2]) ,"=", int(int(a[0])  - int(a[2])))
            else:
                print(int(a[0]) , "/", int(a[2]) ,"=", int(int(a[2])  - int(a[0])))          
        elif(int(a[1])%4==1 and int(a[3])%2 == 1  ):
            print(int(a[0]) , "-", int(a[2]) ,"= ?")        
        elif(int(a[1])%4==2 and int(a[3])%2 == 0  ):
            print(int(a[0]) , "*", int(a[2]) ,"=", int(a[0])  * int(a[2]))
        elif(int(a[1])%4==2 and int(a[3])%2 == 1 ):
            print(int(a[0]) , "*", int(a[2]) ,"= ?")
        elif(int(a[1])%4==3 and int(a[3])%2 == 0  ):
            if(int(a[2])>0):
                print(int(a[0]) , "/", int(a[2]) ,"=", int(int(a[0])  / int(a[2])))
            else:
                print(int(a[0]) , "/", int(a[2]) ,"=", int(int(a[2])  / int(a[0])))    
        elif(int(a[1])%4==3 and int(a[3])%2 == 1  ):
            print(int(a[0]) , "/", int(a[2]) ,"= ?")


def quantize(input_data, len_of_data, verbose = False):
    borders = [-200,-100,-50,0, 50, 100, 200]
    qsignals = [] 

    for i in range(len(input_data)):
        sig = [] 
        if(verbose):
            print(input_data[i])
        for j in range(int(len(input_data[i]))):
            output = 7
            for k in range(7):
                if( input_data[i][j] < borders[k]):
                    output = k
                    break
            if verbose:
                print(output, end = " ")
            sig.append(k)
        if verbose:
            print()
        qsignals.append(sig)
    return qsignals


def decodeOperationsDeeper(input_data, len_data, qsignals, verbose = False):
    inputs = [] 
    for i in range(len_data):

        if (verbose):print(i, end="\t")
        a = input_data[i]
        if(verbose):print(int(a[0]) , int(a[1])%4, int(a[2]) ,int(a[3])%2)

        if(int(a[1])%4==0 and int(a[3])%2 == 0  ):
            if(int(a[0])  + int(a[2]) < 10):
                if(verbose):
                    print( qsignals[int(a[0])] , "+", qsignals[int(a[2])] ,"=", qsignals[int(a[0])  + int(a[2])])            
                    print( qsignals[int(a[0])] , qsignals[10], qsignals[int(a[2])] ,"=", qsignals[int(a[0])  + int(a[2])])            
                inputs.append([ qsignals[int(a[0])] , qsignals[10], qsignals[int(a[2])] , qsignals[int(a[0])  + int(a[2])]])            
        elif(int(a[1])%4==0 and int(a[3])%2 == 1 ):
            if(verbose):
                print( qsignals[int(a[0])] , "+", qsignals[int(a[2])] ,"=", qsignals[14])
                print( qsignals[int(a[0])] , qsignals[10], qsignals[int(a[2])] ,"=", qsignals[14])
            inputs.append( [qsignals[int(a[0])] , qsignals[10], qsignals[int(a[2])] , qsignals[14]])

        elif(int(a[1])%4==1 and int(a[3])%2 == 0  ):
            if(int(a[2])<int(a[0])):
                if(verbose):
                    print( qsignals[int(a[0])] , "-", qsignals[int(a[2])] ,"=", qsignals[int(a[0])  - int(a[2])])
                    print( qsignals[int(a[0])] , qsignals[11], qsignals[int(a[2])] ,"=", qsignals[int(a[0])  - int(a[2])])
                inputs.append( [qsignals[int(a[0])] , qsignals[11], qsignals[int(a[2])] , qsignals[int(a[0])  - int(a[2])]])
            else:
                if(verbose):
                    print( qsignals[int(a[0])] , "-", qsignals[int(a[2])] ,"=", qsignals[int(a[2])  - int(a[0])])   
                    print( qsignals[int(a[0])] , qsignals[11], qsignals[int(a[2])] ,"=", qsignals[int(a[2])  - int(a[0])])   
                inputs.append( [qsignals[int(a[0])] , qsignals[11], qsignals[int(a[2])] , qsignals[int(a[2])  - int(a[0])]])

        elif(int(a[1])%4==1 and int(a[3])%2 == 1  ):
            if(verbose):
                print( qsignals[int(a[0])] , "-", qsignals[int(a[2])] ,"=", qsignals[14])
                print( qsignals[int(a[0])] , qsignals[11], qsignals[int(a[2])] ,"=", qsignals[14])
            inputs.append( [qsignals[int(a[0])] , qsignals[11], qsignals[int(a[2])] , qsignals[14]])        

        elif(int(a[1])%4==2 and int(a[3])%2 == 0  ):
            if(int(a[0])  * int(a[2]) < 10):
                if(verbose):
                    print( qsignals[int(a[0])] , "*", qsignals[int(a[2])] ,"=", qsignals[int(a[0])  * int(a[2])])
                    print( qsignals[int(a[0])] , qsignals[12], qsignals[int(a[2])] ,"=", qsignals[int(a[0])  * int(a[2])])
                inputs.append( [qsignals[int(a[0])] , qsignals[12], qsignals[int(a[2])] , qsignals[int(a[0])  * int(a[2])]])
        elif(int(a[1])%4==2 and int(a[3])%2 == 1 ):
            if(verbose):
                print( qsignals[int(a[0])] , "*", qsignals[int(a[2])] ,"=", qsignals[14])
                print( qsignals[int(a[0])] ,  qsignals[12], qsignals[int(a[2])] ,"=", qsignals[14])
            inputs.append( [qsignals[int(a[0])] ,  qsignals[12], qsignals[int(a[2])] , qsignals[14]])
        elif(int(a[1])%4==3 and int(a[3])%2 == 0  ):
            if(int(a[2])>0):            
                if(verbose):
                    print( qsignals[int(a[0])] , "/", qsignals[int(a[2])] ,"=", qsignals[int( int(a[0])  /int(a[2]))])
                    print( qsignals[int(a[0])] , qsignals[13], qsignals[int(a[2])] ,"=", qsignals[int(int(a[0])  /int(a[2]))])
                inputs.append( [qsignals[int(a[0])] , qsignals[13], qsignals[int(a[2])] , qsignals[ int(int(a[0])  /int(a[2]))]])

        elif(int(a[1])%4==3 and int(a[3])%2 == 1  ):
            if(verbose):
                print( qsignals[int(a[0])] , "/", qsignals[int(a[2])] ,"=", qsignals[14])
                print( qsignals[int(a[0])] , qsignals[13], qsignals[int(a[2])] ,"=", qsignals[14])
            inputs.append( [qsignals[int(a[0])] , qsignals[13], qsignals[int(a[2])] , qsignals[14]])
    return inputs



def initTree():
    GG=nx.DiGraph()    
    GG.add_node(0, k=999,cc=1, id = -1,food =0)    
    return GG,1

def train_tree(input_data, GG, counter , WL, overlap):
    plt.rcParams.update({'font.size': 22})
    data1= input_data  
    print(data1)
    poz = 0
    #print(len(data1)/WL)    
    step = WL-overlap

    for i in range(0,len(data1)-step+1, step):
        #if(data1[i:i+step]==[0,0,1,0] or  data1[i:i+step]==[1,0,0,0] ):
        #    //print(str(i)+ " food")
        poz=0
        for j in range(WL):
            # data
            d=data1[i + j]
            #print(d, end=' ')
            #print (d)
            # neighbours 
            nei= list(GG.neighbors(poz))       
            if len(nei)==0:
                #print (counter, poz, data1[i: i+step])
                GG.add_node(counter,k=d, cc=1, id = -1,food =0)    
                GG.add_edge(poz,counter)
                poz=counter
                counter +=1
            else:
                k=-1
                for n in nei:
                    if(GG.node[n]['k']==d):
                        k=n
                        break
                if(k>=0):
                    poz=k
                    GG.node[k]['cc'] = GG.node[k]['cc'] + 1
                else:
                    GG.add_node(counter,k=d,cc=1, id = -1,food =0)    
                    GG.add_edge(poz,counter)
                    poz=counter
                    counter += 1
    finalNodes = [] 
    for i in range(1, counter):
        yol =  nx.shortest_path(GG,0,i)
        if(len(yol) >WL):
            GG.node[yol[-1]]['id']=len(finalNodes)
            finalNodes.append(yol[-1])
    return GG, counter


def plotGraph(GG, WL, counter):
    plt.rcParams['figure.figsize'] = [15, 10]        
    labels=dict((n,d['k']) for n,d in GG.nodes(data=True))   
    #pos=nx.graphviz_layout(GG, prog='dot')
    pos =graphviz_layout(GG, prog='dot')
    #nx.spring_layout(GG)

    plt.title("node values")
    nx.draw_networkx(GG,  pos=pos, arrows=True, with_labels=True, labels=labels )
    plt.show()

    plt.title("node ids")
    nx.draw_networkx(GG,  pos=pos, arrows=True, with_labels=True )
    plt.show()

    plt.title("node frequency")
    labels=dict((n,d['cc']) for n,d in GG.nodes(data=True))   
    nx.draw_networkx(GG,  pos=pos, arrows=True, with_labels=True, labels=labels )
    plt.show()

    

    plt.title("final nodes ids")
    labels=dict((n,d['id']) for n,d in GG.nodes(data=True))   
    nx.draw_networkx(GG,  pos=pos, arrows=True, with_labels=True, labels=labels )
    plt.show()    


def buildAbstractTree(GG, symbols, ops,counter):
    #print(b)
    decodedOps=decodeOperationsDeeper(input_data=ops,len_data=20, qsignals= quantize(symbols,len_of_data=4) )
    
    GG,counter = train_tree( mergeList(mergeList(decodedOps)), GG, counter , WL, overlap)
    plotGraph(GG,WL,counter)
    return GG, counter

def buildContextTree(GG, ops,counter,WL, overlap=0):
    #print(b)
    #decodedOps=decodeOperationsDeeper(input_data=ops,len_data=20, qsignals= quantize(symbols,len_of_data=4) )
    
    GG,counter = train_tree( ops, GG, counter , WL, overlap)
    plotGraph(GG,WL,counter)
    return GG, counter


def getBranchId(branch,GG, counter , WL, overlap):    
    ids = []
    plt.rcParams.update({'font.size': 22})
    data1= branch  
    
    poz = 0
    #print(len(data1)/WL)    
    step = WL-overlap
    
    for i in range(0,len(data1)-step+1, step):
        k=0
        poz=0
        #print()
        for j in range(WL):
            # data
            d=data1[i + j]
            
            # neighbours 
            nei= list(GG.neighbors(poz))       
            if len(nei)==0:
                k=-1
                break
            else:
                k=-1
                for n in nei:
                    if(GG.node[n]['k']==d):
                        k=n
                        break
                if(k>=0):
                    poz=k                    
                    #GG.node[k]['cc'] = GG.node[k]['cc'] + 1
                else:
                    k=-1
                    break
        #ids.append(k)
        
        #ids.append(GG.node[k]['id'])
        if(k>0):
            ids.append(GG.node[k]['id'])
        else:
            ids.append(-1)
    return ids


symbols= genSample(16)
symbols= ( getWaveletCoefs(symbols))
print()
for i in range(len(symbols)):
    #print(symbols[i])
    print(i,[ int(j) for j in symbols[i]])
#a= getSamplePredef()



