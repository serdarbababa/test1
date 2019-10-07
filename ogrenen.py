from modules.Components import Abstract, Context1, Context2, Actuator, Spektron
import numpy as np
from pywt import wavedec, families
import pywt
import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/serdar/Documents/python/test1/SpikingNeuralNetwork")

plt.rcParams['figure.figsize'] = [15, 10]

#initialize Spektron
spek= Spektron()



#girdi = np.array(input_data)  # np.array([1,2,3,4,5,6,7,8])*1
girdi = np.array([1,2,3,4,5,6,7,8])*1

samples = 128
subsample = 128

girdi = np.linspace(0,2*np.pi,samples)
#girdi = 1*np.sin(2*girdi)+np.sin(5*(girdi+np.pi/3))

girdi = 1*np.sin(2*girdi)

girdi1 = np.linspace(0,2*np.pi,samples)
girdi1 = 3*np.sin(2*girdi1)

girdi2 = np.linspace(0,2*np.pi,samples)
girdi2 = 3*np.sin(4*girdi2)

print("original")
#plt.plot(girdi)
#plt.plot(girdi1)
#plt.plot(girdi2)
#plt.legend([ "Two Periods","Intense Two P", "Four Periods"])
#plt.title("Sample Signals")
#plt.show()


f='db1'

coeff = wavedec(girdi, f, level=int(np.log2(len(girdi))))
coeff = spek.v.mergeList(coeff)

#plt.plot(girdi)
#plt.plot(coeff)


f='db2'

coeff = wavedec(girdi, f, level=int(np.log2(len(girdi))))
coeff = spek.v.mergeList(coeff)
#plt.plot(coeff)

#plt.show()


#todo Growing structure
#todo Learning as growing
#todo



from SpikingNeuralNetwork.snn.neurons.LeakyIntegrateAndFireNeuron import LeakyIntegrateAndFireNeuron
from SpikingNeuralNetwork.snn.learning.stdp import STDP
from SpikingNeuralNetwork.snn.network.snn import SNN

a= LeakyIntegrateAndFireNeuron()
a.calculate_potential()
