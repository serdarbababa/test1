import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
data = np.random.random((50, 5, 5))

fig, ax = plt.subplots()

for i in range(len(data)):
    ax.cla()
    #data[i]=data[i]>0.5

    ax.imshow(data[i],cmap='RdBu')
    ax.set_title("frame {}".format(i))
    # Note that using time.sleep does *not* work here!
    plt.pause(0.1)

    #plt.colorbar(cmap='RdBu')
print ("done")

