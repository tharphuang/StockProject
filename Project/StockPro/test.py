import numpy as np
data = np.array([[9,2], [3,7], [7,8], [3,5], [1,9]])
idex=np.lexsort([-1*data[:,1], data[:,0]])
sorted_data = data[idex,:]
print(sorted_data)