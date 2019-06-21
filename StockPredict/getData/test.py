import numpy as np
a=[[1],[2],[3],[4],[5]]

b=[]
print(a[2][0])
for i in range(len(a)):
    b.append(a[i][0])
print(b)
mae = np.average(np.abs(np.array(b[0:4]) - np.array(b[1:5])))
print(mae)