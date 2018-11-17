import h5py
import numpy as np
import matplotlib.pyplot as plt


f = h5py.File('results.h5')

params = f['optimization']
params = params['params']

size_p = params.size

params0 = np.empty(size_p)

for i in range(size_p):
    params0[i] = params[i]


pts = np.load('initialpts.npz')

x = pts['x']
y = pts['y']
inds = pts['inds']

ff = plt.figure()
ax = ff.add_subplot(111)
ax.plot(x,y,'-o')

x_new = np.copy(x)
y_new = np.copy(y)

print x.size, y.size

N = params0.size/2
for i in range(N):
    x_new[inds[i]] = x[inds[i]] + params0[i]
    y_new[inds[i]] = y[inds[i]] + params0[i+N]

ax.plot(x_new,y_new,'-o')
plt.axis('equal')
plt.show()

np.savez('optResult', x=x_new, y=y_new, inds = inds)


h5file = h5py.File("optResult.h5","w")

dset1 = h5file.create_dataset("x",data=x_new)
dset2 = h5file.create_dataset("y",data=y_new)
dset3 = h5file.create_dataset("inds",data=inds)
