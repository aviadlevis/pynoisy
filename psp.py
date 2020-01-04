#

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

def psp(i,j,dk):
    n = dk.shape[0]
    no2 = int(n/2)
    kx = (i + no2)%n - no2
    ky = (j + no2)%n - no2
    k = np.sqrt(kx*kx + ky*ky)

    # now bin 
    Dk = 1.0
    kb = np.arange(Dk/2.,no2-Dk/2,Dk)
    pb = kb*0.
    nb = kb*0.
    for ii in range(0,n):
        for jj in range(0,n):
            b = int(k[ii,jj]/Dk) 
            if (b < kb.shape[0]):
                p = np.abs(dk[ii,jj])**2
                pb[b] = pb[b] + p
                nb[b] = nb[b] + 1.

    for ii in range(0, kb.shape[0]):
        pb[ii] = pb[ii]/nb[ii]
            
    return(kb, pb)


# get data
i,j,d = np.loadtxt("noisy.out", unpack=True)
n = int(np.sqrt(d.shape[0]))
d = np.reshape(d, (n,n))
i = np.reshape(i, (n,n))
j = np.reshape(j, (n,n))

# plot data
plt.imshow(d)
plt.show()

# take power spectrum
dk = fft.fft2(d)

p = np.log10(np.abs(dk)*np.abs(dk))
p[0,0] = 1.e-6

# show psp
p = np.roll(p, [int(n/2), int(n/2)], axis=(0,1))  # shift peak to middle
plt.imshow(p)
plt.show()

# now show angle-averaged psp
kb, pb = psp(i,j,dk)
plt.plot(np.log10(kb), np.log10(pb))
plt.plot(np.log10(kb), -4.*np.log10(kb) + 6.)
plt.show()
