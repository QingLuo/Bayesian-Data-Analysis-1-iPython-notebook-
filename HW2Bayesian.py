import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as mp

#Problem 4.1
d = norm.rvs(1, 1, size=100)
mu=1
sigma=1
n=100
w=sigma/math.sqrt(n)
#prior
mu_0=2
w_0=1
B=w**2/(w**2+w_0**2)
dmean=np.mean(d)
mu_tilde=dmean+B*(mu_0-dmean)
w_tilde=w*math.sqrt(1-B)

array=np.linspace(0.6,1.6,num=1000)
y=norm.pdf(array,mu_tilde,w_tilde)
mp.plot(array,y,'g-',lw=4)
mp.show()



#problem 4.2
prior=norm.pdf(array,mu_0,w_0)
likelihood=np.exp(-n*(array-dmean)**2/2*sigma**2)
numer=prior*likelihood
distance=array[2]-array[1]
denorm = distance * ((2*sum(numer)-numer[0]-numer[-1])/2)
posterior = numer/denorm
mp.plot(array,posterior,'r-',array,y,'b--',lw=2)
mp.show()

#problem 4.3
ytrapz=np.trapz(numer, x=array, dx=1.0, axis= -1)
from numpy import testing
from numpy.testing import assert_approx_equal 
from numpy.testing import assert_array_almost_equal

def test():
	assert_approx_equal(denorm,ytrapz,significant=3)



def test2():
	assert_array_almost_equal(posterior,y,decimal=2)





