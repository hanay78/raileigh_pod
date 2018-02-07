from derivatives import coeficients
import numpy as np

M=2

#alpha=np.array([0,1,2,3])
#alpha=np.array([0,1,-1,2,-2])
#alpha=np.array([0,1,-1])
alpha=np.array([0,-1,-2])
#alpha=np.array([0,1,-1,2,3])
#alpha=np.array([0,1,-1,2,-2,3,-3,4,-4])
#alpha=np.array([0,-1,-2])
#alpha=np.array([0,1,-1,2,-2,3,-3,4,-4])
#alpha=np.array([-4,-3,-2,-1,0,1,2,3,4])
#alpha=np.array([0,1,2,3,4])
#alpha=np.array([0,0.025,-0.0187,0.05,-0.025])
#alpha=np.array([0,0.0187,-0.025,0.025,-0.05])
delta=coeficients(M,alpha)
print delta[2,2]
