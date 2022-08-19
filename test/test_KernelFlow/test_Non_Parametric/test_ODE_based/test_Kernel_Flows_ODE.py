import numpy as np
from KernelFlow import KernelFlowsNP_ODE

def test_G_calc():
    mu = np.array([4.0])
    kernel_name = "RBF"
    KF_rbf = KernelFlowsNP_ODE(kernel_name, mu)
    x = np.linspace(-np.pi, np.pi, 50).reshape(-1,1)
    y = np.sin(x)
    batch_indices = np.arange(0, 50, 2)
    sample_indices = np.arange(1, 25, 2)
    answer = KF_rbf.G(t= None, X=x, Y=y, batch_indices=batch_indices, not_batch=None, sample_indices=sample_indices)
    actual = np.array([ 0.57577142 , 0.12352084 ,-0.20159232 ,-0.41619602 ,-0.536734   ,-0.5792445,
                        -0.55915152, -0.49107184, -0.38864059, -0.26435791, -0.12945906,  0.00619032,
                        0.13417212 , 0.24756612 , 0.34097189 , 0.41050246 , 0.4537498  , 0.46972325,
                        0.45876187 , 0.42242263 , 0.36334653 , 0.28510493 , 0.19202917 , 0.08902636,
                        -0.01861524, -0.12542776, -0.2259687 , -0.3150254 , -0.38781573, -0.44018095,
                        -0.46876774, -0.47119567, -0.4462074 , -0.39379868, -0.31532574, -0.21358806,
                        -0.09288485,  0.04095595,  0.18057689,  0.31711883,  0.44027703,  0.53838501,
                        0.59852497,  0.60666339,  0.54780965,  0.40619562,  0.16547341, -0.19107156,
                        -0.68029542, -1.31896183])
    assert np.allclose(answer, actual)