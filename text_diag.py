import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib
from scipy import integrate
from scipy.integrate import odeint

mu_a=1
mu_b=1
m_aa = -1
m_bb = -1
m_ab = -0.5
m_ba = -0.5


mu = np.array([mu_a,mu_b])
M = np.array([[m_aa,m_ab],[m_ba,m_bb]])