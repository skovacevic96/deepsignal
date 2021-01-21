from math import pi, sin, cos, sqrt
from cmath import exp

def __bwsk(k, n):
  # Returns k-th pole s_k of Butterworth transfer
  # function in S-domain. Note that omega_c
  # is not taken into account here
  arg = pi * (2 * k + n - 1) / (2 * n)
  return complex(cos(arg), sin(arg))

def __bwj(k, n):
  # Returns (s - s_k) * H(s), where
  # H(s) - BW transfer function
  # s_k  - k-th pole of H(s)
  res = complex(1, 0)
  for m in range(1, n + 1):
    if (m == k):
      continue
    else:
      res /= (bwsk(k, n) - bwsk(m, n))
  return res

def bwh(n=16, fc=400, fs=16e3, length=25):
  # Returns h(t) - BW transfer function in t-domain.
  # length is in ms.
  omegaC = 2*pi*fc
  dt = 1/fs
  number_of_samples = int(fs*length/1000)
  result = []
  for x in range(number_of_samples):
    res = complex(0, 0)
    if x >= 0:
      for k in range(1, n + 1):
        res += (exp(omegaC*x*dt/sqrt(2)*bwsk(k, n)) * bwj(k, n))
    result.append((res).real)
  return result
