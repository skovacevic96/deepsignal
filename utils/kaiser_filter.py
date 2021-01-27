import numpy as np

_i0A = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
    ]

_i0B = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
]

def chbevl(x, vals):
    b0 = vals[0]
    b1 = 0.0

    for i in range(1, len(vals)):
        b2 = b1
        b1 = b0
        b0 = x*b1 - b2 + vals[i]

    return 0.5*(b0 - b2)

def i0(x):
  x = np.abs(x)

  return np.exp(x) * np.piecewise(x, [x<=8.0], [lambda x1: chbevl(x1/2.0-2, _i0A), lambda x1: chbevl(32.0/x1 - 2.0, _i0B) / np.sqrt(x1)])
  
def kaiser_window(N, beta):

  n = np.arange(0, N)
  alpha = (N - 1) / 2.0
  return i0(beta * np.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) / i0(beta)

def kaiser_beta(a):
  if a > 50:
      beta = 0.1102 * (a - 8.7)
  elif a > 21:
      beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
  else:
      beta = 0.0
  
  return beta
  
def kaiser_parameters(ripple, width):
  '''
  ripple - Both passband and stopband ripple strength in dB. 
  width - Difference between fs (stopband frequency) i fp (passband frequency). Normalized so that 1 corresponds to pi radians / sample. That is, the frequency is expressed as a fraction of the Nyquist frequency.
  '''

  a = abs(ripple)
  
  beta = kaiser_beta(a)
  
  numtaps = (a - 7.95) / 2.285 / (np.pi * width) + 1

  return int(np.ceil(numtaps)), beta
  
def lowpass_kaiser_fir_filter(rate=16000, cutoff_freq=4000, width=400, attenuation=65):
  '''
  rate - Signal sampling rate.
  cuttof_freq - Filter cutoff frequency in Hz.
  width - Difference between fs (stopband frequency) i fp (passband frequency) in Hz.
  attenuation - Signal attenuation in the stopband, given in dB.
  Returns: h(n) - impulse response of lowpass sinc filter with applied Kaiser window.
  '''

  nyq =  rate / 2

  cutoff_freq = cutoff_freq / nyq

  numtaps, beta = kaiser_parameters(attenuation,  float(width) / nyq)
  
  if numtaps % 2 == 0:
    numtaps += 1

  pass_zero =  True # zato sto je lowpass
  pass_nyq = False # zato sto je lowpass
  cutoff = np.hstack(([0.0]*pass_zero, cutoff_freq, [1.0]*pass_nyq))

  bands = cutoff.reshape(-1,2)

  alpha = 0.5 * (numtaps-1)
  
  m = np.arange(0, numtaps) - alpha
  h = 0
  for left, right in bands:
      h += right * np.sinc(right * m)
      h -= left * np.sinc(left * m)

  window = kaiser_window(numtaps, beta)
  h = h * window


  left, right = bands[0]
  if left == 0:
    scale_frequency = 0.0
  elif right == 1:
    scale_frequency = 1.0
  else:
    scale_frequency = 0.5 * (left + right)
  
  c = np.cos(np.pi * m * scale_frequency)
  s = np.sum(h * c)
  h /= s
  
  return h
