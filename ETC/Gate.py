import numpy as np

def AND(x1,x2):
  x = np.array([x1,x2])
  w = np.array([0.5,0.5])
  b = -0.7
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

def OR(x1,x2):
  x = np.array([x1,x2])
  w = np.array([0.5,0.5])
  b = -0.2
  tmp = np.sum(w*x) + b

  if tmp <= 0:
    return 0
  else:
    return 1

def XOR(x1, x2):    
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y

def NAND(x1,x2):
  x = np.array([x1,x2])
  w = np.array([-0.5,-0.5])
  b = 0.7
  tmp = np.sum(w*x) + b                                                                                                                   

  if tmp <= 0:
    return 0
  else:
    return 1

def NOT(x):
  if x:
    return 0
  return 1

def Full_Adder(a,b,c):
    x1 = XOR(a,b)
    x2 = AND(a,b)
    y1 = XOR(x1,c)
    y2 = AND(x1,c)
    S = y1
    Cout = OR(x2,y2)
    
    return S, Cout

def Bit_4_Full_Adder(a,b):
  S1, Cout = Full_Adder(int(a[3]),int(b[3]),0)
  S2, Cout = Full_Adder(int(a[2]),int(b[2]),Cout)
  S3, Cout = Full_Adder(int(a[1]),int(b[1]),Cout)
  S4, Cout = Full_Adder(int(a[0]),int(b[0]),Cout)
  
  return Cout, S4, S3, S2, S1