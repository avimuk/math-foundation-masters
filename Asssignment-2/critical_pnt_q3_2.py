import warnings

import sympy as sym
import numpy as np
from sympy import *
import scipy.optimize as optimize
init_printing( use_latex='mathjax' )  # use pretty math output
from scipy.optimize import fsolve
#warnings.filterwarnings('error')

def poly(x,y,n,p):
    ph_len = len(p)
    poly_list = []
    counter = 0
    str1 = ""
    for nc in range(n,-1,-1):
        for i in range(n+1):
            if nc-i >= 0:
                if p[counter] == "0":
                    coeff = "3"
                else :
                    coeff = p[counter]
                if counter == ph_len -1 :

                    poly_list.append(coeff + "*" + x + "^" + str(nc - i) + "*" + y + "^" + str(i))
                elif  counter % 2 == 0:
                    poly_list.append(coeff+ "*"+ x+ "^"+ str(nc - i)+ "*"+ y+ "^"+ str(i))
                    poly_list.append(" - ")
                else :
                    poly_list.append(coeff+ "*"+ x+ "^"+ str(nc - i)+ "*"+ y+ "^"+ str(i) )
                    poly_list.append(" + ")
                counter += 1
    return str1.join(poly_list)

def myFunction(z, *args):
    x = z[0]
    y = z[1]
    F = np.empty((2))
    F[0] = eval(args[0])
    F[1] = eval(args[1])
    return F


def call_critical_point(x_der, y_der):
    zGuess = np.array([1, 1])
    cp=""
    cn=""
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            z = fsolve(myFunction, zGuess, args=(x_der, y_der))
            cp = z
        except RuntimeWarning as  e:
            print(f"\n[Warning] Opps ! fsolve is unable to optimize solution for cn(1,1)..Try another number")
            exit(1)
    print(f"\nfsolve successfully calculated cp(1,1)")
    zGuess = np.array([-1, -1])
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            z = fsolve(myFunction, zGuess, args=(x_der, y_der))
            cn = z
        except RuntimeWarning as  e:
            print(f"\n[Warning]Opps ! fsolve is unable to optimize solution for cn(-1,-1)..Try another number")
            exit(1)
    print(f"fsolve sucessfully calculated cn(-1,-1)")
    return cp,cn

def Get_Critical_Point(p):
    x, y = symbols('x y')
    x_derivative = diff(p,x)
    #print("The x derivative is :")
    #print(x_derivative)

    x_eqtn = str(x_derivative)

    y_derivative = diff(p,y)
    #print("The y derivative is :")
    #print(y_derivative)

    y_eqtn = str(y_derivative)

    cpp,cpn = call_critical_point(x_eqtn,y_eqtn)

    return cpp, cpn


if __name__ == "__main__":
    ip_flag = True
    retry_count = 0
    max_retry = 3
    while (ip_flag):
        p = list(input('Enter your phone number(10 digit): '))
        if (len(p) !=10):
            print(f"Error: Please enter valid 10 digit phone number...Attempt:{retry_count + 1}....Max allowed retry:{max_retry}")
            retry_count += 1
            if (retry_count == max_retry):
                print("Exceeded max retry... Exit")
                exit(1)
        else:
            ip_flag = False


    poly_list = poly("x","y",3,p)
    print("\nThe generated polynomial is :")
    print("----------------------------")
    print(poly_list)
    cpp, cpn = Get_Critical_Point(poly_list)
    print("\nThe Critical Points are")
    print("----------------------------")
    print(cpp)
    print(cpn)