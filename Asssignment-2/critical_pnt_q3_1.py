import sympy as sym
import numpy as np
from sympy import *
import scipy.optimize as optimize
init_printing( use_latex='mathjax' )  # use pretty math output
from scipy.optimize import fsolve

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
    print("---------------------------- :")
    print(poly_list)
