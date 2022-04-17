import numpy as np
import scipy.linalg as la
from scipy.linalg import solve
import sys
import matplotlib.pyplot as plt

"""
This program calulate minima using following logic
 
    Step 0: Generate A(co-efficient matrix RxC) and b(Constant Vector Rx1)
    Step 1: Generate x (vector Cx1)
    Step 2: Using x Gradient = A^T.AX - A^T.b 
    Step 3: Guess learning rate. Let us assume the learning rate → 0.01 and tune it further with trial and error
    Step 4: Find previous_step_size by L-2((Xk - Xk-1)) and keep precision= 0.001 and also keep a large iteration count as max_iters
    Step 5: while previous_step_size > precision and iters < max_iters, continue
			Calculate gradient descent= A^T.AX - A^T.b 
			Reduce current x by learning rate * gradient descent
			Find the y or f(x) = 1/2(L-2(Ax - b))^2 for reduced x at every step
    Step 6: When the loop breaks, we return the x where local minima found and also f(x)
"""

def round_s(n, dp=0):
    if dp == 0:
        n_5s = n + 5 * pow(10, -(dp+1))
    else:
        n_5s = n + 5 * pow(10, -(dp))
    if '.' in str(n_5s) and '-' in str(n_5s):
        return(float(str(n_5s)[0:dp+2]))
    elif '.' in str(n_5s) or '-' in str(n_5s):
        return (float(str(n_5s)[0:dp + 1]))
    else:
        return (float(str(n_5s)[0:dp]))

def get_norm_L2(A,ds):
    """
      Calculate L2 norm

      input: A is an m x n  coefficient matrix
             m # of rows
             n # of cols
             ds decimal point
      output: L-2 normal form value
      """


    #L2
    r= np.sqrt(np.max(np.linalg.eigvals(np.inner(A, A))))
    r = round_s(float(r),ds)
    #print(f"L2 norm is: {r}")
    #print(np.linalg.norm(A, 2))
    return r

def get_grad(A,b,x,ds):
    Ax = A.dot(x)
    A_trasp = A.transpose()
    gradient = (A_trasp.dot(Ax)) - (A_trasp.dot(b))
    #gradient= get_norm_L2(grd_matrix,ds)
    #print(gradient)
    return gradient

def get_Fx(A,b,x,ds):
    Ax = A.dot(x)
    A_calc1 = (Ax - b)
    l2_norm = get_norm_L2(A_calc1, ds)
    y = 1 / 2 * l2_norm ** 2
    return y

# Driver Code
if __name__ == "__main__":

    # Keep generating input matrix until LI
    gd_flag = True
    retry_count = 0
    max_retry = 3
    while (gd_flag):

        ph_n = input("Enter last 4 digit of your mobile number: ")
        if(len(ph_n) !=4):
            print(f"Error: Please enter 4 digit number...Attempt:{retry_count +1}....Max allowed retry:{max_retry}")
            retry_count += 1
            if (retry_count == max_retry):
                print("Exceeded max retry... Exit")
                exit(1)
        else:
            gd_flag=False

    rows = int(ph_n[:2])
    cols = int(ph_n[2:])

    if(rows ==0 or cols==0):
        print("Error: Zero matrix size not allowed... Re-run with non-zero matrix size")
        exit(1)
    # Input data format r.dddd
    d = 4
    """
    # [Decimal] Random input coefficient matrix
    a_temp = np.random.rand(rows, cols) * 10
    A_mat = np.around(a_temp, d) # there can random rounding of Zeros like 1.91 which is essentially 1.9100

    # [Decimal] input constant or b vector
    b_temp = np.random.rand(rows, 1) * 10
    b_mat = np.around(b_temp, 4)  # there can random rounding of Zeros like 1.91 which is essentially 1.9100
    """
    # [Integer] Random input coefficient matrix
    A_mat = np.random.randint(10, size=(rows, cols))

    # [Integer] input constant or b vector
    b_mat = np.random.randint(10, size=(rows, 1))

    print(f"\nInput matrix with size:{rows} x {cols}")
    print("-----------------------------------")
    print(A_mat)
    print(f"\nConstant vector or b vector with size:{rows} x 1")
    print("--------------------------------------------------")
    print(b_mat)
    # ds rounding for output and intermediate calculations
    ds = 4

    #Initialize x with random numbers
    cur_x = np.random.randint(10, size=(cols, 1))

    # Learning rate: This value is obtained based on trial and error
    # Test range: From 0101 up to 9999 input and with no NAN error
    rate = 0.00001

    precision = 0.001  # This tells us when to stop the algorithm
    previous_step_size = 1  # Initialize step size

    # maximum number of iterations
    # Increase if minima is not found in the range
    max_itr = 10000
    itr_cnt = 0  # iteration counter
    y_arr = []
    itr_arr = []

    y_arr.append(get_Fx(A_mat, b_mat,cur_x,ds))
    itr_arr.append(itr_cnt)

    while previous_step_size > precision:
        prev_x = cur_x
        cur_x = cur_x - rate * get_grad(A_mat,b_mat,prev_x,ds)  # Grad descent
        #cur_x = cur_x - rate * df(prev_x)
        previous_step_size = get_norm_L2((cur_x - prev_x),ds)
        previous_step_size = round_s(float(previous_step_size),ds)
        print(f"previous step size: {previous_step_size}")
        itr_cnt += 1  # iteration count
        print(f"Iteration:{itr_cnt}")
        print(f"y:{get_Fx(A_mat, b_mat,cur_x,ds)}")
        y_arr.append(get_Fx(A_mat, b_mat,cur_x,ds))
        itr_arr.append(itr_cnt)
        if(itr_cnt > max_itr):
            print("[Failure] Local minima does not occur in the given iteration...Try again")
            exit(1)

    print(f"[Success] The local minima occurs at iteration:{itr_cnt}")
    #print(f"[Success] The local minima occurs for X at :{cur_x}")
    y = get_Fx(A_mat, b_mat,cur_x,ds)
    print(f"The local minimum value:{round_s(float(y),ds)}")