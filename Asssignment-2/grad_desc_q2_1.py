import numpy as np
import scipy.linalg as la
from scipy.linalg import solve
import sys

"""
This program:
    1. Uses last 4 digits of your mobile number(n1n2n3n4) and generate matrix A of size (n1n2 × n3n4)
    2. It also calculates the L-Frobenius norm of the generated matrix
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

def get_norm_all(A,m,n,ds):
    """
      Calculate all norms

      input: A is an m x n  coefficient matrix
             m # of rows
             n # of cols
             ds decimal point
      output: L-Frobenius normal form value
      """

    #rows m and columns n, for square matrix its same

    #Frobenius norm
    f = 0
    for i in np.arange(m):
        for j in np.arange(n):
            f = f + np.sum(np.power(np.abs(A[i, j]), 2))

    r=round_s(float(np.sqrt(f)),ds)
    print(f"\nL-Frobenius norm for the matrix is: {r}\n")

    #print(np.sqrt(np.sum(np.abs(A) ** 2)))
    #print(np.linalg.norm(A, 'fro'))
    """
    print(f"Other optional norms for verification")

    #L2
    r= round_s(float(np.sqrt(np.max(np.linalg.eigvals(np.inner(A, A))))),ds)
    print(f"L2 norm is: {r}")

    #print(np.linalg.norm(A, 2))

    #L-infinity
    rowsums = []
    for i in np.arange(m):
        v = np.sum(np.absolute(A[i, :]))
        rowsums.append(v)
    r=round_s(float(np.max(rowsums)),ds)
    print(f"L-infinity norm is: {r}")

    #print(np.max(np.sum(np.abs(A), axis=1)))
    #print(np.linalg.norm(A, np.inf))

    #L1
    columns = []
    for i in np.arange(n):
        v = np.sum(np.abs(A[:, i]))
        columns.append(v)
    r=round_s(float(np.max(columns)),ds)
    print(f"L1 norm is: {r}")

    #print(np.max(np.sum(np.abs(A), axis=0)))
    #print(np.linalg.norm(A, 1))
    """

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

    rows = ph_n[:2]
    cols = ph_n[2:]

    if (int(rows) == 0 or int(cols) == 0):
        print("Error: Zero matrix size not 010allowed... Re-run with non-zero matrix size")
        exit(1)

    rows = rows.replace("0","3")
    cols = cols.replace("0","3")

    rows = int(rows)
    cols = int(cols)

    #if(rows ==0 or cols==0):
    #    print("Error: Zero matrix size not allowed... Re-run with non-zero matrix size")
    #    exit(1)
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
    ds = 5
    get_norm_all(A_mat, rows, cols, ds)
