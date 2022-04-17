import numpy as np
import scipy.linalg as la
from scipy.linalg import solve
import sys

"""
Program to generate decimal matrix and calculate L-Frobenius norm of the matrix
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


# Driver Code
if __name__ == "__main__":
    # Input Matrix Dimensions
    ip_flag = True
    retry_count = 0
    max_retry = 3
    while (ip_flag):
        rows = int(input("Enter row: "))
        cols = int(input("Enter column: "))

        # m > n check
        if (rows < cols):
            print(f"Error: Please enter row > col...Attempt:{retry_count + 1}....Max allowed retry:{max_retry}")
            retry_count += 1
            if (retry_count == max_retry):
                print("Exceeded max retry... Exit")
                exit(1)
        else:
            ip_flag = False

    # Input data format r.dddd
    d =4

    # Random input coefficient matrix
    a_temp = np.random.rand(rows, cols) * 10
    A_mat = np.around(a_temp, d)

    # ds rounding for output and intermediate calculations
    ds =4

    print(f"\nInput matrix with dimensions:{rows} x {cols}")
    print("-----------------------------------")
    print(A_mat)
    get_norm_all(np.copy(A_mat),rows,cols,ds)

