import numpy as np
import scipy.linalg as la
from scipy.linalg import solve
import sys

"""
Program to:
    1. Generate the orthogonal matrix QR from a matrix A by performing  the Gram-Schmidt orthogonalization method
    2. Calculate L-Frobenius norm for (A - QR)
    3. Also print addition, substration, multiplication and deletion count
"""


def round_s(n, dp=0):
    if dp == 0:
        n_5s = n + 5 * pow(10, -(dp + 1))
    else:
        n_5s = n + 5 * pow(10, -(dp))
    if '.' in str(n_5s) and '-' in str(n_5s):
        return (float(str(n_5s)[0:dp + 2]))
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
    print(f"{r}\n")

    #print(np.sqrt(np.sum(np.abs(A) ** 2)))
    #print(np.linalg.norm(A, 'fro'))

    print(f"Other optional norms for verification")

    #L2
    r= np.sqrt(np.max(np.linalg.eigvals(np.inner(A, A))))
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


def check_gs(A, m, n, ds):
    """
      Eligibility for GS: Checks if matrix is LI via full column rank

      input: A is an m x n  coefficient matrix
             m # of rows
             n # of cols
             ds decimal point
      output: 0- true , 1 - false
      """

    # rows m and columns n, for square matrix its same
    rank = np.linalg.matrix_rank(A)
    if (rank == n):
        print(f"All column vectors are LI. Trivial solution(rank=unknown)")
        return 0
    else:
        print(f"{n - rank} Column vectors are LD. Non-Trivial solution(rank<unknown)")
        return 1


def gram_schmidt(A):
    (m, n) = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    addition_count = 0
    substraction_count = 0
    multiplication_count = 0
    divison_count = 0
    for j in range(n):
        q = A[:, j]  # i-th column of A
        for i in range(j):
            k = Q[:, i]
            R[i, j] = k.dot(q)
            addition_count += m -1
            multiplication_count += m
            q = q - (R[i, j] * k)
            multiplication_count += m
            substraction_count += m

        if np.array_equal(q, np.zeros(q.shape)):
            raise np.linalg.LinAlgError("The column vectors are not linearly independent")

        # normalize q
        norm = np.sqrt(np.dot(q, q))
        addition_count += m - 1
        multiplication_count += m

        # write the vector back in the matrix
        Q[:, j] = q / norm
        divison_count += m
        R[j, j] = norm
    print("\nOrthogonal matrix Q from input matrix")
    print("---------------------------------------")
    print(Q)
    print("\nRight Triangular matrix R from input matrix")
    print("---------------------------------------")
    print(R)
    print("\nOperation summary")
    print("------------------")
    print(f"Addition count: {addition_count}")
    print(f"Substraction count:{substraction_count}")
    print(f"Multiplication count:{multiplication_count}")
    print(f"Division count:{divison_count}")

    #print("\n (A - QR)")
    #print("------------------")
    #print(A - Q.dot(R))
    print("\nL-Frobenius norm for (A - QR)")
    print("-------------------------------")
    get_norm_all(A - (Q.dot(R)),m,n,ds)
    return

# Driver Code
if __name__ == "__main__":

    # Keep generating input matrix until LI
    gs_flag = True
    retry_count =0
    max_retry=10
    while(gs_flag):
        # Input Matrix Dimensions
        rows = int(input("Enter row: "))
        cols = int(input("Enter column: "))

        if (rows < cols):
            print(f"Error: Please enter row > col")
            exit(1)

        # Input data format r.dddd
        d = 4

        # Random input coefficient matrix
        a_temp = np.random.rand(rows, cols) * 10
        A_mat = np.around(a_temp, 4)

        # Negative test1
        """
        A_mat_neg = np.array(
            [[1., 2., 3., 4.],
             [1., 2., 3., 4.],
             [1., 2., 3., 4.]]
        )
        rows = len(A_mat_neg)
        cols = A_mat_neg[0].size
        A_mat = A_mat_neg
    
        
        # Negative test2
        A_mat_neg = np.array(
                        [[1., 2., 5., 8.],
                        [1., 3., 6., 9.],
                        [1., 4., 7., 10.]]
                            )
    
        rows= len(A_mat_neg)
        cols= A_mat_neg[0].size
        A_mat= A_mat_neg
        """
        # ds rounding for output and intermediate calculations
        ds = 4

        print(f"\nInput matrix with dimensions:{rows} x {cols}")
        print("-----------------------------------")
        print(A_mat)

        is_GS = check_gs(np.copy(A_mat), rows, cols, ds)
        if (is_GS == 0):
            print("[Success] Gram-Schmidt(Orthonormal) is applicable on all columns of the matrix")
            gs_flag = False
        else:
            print("[Failure] Gram-Schmidt(Orthonormal) is not applicable...Retrying..\n")
            retry_count +=1
            if(retry_count == max_retry):
                print("Max retry reached... Exit")
                exit(1)
    gram_schmidt(A_mat)

