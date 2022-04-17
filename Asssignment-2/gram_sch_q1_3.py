import numpy as np
import scipy.linalg as la
from scipy.linalg import solve
import sys

"""
Program to generate the orthogonal matrix Q from a matrix A by performing  the Gram-Schmidt orthogonalization method
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
    Q = np.copy(A)
    for i in range(n):
        q = A[:, i]  # i-th column of A

        for j in range(i):
            q = q - np.dot(A[:, j], A[:, i]) * A[:, j]

        if np.array_equal(q, np.zeros(q.shape)):
            raise np.linalg.LinAlgError("The column vectors are not linearly independent")

        # normalize q
        q = q / np.sqrt(np.dot(q, q))

        # write the vector back in the matrix
        Q[:, i] = q
    return Q

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
    print("\nOrthogonal matrix Q from input matrix")
    print("---------------------------------------")
    q= gram_schmidt(A_mat)
    print(q)

