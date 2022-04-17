import numpy as np
import scipy.linalg as la
from scipy.linalg import solve
import sys

"""
Program to extend previous code to generate input matrix and check if Gram-Schmidt Algorithm can be applied on the matrix

Theory:
Gram-Schmidt(GS) Algorithm can be applied on a set of LI vectors. For a matrix it should consist of LI column vectors.
So to check a matrix for GS(Orthonormal) eligibility, we have to check if the matrix is full column rank.

Logic:
1) Get the rank of the matrix
2) If rank = total number of columns then GS can be applied to the matrix
	Else return error message
	
[Note]: A set of linearly dependent vectors cannot be made orthonormal, but it can be made orthogonal.
e.g. if our set has 𝑚 vectors, and spans a subspace of dimension 𝑛, Gram-Schmidt will give us the 
zero vector for 𝑣𝑛+1,...,𝑣𝑚. This set is still orthogonal by definition, but we can't multiply the 
zero vector by a scalar to give it unit length. However, we can just throw away 𝑣𝑛+1,...,𝑣𝑚, and 
then it can be made orthonormal, as the remaining vectors are orthogonal and nonzero and at that point 
we get an orthonormal basis for the subspace spanned by our linearly dependent set.
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


def check_GS(A, m, n, ds):
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
    print("\n Analysis")
    print("-----------")
    print(f"Rank of the matrix:{rank}")
    print(f"Total columns of the matrix:{n}")
    if (rank == n):
        print(f"All column vectors are LI. Trivial solution(rank=unknown)")
        return 0
    else:
        print(f"{n - rank} Column vectors are LD. Non-Trivial solution(rank<unknown)")
        return 1


# Driver Code
if __name__ == "__main__":

    # Input Matrix Dimensions
    ip_flag = True
    retry_count = 0
    max_retry = 3
    while (ip_flag):
        rows = int(input("Enter row: "))
        cols = int(input("Enter column: "))
        if (rows < cols):
            print(f"Error: Please enter row > col...Attempt:{retry_count + 1}....Max allowed retry:{max_retry}")
            retry_count += 1
            if (retry_count == max_retry):
                print("Exceeded max retry... Exit")
                exit(1)
        else:
            ip_flag = False

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
    is_GS = check_GS(np.copy(A_mat), rows, cols, ds)
    print("\nConclusion")
    print("-----------")
    if (is_GS == 0):
        print("[Success] Gram-Schmidt(Orthonormal) is applicable on all columns of the matrix")
    else:
        print("[Failure] Gram-Schmidt(Orthonormal) is not applicable on all columns of the matrix.\n")
