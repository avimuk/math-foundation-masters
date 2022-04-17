import numpy as np
import scipy.linalg as la
from scipy.linalg import solve
import sys


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


# Gauss Seidel iteration
def gausSeidel(A, b, x, n):
    """
      Gaussian Seidel iterations

      input: A is an n x n  coefficient matrix
             b is an n x 1  value vector
             x is an n x 1  assume vector
             max_iterations maximum iterations
             tolerance Tolerance limit
      output: x is the solution vector
              All iteration vector
              Converge success/Failure
      """

    L = np.tril(A)
    U = A - L

    for i in range(n):
        x = np.matmul(np.linalg.inv(L), b - np.matmul(U, x))
        print(x)
    return x

# Gauss Jacobi iteration
def gaussJacobi(A, b, x, n):
    """
      Gaussian Jacobi iterations

      input: A is an n x n  coefficient matrix
             b is an n x 1  value vector
             x is an n x 1  assume vector
             max_iterations maximum iterations
             tolerance Tolerance limit
      output: x is the solution vector
              All iteration vector
              Converge success/Failure
      """

    L = np.tril(A)
    U = A - L

    for i in range(n):
        s = -(L + U)
        x = np.matmul(s, x) + b
        print(x)
    return x

# Iteration array(constant for Gauss Siedel and Gauss Jacobi)
def get_iteration_array(A,l,algo,ds):
    """
      Calculate Iteration array

      input: A is an n x n  coefficient matrix
             l  length of the matrix
             algo s(Gauss Siedel) j(Gauss Jacobi)
             ds decimal rounding
      output: x is the iteration vector
              All iteration vector
              Converge success/Failure
      """

    #Normalize the co-efficient matrix
    for i in range(0,l):  # Loop through the columns of the matrix
        diag= abs(A[i,i])
        for j in range(0,l):  # Loop through rows below diagonal for each column
            if(diag != 0):
                k= round_s(float(np.divide(A[i,j],diag)),ds)
                A[i, j] = k
#
    #print(A)
    ## make the diagonal 0
    np.fill_diagonal(A, 0)

    # Fine L , U and I
    L = np.tril(A)
    U = A - L
    I = np.identity(l)

    if algo == "s":
        k = - np.linalg.inv(I+L)
        p = np.matmul(k, U)
    elif algo == "j":
        #print(L+U)
        p= -(L + U)
    else:
        sys.exit("Invalid algo: please enter s or j")

    #np.fill_diagonal(p, 0)  # to handle -0
    return p

def get_norm_all(A,n,m,ds):
    """
      Calculate all norms

      input: A is an n x n  coefficient matrix
             b is an n x 1  value vector
             x is an n x 1  assume vector
             max_iterations maximum iterations
             tolerance Tolerance limit
      output: x is the solution vector
              All iteration vector
              Converge success/Failure
      """

    #rows n and columns m, for square matrix its same

    #L2
    r= round_s(float(np.sqrt(np.max(np.linalg.eigvals(np.inner(A, A))))),ds)
    print(f"L2 norm is: {r}")

    #print(np.linalg.norm(A, 2))

    #L-infinity
    rowsums = []
    for i in np.arange(n):
        v = np.sum(np.absolute(A[i, :]))
        rowsums.append(v)
    r=round_s(float(np.max(rowsums)),ds)
    print(f"L-infinity norm is: {r}")

    #print(np.max(np.sum(np.abs(A), axis=1)))
    #print(np.linalg.norm(A, np.inf))

    #L1
    columns = []
    for i in np.arange(m):
        v = np.sum(np.abs(A[:, i]))
        columns.append(v)
    r=round_s(float(np.max(columns)),ds)
    print(f"L1 norm is: {r}")

    #print(np.max(np.sum(np.abs(A), axis=0)))
    #print(np.linalg.norm(A, 1))


    #Frobenius norm
    f = 0
    for i in np.arange(n):
        for j in np.arange(m):
            f = f + np.sum(np.power(np.abs(A[i, j]), 2))

    r=round_s(float(np.sqrt(f)),ds)
    print(f"L-Frobenius norm is: {r}")

    #print(np.sqrt(np.sum(np.abs(A) ** 2)))
    #print(np.linalg.norm(A, 'fro'))


# Driver Code
if __name__ == "__main__":
    # as np.array([a11,a12,a13], [a21,a22,a23], [a31,a32,a33])
    #A = eval(input("Enter matrix A: "))
    #b = eval(input("Enter matrix b: "))
    #x = eval(input("Enter guess of x: "))
    #n = eval(input("Enter no of iterations: "))

    #A=np.array([[5., -1., 3.], [2., -8., 1.], [-2., 0., 4.]])
    #b=[-1.,2.,3.]

    # Random input coefficient array

    a_temp = np.random.rand(4, 4) * 10
    A_mat = np.around(a_temp, 0)

    # Random input value vector
    b_temp = np.random.rand(4) * 10
    b_mat = np.around(b_temp, 0)

    #A_mat = np.random.randint(10, size=(4, 4))
    #b_mat = np.random.randint(10, size=(4))

    #a = np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]])
    #b = np.array([-1, 2, 3])
    #x=np.zeros(3)

    #a = np.array([[9, 2, 6, 3], [8, 2, 4, 2], [6, 4, 8, 6], [1, 3, 8, 1]])
    #b = np.array([9, 8, 9, 4])
    # x=np.zeros(3)

    #A_mat = np.random.randint(10, size=(4, 4))
    #b_mat = np.random.randint(10, size=(4))

    np.random.seed(42)
    l = len(A_mat)

    # Guess vector
    x=np.zeros(l)
    #x = np.zeros_like(b_mat, dtype=np.double)

    # No of iterations
    n= 10

    # s= Siedel and j= Jacobi
    algo="j"

    # ds rounding
    ds =3

    print(f"Input coefficient vector:\n{A_mat}")
    print(f"Input value vector:\n{b_mat}")
    print(f"Input guess vector:\n{x}")
    #print(A_mat)
    #print(b_mat)
    #print(x)

    # Get the iteration array
    a_it= get_iteration_array(np.copy(A_mat),l,algo,ds)
    print(f"Iteration matrix:\n{a_it}")

    l= len(a_it)
    print("\nAll norm value:\n")
    get_norm_all(np.copy(a_it),l,l,ds)

    if algo == "s":
        # Perform Gauss Seidel iterations
        print(f"\nGauss Seidel iteration for {n} iterations:")
        x = gausSeidel(np.copy(A_mat), np.copy(b_mat), x, n)
        print(f"\nAfter {n} iterations Gauss Seidel solution vectior:\n{x}")
        #print("Solution vector for Gauss Seidel iterations is: \n", solve(A_mat, b_mat))
    elif algo == "j":
        # Perform Gauss Jacobi iterations
        print(f"\nGauss Jacobi iteration for {n} iterations:\n")
        x = gaussJacobi(np.copy(A_mat), np.copy(b_mat), x, n)
        print(f"\n After {n} iterations Gauss Jacobi solution vectior:\n{x}")
        #print("Solution vector for Gauss Jacobi iterations   is: \n", solve(A_mat, b_mat))
    else:
        sys.exit("Invalid algo: please enter s or j")

