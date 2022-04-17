import numpy as np
from scipy.linalg import solve
import sys

def gauss_jacobi(A, b,x,max_iterations,tolerance):
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

    T = A - np.diag(np.diagonal(A))
    success = False
    for k in range(max_iterations):
        x_old = x.copy()
        x[:] = (b - np.dot(T, x)) / np.diagonal(A)

        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
            success= True

        print(np.around(x,4))
        if success:
            break
    if success:
        print(f"[Success] It has converged after {k +1} iterations !!!")
        #print(f"Final solution vector is:{x}")
    else:
        print("\n[Failure] It has not converged. Try with larger iteration count")
    return x


def gauss_seidel(A, b,x,max_iterations,tolerance ):
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

    for k in range(max_iterations):
        x_old = x.copy()
        success = False

        # Loop over rows
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, (i + 1):], x_old[(i + 1):])) / A[i, i]
        # Stop condition
        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
            success = True
        print(np.around(x, 4))
        if success:
            break
    if success:
        print(f"[Success] It has converged after {k +1} iterations !!!")
        print(f"Final solution vector is:{x}")
    else:
        print("\n[Failure] It has not converged. Try with larger iteration count")
        print(f"Final solution vector is:{x}")
    return x


if __name__ == "__main__":
    #np.random.seed(42)

    """
    Random float input coefficient array
    """
    #a_temp = np.random.rand(4, 4) * 10
    #A_mat = np.around(a_temp, 0)  # there can random rounding of Zeros like 1.91 which is essentially 1.9100
    #print(A_mat)
    """
    Random float input value vector
    """
    #b_temp = np.random.rand(4) * 10
    #b_mat = np.around(b_temp, 0)  # there can random rounding of Zeros like 4.738 which is essentially 1.9100

    """
    Positive usecase[It will converge]
    """
    #A_mat=np.array([[5,-2,3,5],[-3,9,1,6],[2,-1,-7,8],[5,-6,-7,9]])
    #b_mat = np.array([-1,4,3,9])

    #A_mat = np.array([[4,2,-1], [3,-5,1], [1,0,2]])
    #b_mat= np.array([-1,3,-4])

    #A_mat = np.array([[4,2,3,3], [1,3,3,1], [1,2,4,2],[2,2,3,3]])
    #b_mat= np.array([3,4,4,2])

    """
    Negative usecase[It will not converge]
    """
    #A_mat = np.array([[9,2,6,3], [8,2,4,2], [6,4,8,6], [1,3,8,1]])
    #b_mat= np.array([9,8,9,4])

    """
    Random int input value vector
    """
    np.random.seed(42)
    A_mat = np.random.randint(10, size=(4, 4))
    b_mat = np.random.randint(10, size=(4))

    """
    Other inputs
    """
    max_iterations = 10
    tolerance = 0.1
    x = np.zeros_like(b_mat, dtype=np.double)
    algo="s"

    print(f"Input coefficient matrix:\n{A_mat}")
    print(f"Input value vector:\n{b_mat}")
    print(f"Input guess vector:\n{x}")
    print(f"Max iterations:\n{max_iterations}")
    print(f"Tolerance:\n{tolerance}")

    """
    Algo execution
    """

    if algo == "s":
        print(f"\nStarted Gauss Seidel iteration for {max_iterations} iterations:")
        k=gauss_seidel(A_mat,b_mat,x,max_iterations,tolerance)
        #print(f"\nAfter {max_iterations} iterations Gauss Seidel current solution vector:\n{k}")
    elif algo == "j":
        print(f"\nStarted Gauss Jacobi iteration for {max_iterations} iterations:")
        k=gauss_jacobi(A_mat, b_mat,x,max_iterations, tolerance)
        #print(f"\nAfter {max_iterations} iterations Gauss Jacobi current solution vector:\n{k}")
        #print(x)
    else:
        sys.exit("Invalid algo: please enter s or j")

    #print("Solution vector for Gauss Seidel iterations is: \n", solve(a, b))