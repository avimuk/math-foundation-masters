import sys
import numpy as np
import time

def ge_without_pivot(D, g, ds):
    """
      Gaussian elimination without partial pivoting.

      input: A is an n x n  matrix
             b is an n x 1  vector
      output: x is the solution of Ax=b
              with the entries permuted in
              accordance with the pivoting
              done by the algorithm
      """

    A = np.array((D), dtype=float)
    f = np.array((g), dtype=float)
    n = f.size

    validate_input(A, f)

    addition_count = 0
    substraction_count = 0
    multiplication_count = 0
    divison_count = 0

    # Elimination
    for i in range(0, n - 1):  # Loop through the columns of the matrix
        for j in range(i + 1, n):  # Loop through rows below diagonal for each column
            if A[i, i] == 0:
                sys.exit("Error: Zero on diagonal can cause division by Zero issue! Execute ge_with_pivot(A, f)")
            m = round_s(float(A[j, i] / A[i, i]),ds)
            divison_count += 1
            A[j, :] = A[j, :] - m * A[i, :]
            substraction_count += n
            multiplication_count += n
            f[j] = round_s(float(f[j] - m * f[i]),ds)
            substraction_count += 1
            multiplication_count += 1

    # Back Substitution
    n = f.size
    x = np.zeros(n)  # Initialize the solution vector, x, to zero
    x[n - 1] = round_s(float(f[n - 1] / A[n - 1, n - 1]),ds)  # Solve for last entry first
    divison_count += 1
    for i in range(n - 2, -1, -1):  # Loop from the end to the beginning
        sum_ = 0
        for j in range(i + 1, n):  # For known x values, sum and move to rhs
            sum_ = round_s(float(sum_ + A[i, j] * x[j]),ds)
            addition_count += 1
            multiplication_count += 1

        x[i] = round_s(float((f[i] - sum_) / A[i, i]),ds)
        substraction_count += 1
        divison_count += 1

    print(f"\nGauss elimination without pivoting summary:")
    print(f"-------------------------------------------")
    #print(f"Solution is: {x}")
    print(f"Addition count: {addition_count}")
    print(f"Substraction count:{substraction_count}")
    print(f"Multiplication count:{multiplication_count}")
    print(f"Division count:{divison_count}")

    return A, f


def ge_with_pivot(D, g,ds):
    """
      Gaussian elimination with pivoting.

      input: A is an n x n numpy matrix
             b is an n x 1 numpy array
      output: x is the solution of Ax=b
              with the entries permuted in
              accordance with the pivoting
              done by the algorithm
      """


    A = np.array((D), dtype=float)
    f = np.array((g), dtype=float)
    n = len(f)

    validate_input(A,f)

    addition_count = 0
    substraction_count = 0
    multiplication_count = 0
    divison_count = 0
    swap_count = 0


    for i in range(0, n - 1):  # Loop through the columns of the matrix

        if np.abs(A[i, i]) == 0:
            for k in range(i + 1, n):
                if np.abs(A[k, i]) > np.abs(A[i, i]):
                    A[[i, k]] = A[[k, i]]  # Swaps ith and kth rows to each other
                    f[[i, k]] = f[[k, i]]
                    swap_count += 2
                    break

        for j in range(i + 1, n):  # Loop through rows below diagonal for each column
            m = round_s(float(A[j, i] / A[i, i]),ds)
            divison_count += 1

            A[j, :] = A[j, :] - m * A[i, :]
            substraction_count += n
            multiplication_count += n

            f[j] = round_s(float(f[j] - m * f[i]),ds)
            substraction_count += 1
            multiplication_count += 1


    # Back Substitution
    n = f.size
    x = np.zeros(n)  # Initialize the solution vector, x, to zero
    x[n - 1] = round_s(float(f[n - 1] / A[n - 1, n - 1]),ds)  # Solve for last entry first
    divison_count += 1
    for i in range(n - 2, -1, -1):  # Loop from the end to the beginning
        sum_ = 0
        for j in range(i + 1, n):  # For known x values, sum and move to rhs
            sum_ = round_s(float(sum_ + A[i, j] * x[j]),ds)
            addition_count += 1
            multiplication_count += 1

        x[i] = round_s(float((f[i] - sum_) / A[i, i]),ds)
        substraction_count += 1
        divison_count += 1

    print(f"\nGauss elimination with pivoting summary:")
    print(f"-------------------------------------------")
    #print(f"Solution is: {x}")
    print(f"Addition count: {addition_count}")
    print(f"Substraction count:{substraction_count}")
    print(f"Multiplication count:{multiplication_count}")
    print(f"Division count:{divison_count}")
    print(f"Swap count:{swap_count}")

    return A, f

def validate_input(D, g):
    l=len(D)
    s=g.size
    if l != s:
        sys.exit(f"Invalid argument: incompatible sizes between A & b | A:{l}, b:{s}")


def round_s(n, dp=0):
    """
      Arithmetic rounding

      input: Actual number
             Rounding factor(like 5 for 5s)
      output: Rounded number
      """

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

if __name__ == "__main__":
    # Static Input co-efficient matrix [Uncomment to run]
    #A = np.array([[1., -1., 1., -1.],
    #              [1., 0., 0., 0.],
    #              [1., 1., 1., 1.],
    #              [1., 2., 4., 8.]])
    #
    ## Input value vector
    #f = np.array([[14.],
    #              [4.],
    #              [2.],
    #              [2.]])
    #

    # Random large Input co-efficient matrix [Uncomment to run]
    start_sample =100
    end_sample=1100
    for sample in range(start_sample, end_sample):
        if sample % 100 == 0:
            a_temp = np.random.rand(sample, sample) * 10
            A_mat = np.around(a_temp, 4)  # there can random rounding of Zeros like 1.91 which is essentially 1.9100
            # print(A_mat)

            b_temp = np.random.rand(sample, 1) * 10
            b_mat = np.around(b_temp, 4)  # there can random rounding of Zeros like 4.738 which is essentially 1.9100

            ds = 5
            print(f"\n###############################################")
            print(f"Starting processing for matrix {sample} x {sample}")
            print(f"###############################################\n")
            start = time.process_time()
            ge_without_pivot(A_mat, b_mat, ds)
            t1 = (time.process_time() - start)

            print(f"Actual time taken for ge_without_pivot: {t1}")

            start = time.process_time()
            ge_with_pivot(A_mat, b_mat, ds)
            t2 = (time.process_time() - start)
            print(f"Actual time taken for ge_with_pivot: {t2}")

