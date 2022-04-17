import sys

def is_ddm(D, n):
    """
      Function to check whether given matrix is Diagonally Dominant Matrix

      input: D is an n x n  matrix
             n length of the matrix
      output: True for DDM else False
      """

    # for each row
    for i in range(0, n):
        # for each column, finding
        # sum of each row.
        sum = 0
        for j in range(0, n):
            sum = sum + abs(D[i][j])

        # removing the
        # diagonal element.
        sum = sum - abs(D[i][i])

        # checking if diagonal
        # element is less than
        # sum of non-diagonal
        # element.
        if (abs(D[i][i]) < sum):
            return False

    return True


def find_steps_for_ddm(D,n):
    """
      Function to check the minimum steps required to convert the given matrix to DDM matrix

      input: D is an n x n  matrix
             n length of the matrix
      output: s [0: Can't be made DDM | >0: Can made DDM with given steps]
      """

    result = 0

    # For each row
    for i in range(n):

        # To store the sum of the current row
        sum = 0
        for j in range(n):
            sum += abs(D[i][j])

        # Remove the element of the current row
        # which lies on the main diagonal
        sum -= abs(D[i][i])

        # Checking if the diagonal element is less
        # than the sum of non-diagonal element
        # then add their difference to the result
        if abs(D[i][i]) < abs(sum):
            result += abs(abs(D[i][i]) - abs(sum))

    return result


# Driver Code
if __name__ == "__main__":

    # Positive use case
    #A = [[3, 4, 1],
    #     [1, -3, 2],
    #     [-1, 2, 4]]
    #n = len(A)
#
    A =[[3, -2, 1],
        [1, -3, 2],
        [-1, 2, 4]]
    n = len(A)

    if is_ddm(A, n):
        print("Yes ! Given matrix is Diagonally dominant")
        sys.exit(0)
    else:
        print("Given matrix is not Diagonally dominant")

    print("Trying to check if the given matrix can be made Diagonally dominant")

    r = find_steps_for_ddm(A,n)

    if r >0:
        print(f"Given matrix can be made Diagonally dominant using {r} steps")
    else:
        print(f"Sorry ! Given matrix can not be made Diagonally dominant")



