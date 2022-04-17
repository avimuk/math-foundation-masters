import time
start = time.process_time()
n = 1000000
# sum
def generate_sum():
    r = 1
    for i in range(1,n):
        r += i + 1
    #print(r)
    sum_time_taken = (time.process_time() - start)
    print(f"Total time taken for sum of 10^6 numbers: {sum_time_taken}")
    sum_time_taken_per_stmt = sum_time_taken / n
    print(f"Single addition time taken for sum of 10^6 numbers:{sum_time_taken_per_stmt}")


def generate_product():
    r = 1
    for i in range(1,n):
        r *= i + 1
    #print(r)
    mul_time_taken = (time.process_time() - start)
    print(f"Total time taken for multiplication of 10^6 numbers: {mul_time_taken}")
    mul_time_taken_per_stmt = mul_time_taken / n
    print(f"Single multiplication time taken for multiplication of 10^6 numbers:{mul_time_taken_per_stmt}")

def generate_division():
    r = 1
    for i in range(1,n):
        r /= i + 1
    #print(r)
    div_time_taken = (time.process_time() - start)
    print(f"Total time taken for division of 10^6 numbers: {div_time_taken}")
    div_time_taken_per_stmt = div_time_taken / n
    print(f"Single division time taken for division of 10^6 numbers:{div_time_taken_per_stmt}")

if __name__ == "__main__":
    generate_sum()
    generate_product()
    generate_division()


