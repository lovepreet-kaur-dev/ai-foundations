from foundations.linalg.core import vec_add, scale
import numpy as np

def linear_combination(scalars: list[float], vectors: list[list[float]]) ->  list[float]:
    terms = [scale(c, v) for c, v in zip(scalars, vectors, strict=True)]
    result = terms[0]
    for t in terms[1: ]: # with terms[1: ] we are iterating over all the terms in the list except for the first one. 
        result = vec_add(result, t)
    return result

def matvec_columns(A: list[list[float]], x: list[float]) -> list[float]:
    columns = [[row[j] for row in A] for j in range(len(A[0]))] # len(A[0] gives the number of columns because A is a list of lists, where each inner list represents a row of the matrix. The length of the first row (A[0]) corresponds to the number of columns in the matrix. By using len(A[0]), we can determine how many columns there are and iterate over that range to extract each column from the matrix A.
    return linear_combination(x, columns)

if __name__ == "__main__":
    A = [[2.0, 0.0], [1.0,  3.0]]
    x = [1.0, 2.0]
    print(matvec_columns(A, x))
    An, xn = np.array(A), np.array(x)
    col_view = sum (xn[j] * An[:, j]  for j in range(An.shape[1]))  #in this case it is An-shape and not len(A[0]) as before because An is a numpy array and we can use the shape attribute to get the number of columns. The shape attribute returns a tuple representing the dimensions of the array, where An.shape[1] gives us the number of columns in the array. This is equivalent to len(A[0]) for a regular Python list, but using shape is more efficient and concise when working with numpy arrays.
    assert np.allclose(col_view, An @ xn)  # @ is the matrix multiplication operator in numpy. It performs the dot product between the matrix An and the vector xn, resulting in a new vector that represents the linear combination of the columns of An weighted by the corresponding elements of xn. 
    assert np.allclose(matvec_columns(A, x), An @ xn)

    print((An.shape[1]))

    #What is the difference between len(A[0]) and An.shape[1]? The difference between len(A[0]) and An.shape[1] lies in the data structures they are applied to. len(A[0]) is used for a regular Python list of lists (A), where A[0] represents the first row of the matrix, and len(A[0]) gives the number of columns in that row. On the other hand, An.shape[1] is used for a NumPy array (An), where shape is an attribute that returns a tuple representing the dimensions of the array. An.shape[1] gives the number of columns in the NumPy array. 
    #show the attribute shape formal syntax: The attribute shape is accessed using dot notation, like this: array_name.shape. For example, if you have a NumPy array named An, you can get its shape by writing An.shape. This will return a tuple representing the dimensions of the array, where the first element is the number of rows and the second element is the number of columns. how many elements are in the tuple returned by shape? The tuple returned by the shape attribute contains two elements for a 2D array: the first element represents the number of rows, and the second element represents the number of columns. For example, if An is a 2D NumPy array with 3 rows and 4 columns, An.shape would return (3, 4).

