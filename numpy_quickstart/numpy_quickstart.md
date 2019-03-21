
# Numpy quickstart tutorial

## Matrix declaration
```python
# array creation and attributes

import numpy as np

arr = np.arange(15)
print("arr value: \n" + str(arr) + "\n" + "arr.dtype: " + str(arr.dtype) + ", arr.itemsize: " + str(arr.itemsize) + "\n")

arr = np.reshape(arr, (3, 5))
print("reshaped arr: \n" + str(arr) + "\n" + "arr.dim: " + str(arr.ndim) + ", arr.shape: " + str(arr.shape) + "\n")

zeros_arr = np.zeros((2,3))
print("zeros_arr: \n" + str(zeros_arr) + "\n" + "zeros_arr.dtype: " + str(zeros_arr.dtype) + "\n")

ones_arr = np.ones((3, 4), dtype=np.int32)
print("ones_arr: \n" + str(ones_arr) + "\n" + "ones_arr.dtype: " + str(ones_arr.dtype) + "\n")

empty_arr = np.empty((2, 3))
print("empty_arr: \n" + str(empty_arr) + "\n" + "empty_arr.dtype: " + str(empty_arr.dtype))
```
output:
```

    arr value: 
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
    arr.dtype: int32, arr.itemsize: 4
    
    reshaped arr: 
    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]]
    arr.dim: 2, arr.shape: (3, 5)
    
    zeros_arr: 
    [[0. 0. 0.]
     [0. 0. 0.]]
    zeros_arr.dtype: float64
    
    ones_arr: 
    [[1 1 1 1]
     [1 1 1 1]
     [1 1 1 1]]
    ones_arr.dtype: int32
    
    empty_arr: 
    [[0. 0. 0.]
     [0. 0. 0.]]
    empty_arr.dtype: float64
    

```

## Matrix operations

```python
# array arithmetic operations

import numpy as np

# arithmetic operators on arrays apply elementwise

arr_a = np.array([1, 2, 3, 4])
arr_b = np.full(4, np.pi)
print("arr_a: " + str(arr_a))
print("arr_b: " + str(arr_b))

print("arr_a**2: " + str(arr_a**2))

print("arr_a * arr_b: " + str(arr_a * arr_b))

arr_rand = np.random.random((2,3))
print("arr_rand: \n" + str(arr_rand))

print("arr_rand.sum(): " + str(arr_rand.sum()))

# minimum of the first column of matrix
print("arr_rand[:, 0].min(): " + str(arr_rand[:, 0].min()))

# sum of each column
print("arr_rand.sum(axis=0): " + str(arr_rand.sum(axis=0)))
```
output:
```

    arr_a: [1 2 3 4]
    arr_b: [3.14159265 3.14159265 3.14159265 3.14159265]
    arr_a**2: [ 1  4  9 16]
    arr_a * arr_b: [ 3.14159265  6.28318531  9.42477796 12.56637061]
    arr_rand: 
    [[0.44355279 0.4140115  0.96297641]
     [0.92558623 0.93851033 0.32560827]]
    arr_rand.sum(): 4.010245531011682
    arr_rand[:, 0].min(): 0.4435527856901734
    arr_rand.sum(axis=0): [1.36913902 1.35252183 1.28858469]
    

```

## Matrix indexing, slicing and iterating
```python
# matrix indexing, slicing and iterating

import numpy as np

# one-dimensional
arr = np.arange(10)**3

print("arr: " + str(arr))
print("arr[:6:2]: " + str(arr[:6:2]))

# in reversed order
print("arr[::-1]: " + str(arr[::-1]) + "\n")

# multi-dimensional

# [concept] arrays can have one index per axis,
# These indices are given in a tuple separated by commas

def f(x,y):
    return 10 * x + y

# pass function as element generator to generate matrix
arr = np.fromfunction(f, (2, 3))
print("np.fromfunction(f, (2, 3)): \n" + str(arr) + "\n")

# second row of matrix
print("arr[1,:]: " + str(arr[1,:]))

# last column of matrix
print("arr[:, -1]: " + str(arr[:, -1]) + "\n")

# iterating

arr_2d = np.arange(6).reshape((2,3))
print("arr_2d: \n" + str(arr_2d) + "\n")

# iterating over multi-dimensional array is done with respect to first axis(x)
print("iterate over rows: ")
for row in arr_2d:
    print(row)

# so how to iterate over columns? transpose it!
print("\niterate over columns: ")
for column in arr_2d.T:
    print(column)

arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\narr_3d: \n" + str(arr_3d) + "\n")

# iterate over z-axis
print("\niterate over z-axis: ")
for x in range(arr_3d.shape[0]):
    for y in range(arr_3d.shape[1]):
        print(arr_3d[x, y, :])

# iterate every element in matrix
print("\niterate every element:")
for e in arr_3d.flat:
    print(e)
```
output:
```

    arr: [  0   1   8  27  64 125 216 343 512 729]
    arr[:6:2]: [ 0  8 64]
    arr[::-1]: [729 512 343 216 125  64  27   8   1   0]
    
    np.fromfunction(f, (2, 3)): 
    [[ 0.  1.  2.]
     [10. 11. 12.]]
    
    arr[1,:]: [10. 11. 12.]
    arr[:, -1]: [ 2. 12.]
    
    arr_2d: 
    [[0 1 2]
     [3 4 5]]
    
    iterate over rows: 
    [0 1 2]
    [3 4 5]
    
    iterate over columns: 
    [0 3]
    [1 4]
    [2 5]
    
    arr_3d: 
    [[[1 2]
      [3 4]]
    
     [[5 6]
      [7 8]]]
    
    
    iterate over z-axis: 
    [1 2]
    [3 4]
    [5 6]
    [7 8]
    
    iterate every element:
    1
    2
    3
    4
    5
    6
    7
    8
    
```

## References
* [Numpy Official Quickstart tutorial](https://docs.scipy.org/doc/numpy/user/quickstart.html)