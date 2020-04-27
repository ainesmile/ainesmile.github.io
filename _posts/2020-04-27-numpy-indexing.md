---
layout: post
title:  "NumPy: Indexing"
date:   2020-04-27 +1800
keywords: NumPy
categories: Machine-Learning
---

NumPy provides two fundamental objects: an N-dimensional array object(**ndarray**) and a universal function object(**ufunc**). 
An ndarray is a **homogeneous** collection of "items" indexed using N integers. 
There are two essential parts of an ndarray: 1) the **shape**, and 2) the data-type(**dtype**). Here, homogeneous means the dtype of the "items" are all same, each item occupies the same size block of memory, and each block of memory is interpreted in exactly the same way.

```python
>>> x = np.array([[1, 2, 3], [2, 3, 4]])
>>> x.shape
(2, 3)
>>> x.dtype
dtype('int64') # 64-bit platform
```


In this article, we will talk about indexing through:

* Memory layout of ndarray and strides
* View and copy
* Indexing styles
* Indexing


From the perspective of the memory layout and strides, we can understand what indexing really are. And view and copy are the two different results of the indexing operators.


## 1. Memory Layout of ndarray and Strides

> On a fundamental level, an ndarray is just a one-dimensional sequence of memory with fancy indexing code that maps an N-dimensional index into a one-dimensional index.

In general, from the perspective of the memory layout, an ndarray is just a 1-dimensional sequence of memory, addressed by a single index(the memory address). However, it is not convenient for us to use a 1-dimensional indexing to locate an N-dimensional ndarray. We prefer to use an N-dimensional fancy indexing to locate an N-dimensional ndarray.


NumPy uses strides to map the fancy N-dimensional indexing into the one-dimensional memory address. For an ndarray with `n` dimensions, 
the strides of this ndarray is a `n` elements tuple, each element means 
the number of bytes needed to jump to the next element in that dimension.

What's more, with strides we can easily locate an ndarray that the underlying memory are even not contiguous which is not often to see.

```python
>>> x = np.arange(24, dtype=np.int32).reshape(4, 3, 2)
>>> x.strides
>>> bytearray(x.data)
bytearray(b'
\x00\x00\x00\x00
\x01\x00\x00\x00
\x02\x00\x00\x00
\x03\x00\x00\x00
\x04\x00\x00\x00
\x05\x00\x00\x00
\x06\x00\x00\x00
\x07\x00\x00\x00
\x08\x00\x00\x00
\t\x00\x00\x00
\n\x00\x00\x00
\x0b\x00\x00\x00
\x0c\x00\x00\x00
\r\x00\x00\x00
\x0e\x00\x00\x00
\x0f\x00\x00\x00
\x10\x00\x00\x00
\x11\x00\x00\x00
\x12\x00\x00\x00
\x13\x00\x00\x00
\x14\x00\x00\x00
\x15\x00\x00\x00
\x16\x00\x00\x00
\x17\x00\x00\x00')
>>> x[0, 0, 1], x[3, 2, 0]
(1, 22)
# offset = (0-1)*4 + (2-0)*8 + (3-0)*24 = 84 bytes
>>> int.from_bytes(b'\x16\x00\x00\x00', 'little')
22
```

Ndarray `x`'s dtype is `int32`, so each element occupies 4 bytes, the whole data occupy `4*24=96` bytes. In the last dimension, the stride is `4=32/8` and there are 2 elements, so the stride in the second dimension is `4*2=8`. Similarly, for the first dimension, the stride is `8*3=24`. Therefore, `x.strides=(24, 8, 4)`. In this case, if we want to move from `x[0, 0, 1]` to `x[3, 2, 0]`, we need to move `(0-1)*4 + (2-0)*8 + (3-0)*24 = 84` bytes, means from `b'\x01\x00\x00\x00` to `b'\x16\x00\x00\x00`.


And if we change the dtype, the number of memory block that each item occupied will change, so the stride will change too. 

```python
>>> data = np.arange(24, dtype='int8')
>>> x = data.reshape(2, 3, 4)
>>> x.strides
(12, 4, 1)
>>> x.dtype = 'int16'
>>> x.strides
(12, 4, 2)
```


## 2. View and Copy

In NumPy, view means 

> an array that does not own its data, but referees to another array's data instead.

When we create a view of an array, the two arrays share the same underlying data buffer, which is faster and can save memory. While when we create a copy of an array, a new data buffer will be created. So the copied array and base array are independent, which means any changes made to the array will not affect the other.

```python
>>> data = np.array([0, 1, 2])
>>> x = data.view()
>>> y = data.copy()
>>> x[0] = 3
>>> y[0] = 4
>>> data
array([3, 1, 2])
```

## 3. Indexing Styles(Orders)

There are two styles of N-dimensional indexing for an ndarray: the **C-style(row-major)** and the **Fortran-style(column-major)**. In the C-style, the rightmost dimension indexing "varies the fastest". That is, to move to the next block of memory, the last dimension index changes first. New NumPy arrays are by default in row-major order. To the contrary, in the Fortran-style,
the leftmost dimension indexing "varies the fastest".


```python
>>> data = np.arange(6, dtype='int8')
>>> x = np.reshape(data, (2, 3), 'C')
>>> y = np.reshape(data, (2, 3), 'F')
# data, x, and y share the same underlying data buffer
>>> np.shares_memory(data, x), np.shares_memory(data, y)
(True, True)
>>> x
array([[0, 1, 2],
       [3, 4, 5]], dtype='int8')
>>> y
array([[0, 2, 4],
       [1, 3, 5]], dtype='int8')
```

Here, `data`, `x`, and `y` share the same underlying data buffer in memory, which occupy 6 blocks of memory in 1-dimension. Each block represents a 8 bytes int scaler, the block `i` represents the number `i`(`i=0, 1, 2, 3, 4, 5`). With different orders, the indexing for `x` and `y` are different although they share the same blocks of memory.

**<center>Table1: Indexing Styles</center>**

| Memory Address | C-style index(x) | Fortran-style index(y) |
| ---- | ---- | ---- |
| 0 | (0, 0) | (0, 0) |
| 1 | (0, 1) | (1, 0) |
| 2 | (0, 2) | (0, 1) |
| 3 | (1, 0) | (1, 1) |
| 4 | (1, 1) | (0, 2) |
| 5 | (1, 2) | (1, 2) |


Obviously, an ndarray's indexing style will affect its strides. If two ndarraies share the same data buffer having different orders, their strides will be different. And if we change the strides, the array's order will change too.


```python
# x and y share the same data buffer, with different orders and strides
>>> data = np.arange(120)
>>> x = data.reshape(2, 3, 4, 5, order='C')
>>> y = data.reshape(2, 3, 4, 5, order='F')
>>> x.strides, y.strides
((480, 160, 40, 8), (8, 16, 48, 192))
>>> x.flags['C_CONTIGUOUS'], x.flags['F_CONTIGUOUS']
(True, False)
>>> y.flags['C_CONTIGUOUS'], y.flags['F_CONTIGUOUS']
(False, True)

# Now, change y's strides to x's
>>> y.strides = (480, 160, 40, 8)
# y will equal to x
>>> np.allclose(x, y)
True
# and y's order flag changes too, equals to x's
>>> y.flags['C_CONTIGUOUS'], y.flags['F_CONTIGUOUS']
(True, False)
```

## 4. Array Indexing


There are three kinds of indexing available using the `x[obj]` syntax: filed access, basic slicing, and advanced indexing, which one occurs depends on `obj`. For an N-dimension array, we use `,` to separate dimensions, each dimension could be sliced through the same syntax. In this article, we will talk about basic indexing and advanced indexing.


### 4.1 Basic Indexing

Basic indexing always returns another view of the array. Basic indexing occurs when `obj` is:

* integers
* slice objects, `x[start:stop:step]` or `x[slice(i, j, k)]`
* tuples of slice objects and integers, e.g. `x[(slice(i, j, k), slice(i, j, k))]`, `x[(slice(i, j, k), 0, slice(i, j, k))]`
* Ellipsis objects, e.g.`x[...,0]`
* newaxis objects, e.g.`x[np.newaxis, 0]`




### 4.1.1 Slice object

The basic syntax for slicing is `seq[i:j:k]`, 

* start-index i, included, default value is 0 for k > 0 and n-1 for k < 0
* stop-index j, excluded, default value is n for k > 0 and -n-1 for k < 0
* step-size k, default value is 1, cann't be 0

where `n` is the number of elements in the corresponding dimension. `seq[i:j:k]` is actually just a nice shorthand for `seq[slice(i, j, k)]`. 

```python
# 1. normal values
>>> x = np.arange(10)
>>> x[2:8:2]
array([2, 4, 6])
# 2. shorthand, seq[i:j:k] == seq[slice(i, j, k)]
>>> np.allclose(x[2:8:2], x[slice(2,8,2)])
True
```

Some examples for the dafault values:

```python
# 3. default values
# k_default = 1, i_default = 0
>>> x[:5]
array([0, 1, 2, 3, 4])
# k > 0, then i_default = 0, j_default = 10
>>> x[::2]
array([0, 2, 4, 6, 8])
# k < 0, then i_default = 9, j_default = -11
>>> x[::-1]
array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
```
There are some special cases for slicing, including `None`, negative values, and out-of-bounds start/stop values. `None` is literately None, thus `seq[:] == seq[::] = seq[None:None:None]`.

For negative values, a negative `i` equals to `n+i`, and a negative `j` equals to `n+j`, and a negative `k` means go in reverse order:

```python
# 4. negative values
# negative i or j, i = n + i, j = n + j
>>> x[-6:8], x[-6:-2]
(array([4, 5, 6, 7]), array([4, 5, 6, 7]))
# negative k, from right to left
>>> x[4:2:-1]
array([4, 3])
```

For out-of-bound values, the start index `i` will become the first available index, and the stop index `j` will become the last available index:

```python
# 5. out-of-bound i or j
>>> x[-10:20]
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> x[20:-10:-1]
array([9, 8, 7, 6, 5, 4, 3, 2, 1])
```

With the right syntax, we can still get an empty array. If the start index `i` in the left/right of the stop index `j`, and `k` in the reverse/normal order, empty array will be returned. This empty array in a view of the original array, but shares no data(there is no data to share).

```python
# 6. in wrong slice order, an empty array returned, and shares no memory
>>> y = x[2:4:-1]
>>> y
array([], dtype=int64)
>>> z = x[4:2:1]
>>> z
array([], dtype=int64)
>>> np.shares_memory(x, y), np.shares_memory(x, z), np.shares_memory(y, z)
(False, False, False) 
```


**<center>Table2: Slicing</center>**

| seq[i:j:k] | included| default value | negative value | None | out-of-bound |
| ---- | ---- | ---- | ---- | ---- |
| strat-index i | Yes | k>0: 0; <br> k<0: n-1 | i=n+i | : | k>0: 0 <br> k<0: n-1 |
| stop-index j | No | k>0: n; <br> k<0: -n-1 | j=n+j | : | k>0: n <br> k<0: 0 |
| step-size k | N/A | 1 | reverse order | 1 | N/A |

<br>

### 4.1.2 Newaxis and Ellipsis

We use `newaxis` to expand dimensions and use `...` to represent all the non-explicit dimensions.

```python
>>> x = np.arange(6)
>>> x[np.newaxis, 1:3:1, np.newaxis]
array([[[1],
        [2]]])
>>> x[np.newaxis, 1:3:1, np.newaxis].shape
(1, 2, 1)
>>> y = np.arange(120).reshape(2, 3, 4, 5)
>>> y[0,...,1]
array([[ 1,  6, 11, 16],
       [21, 26, 31, 36],
       [41, 46, 51, 56]])
```


## 4.2 Advanced Indexing

Advanced indexing is triggered when the `obj` is

* a non-tuple sequence object
* an ndarray(of data type integer or bool)
* a tuple with at least one sequence object or ndarray(of data type integer or bool)


There are two types of advanced indexing: integer and Boolean. Advanced indexing always returns a copy of the data.


### 4.2.1 Integer Array Indexing

Integer array indexing allows selection of arbitrary items in the array based on their N-dimensional index. For an ndarray with `n` dimension, we need `n` integer index arrays; one for each dimension. If the `n` integer index arrays don't have the same shape, **broadcasting** may be applied. In final, all the integer index arrays should have the same shape.


```python
>>> x = np.arange(24).reshape(2, 3, 4)
>>> index1 = np.array([0, 1])
>>> index2 = np.array([[2, 1], [0, 2]])
>>> index3 = np.array([[3, 2], [1, 0]])
>>> x[index1, index2, index3]
array([[11, 18],
       [ 1, 20]])
```

In this example, `x` has 3 dimensions, thus we need 3 integer index arrays. `index1` is for the first dimension, `index2` is for the second dimension, and `index3` is for the third dimension. As `index1`'s shape is (2, ), broadcasting will be appiled, and `index1` turned into `np.array([[0, 1], [0, 1]])`. In this way, the first resulting element is `x[0, 2, 3]`, the second is `x[1, 1, 2]`, and so on. After broadcasting, each integer index array has the shape `(2, 2)`, therefore the resulting array have the shape `(2, 2)`.

### 4.2.2 Boolean Array Indexing

Boolean array indexing is used to select elements from an array based on **logical conditions**. The simplest case is like

```python
>>> x = np.arange(5)
>>> index = [True, True, False, False, True]
>>> x[index]
array([0, 1, 4])
```

A common use case for Boolean index array is filtering for desired element values, e.g. select all elements which are not NaN.

```python
>>> x = np.array([[0, 1], [np.nan, 2], [np.nan, np.nan]])
>>> x[~np.isnan(x)]
array([0, 1, 2])
```

### 4.2.3 Combing Integer and Boolean Array Indexing

```python
>>> x = np.arange(24).reshape(2, 3, 4)
>>> index1 = np.array([True, False])
>>> index2 = np.array([[2, 1], [0, 2]])
>>> index3 = np.array([[3, 2], [1, 0]])
>>> x[index1, index2, index3]
array([[11,  6],
       [ 1,  8]])
```

In this example, `index1` for the first dimension, means select the first element in the first dimension. Therefore, the indeies are `(0, 2, 3), (0, 1, 2), (0, 0, 1), (0, 2, 0)`.



## 4.3 Combing Basic and Advanced Indexing


Actually, when combing basic and advanced indexing, the rules are still same, element will be chosen one by one, and a copy of the original array will be returned.

```python
>>> x = np.arange(24).reshape(2, 3, 4)
>>> index1 = np.array([0, 1])
>>> index3 = np.array([[3, 2], [0, 2]])
# index2 use the basic indexing
>>> y = x[index1, :, index3]
>>> y
array([[[ 3,  7, 11],
        [14, 18, 22]],

       [[ 0,  4,  8],
        [14, 18, 22]]])
>>> np.shares_memory(x, y)
False
```

In this example, broadcasting are applied first, turn `index1` into `np.array([[0, 1], [0, 1]])`. `:` in the second dimension is a basic indexing, means select all the elements in the second dimension, that is `[0, 1, 2]`. Therefore, the result will be:

```python
>>> result = np.array([
    [
        [x[0, 0, 3], x[0, 1, 3], x[0, 2, 3]],
        [x[1, 0, 2], x[1, 1, 2], x[1, 2, 2]]
    ],
    [
        [x[0, 0, 0], x[0, 1, 0], x[0, 2, 0]],
        [x[1, 0, 2], x[1, 1, 2], x[1, 2, 2]]
    ]
])
>>> np.allclose(result, y)
True
>>> result.shape
(2, 2, 3)
```

Another example:

```python
>>> x = np.arange(24).reshape(2, 3, 4)
>>> index1 = np.array([0, 1])
>>> index2 = np.array([[1, 2], [0, 2]])
>>> y = x[index1, index2, 0]
>>> y
array([[ 4, 20],
       [ 0, 20]])
>>> result = np.array([
    [x[0, 1, 0], x[1, 2, 0]],
    [x[0, 0, 0], x[1, 2, 0]]
])
>>> np.allclose(result, y)
True
>>> np.shares_memory(x, y)
False
```

### 4.4 Summary Table

**<center>Table3: Basic Indexing and Advanced Indexing</center>**

| | Basic Indexing | Advanced Indexing | Combined |
| ---- | ---- | ---- | ---- |
| obj | integer, slice, ellipsis, newaxis | contains non-tuple seq obj or ndarray | combined |
| view or copy | view | copy | copy |