---
layout: post
title:  "NumPy: Joining Arrays Together"
date:   2020-04-06 +1100
keywords: NumPy
categories: Machine-Learning
---


NumPy provides several functions that allow us to join arrays, including **stack**, **concatenate**, **vstack**, **hstack**, **dstack**, **column_stack**, and **block**. In this article, we will:

* describe these joining array functions from the perspective of shape, ndim and axis
* give more formalize descriptions and more examples
* compare them to each other


## 1. np.stack

> `numpy.stack(arrays, axis=0, out=None)`: Join a sequence of arrays along a **new** axis.

So the number of dimensions will increase 1. And all the input arrays must have the same shape. With input arrays
`a_0, a_1, ... a_(n-1)`, each array has the same shape `(s_0, s_1, ... s_(m-1))` and the same number of dimensions `m`, then 
 `stacked_array = np.stack((a_0, a_1, ... a_(n-1)), axis=d))` will have the shape

```python
for 0 <= d <= m-1,

if d == 0: shape = (n, s_0, ..., s_(m-1)), ndim = m + 1

if d == i(0<i<m-1): shape = (s_0, ..., s_(i-1), n, s_(i+1), ..., s_(m-1)), ndim = m + 1

if d == m-1: shape = (s_0, s_1, ..., n, s_(m-1)), ndim = m + 1
```

For example:

```python
>>> shape = (2, 3, 4, 5)
>>> a1 = np.arange(120).reshape(shape)
>>> a2 = np.arange(120, 120*2).reshape(shape)
>>> a3 = np.arange(120*2, 120*3).reshape(shape)
>>> a4 = np.arange(120*3, 120*4).reshape(shape)
>>> a5 = np.arange(120*4, 120*5).reshape(shape)
>>> a6 = np.arange(120*5, 120*6).reshape(shape)
>>> arrays = [a1, a2, a3, a4, a5, a6]
>>> r0 = np.stack(arrays, axis=0)
>>> r1 = np.stack(arrays, axis=1)
>>> r2 = np.stack(arrays, axis=2)
>>> r3 = np.stack(arrays, axis=3)
>>> r4 = np.stack(arrays, axis=4)
>>> r0.shape, r1.shape, r2.shape, r3.shape, r4.shape
((6, 2, 3, 4, 5),
 (2, 6, 3, 4, 5),
 (2, 3, 6, 4, 5),
 (2, 3, 4, 6, 5),
 (2, 3, 4, 5, 6))
```


## 2. np.concatenate

> `np.concatenate(arrays, axis=0, out=None)`: Join a sequence of arrays along an **existing** axis.

So the number of dimension will not increase. It requires all arrays must have the same shape, except in the dimension corresponding to axis. If `axis=None`, the number of dimension will be 1, arrays will be flattened.


If we want to execute `concatenated_array=np.concatenate((a_0, ..., a_(n-1)), axis=d, out=None)`, where `d` is an integer, the input arrays `a_0, ..., a_(n-1)` must be

```python
for 0 <= d <= m-1 and for all 0 <= j <= n-1: 

if d = 0, a_j.shape = (t_j, s_1, ..., s_(m-1))
if d = i(0<i<m-1), a_j.shape = (s_0, ..., s_(i-1), t_j, s_(i+1), ..., s_(m-1))
if d = m-1, a_j.shape = (s_0, ..., s_(m-2), t_j)
```

then the `concatenated_array` will remain the same shape except the corresponding axis, which will be the sum of all the arrays, that is

```python
for 0 <= d <= m-1, 

if d = 0, concatenated_array.shape = (t, s_1, ..., s_(m-1))
if d = i(0<i<m-1), concatenated_array.shape = (s_0, ..., s_(i-1), t, s_(i+1), ..., s_(m-1))
if d = m-1, concatenated_array.shape = (s_0, ..., s_(m-2), t)

where t = sum(t_0, ..., t_(n-1))
```

For example:

```python
>>> a0 = np.arange(120).reshape(2, 3, 4, 5)
>>> a1 = np.arange(120, 150).reshape(2, 3, 1, 5)
>>> a2 = np.arange(150, 330).reshape(2, 3, 6, 5)
>>> concatenate_array = np.concatenate((a0, a1, a2), axis=2)
>>> concatenate_array.shape, concatenate_array.ndim
((2, 3, 11, 5), 4) # 11 = 4+1+6
```

## 3. np.vstack

> `np.vstack(tup)`: Stack arrays in sequence vertically(row wise).
> This is equivalent to concatenation along the first axis after 1-D arrays of shape(N, ) have been reshaped to (1, N).

`np.vstack` requires the input arrays have the same shape along all but the first axis. 1-D arrays must have the same length. Therefore, `np.vstack(arrays) == np.concatenate(arrays, axis=0)`.

For example:

```python
>>> a0 = np.arange(120).reshape(4, 2, 3, 5)
>>> a1 = np.arange(120, 150).reshape(1, 2, 3, 5)
>>> a2 = np.arange(150, 330).reshape(6, 2, 3, 5)
>>> r0 = np.vstack((a0, a1, a2))
>>> r1 = np.concatenate((a0, a1, a2), axis=0)
>>> np.allclose(r0, r1)
True
```

## 4. np.hstack

> `np.hstack(tup)`: Stack arrays in sequence horizontally(column wise). This is equivalent to concatenation along the second axis, except for 1-D arrays where it concatenates along the first axis.

Therefore,

```python
1-D arrays:
np.hstack(arrays) == np.concatenate(arrays, axis=0) == np.concatenate(arrays, axis=None)

arrays >= 2-D:
np.hstack(arrays) == np.concatenate(arrays, axis=1)
```


For example,

```python
# 1-D arrays
>>> a = np.array([1,2,3])
>>> b = np.array([2,3,4])
>>> hstacked = np.hstack((a,b))
>>> concatenated_0 = np.concatenate((a, b), axis=0)
>>> concatenated_1 = np.concatenate((a, b), axis=None)
>>> np.allclose(hstacked, concatenated_0)
True
>>> np.allclose(hstacked, concatenated_1)
True

# arrays >= 2-D
>>> a0 = np.arange(120).reshape(2, 4, 3, 5)
>>> a1 = np.arange(120, 150).reshape(2, 1, 3, 5)
>>> a2 = np.arange(150, 330).reshape(2, 6, 3, 5)
>>> r0 = np.hstack((a0, a1, a2))
>>> r1 = np.concatenate((a0, a1, a2), axis=1)
>>> np.allclose(r0, r1)
True
```

## 5. np.dstack

> `np.dstack(tup)`: Stack arrays in sequence depth wise(along third axis).

This is equivalent to concatenation along the third axis. For arrays more than 2-D, `np.dstack(arrays) == np.concatenate(arrays, axis=2)`. 1-D arrays with shape(N,) will be reshaped to (1,N,1), and 2-D arrays with shape(M, N) will be reshaped to (M, N, 1). After reshape, 1-D arrays and 2-D arrays have at least 3 dimensions, `axis=2` will be okay.

Therefore,

```python
1-D arrays with shape (N, ):
np.dstack(arrays) == np.concatenate(arrays.reshape(1, N, 1), axis=2)

2-D arrays with shape (M, N):
np.dstack(arrays) == np.concatenate(arrays.reshape(M, N, 1), axis=2)

>= 3-D arrays:
np.dstack(arrays) == np.concatenate(arrays, axis=2)
```

For example:

```python
1-D arrays
>>> a0 = np.arange(6)
>>> a1 = np.arange(6, 12)
>>> a2 = np.arange(12, 18)
>>> r0 = np.concatenate((a0.reshape(1, 6, 1), a1.reshape(1, 6, 1), a2.reshape(1, 6, 1)), axis=2)
>>> r1 = np.dstack((a0, a1, a2))
>>> np.allclose(r0, r1)
True

2-D arrays
>>> a0 = np.arange(6).reshape(2, 3)
>>> a1 = np.arange(6, 12).reshape(2, 3)
>>> a2 = np.arange(12, 18).reshape(2, 3)
>>> r0 = np.concatenate((a0.reshape(2, 3, 1), a1.reshape(2, 3, 1), a2.reshape(2, 3, 1)), axis=2)
>>> r1 = np.dstack((a0, a1, a2))
>>> np.allclose(r0, r1)
True

>= 3-D arrays
>>> a0 = np.arange(120).reshape(2, 3, 4, 5)
>>> a1 = np.arange(120, 150).reshape(2, 3, 1, 5)
>>> a2 = np.arange(150, 330).reshape(2, 3, 6, 5)
>>> r0 = np.concatenate((a0, a1, a2), axis=2)
>>> r1 = np.dstack((a0, a1, a2))
>>> np.allclose(r0, r1)
```



## 6. np.column_stack

`np.column_stack(tup)` equals to concatenate arrays along the second axis, 1-D arrays with shape(N, ) will be reshape to (N, 1).

Therefore,

```python
1-D arrays:
np.column_stack(arrays) == np.concatenate(arrays.reshape(-1, 1), axis=1)

>= 2-D arrays:
np.column_stack(arrays) == np.concatenate(arrays, axis=1) == np.hstack(arrays)
```

For example:

```python
# 1-D arrays
>>> a = np.array([1,2,3])
>>> b = np.array([2,3,4])
>>> r0 = column_stack((a, b))
>>> r1 = np.concatenate((a.reshape(-1, 1), b.reshape(-1, 1)), axis=1)
>>> np.allclose(r0, r1)
>>> True

# arrays >= 2-D
>>> a0 = np.arange(120).reshape(2, 4, 3, 5)
>>> a1 = np.arange(120, 150).reshape(2, 1, 3, 5)
>>> a2 = np.arange(150, 330).reshape(2, 6, 3, 5)
>>> r0 = np.column_stack((a0, a1, a2))
>>> r1 = np.concatenate((a0, a1, a2), axis=1)
>>> r2 = np.hstack((a0, a1, a2))
>>> np.allclose(r0, r1), np.allclose(r0, r1)
(True, True)
```

## 7. np.block

> `np.block(arrays)`: Assemble an ndarray from nested lists of blocks.

> Blocks in the innermost lists are concatenated along the last dimension(-1), then these are concatenated along the second-last dimension(-2), and so on until the outermost list is reached.

Therefore, `np.block(arrs)` will execute `r_0_i = concatenate((innermost_lists_i), axis=-1)` first, and then execute `r_1_j = concatenate([..., r_0_i...,], axis=-2)`, and so on until the outermost list is reached.

For example:

```python
>>> A11 = np.ones((2, 3, 1))
>>> A12 = np.zeros((2, 3, 5))
>>> A21 = np.ones((2, 3, 4))
>>> A22 = np.zeros((2, 3, 2))

>>> B11 = np.zeros((1, 3, 2))
>>> B12 = np.ones((1, 3, 4))
>>> B21 = np.zeros((1, 3, 1))
>>> B22 = np.ones((1, 3, 5))

# A11, A12 required to be sampe shape except the last axis
>>> A1 = np.concatenate((A11, A12), axis=-1)
>>> A2 = np.concatenate((A21, A22), axis=-1)
>>> A1.shape, A2.shape
((2, 3, 6), (2, 3, 6))

# A1, A2 required to be sampe shape except the last second axis
>>> A_con = np.concatenate((A1, A2), axis=-2)
>>> A_block = np.block([
    [[A11, A12], [A21, A22]]
])
>>> np.allclose(A_con, A_block)
True
>>> A_block.shape
(2, 6, 6)

>>> B1 = np.concatenate((B11, B12), axis=-1)
>>> B2 = np.concatenate((B21, B22), axis=-1)
>>> B_con = np.concatenate((B1, B2), axis=-2)
>>> B_block = np.block([
    [[B11, B12], [B21, B22]]
])

# A_con, B_con required to be same shape except last third axis
>>> result_con = np.concatenate((A_con, B_con), axis=-3)
>>> result_block = np.block([
    [[A11, A12], [A21, A22]],
    [[B11, B12], [B21, B22]],
])

>>> np.allclose(result_con, result_block)
True
>>> result_block.shape
(3, 6, 6)
```




## 8. summary

The different between **stack** and **concatenate** is easy to see. The main differences are including the input arrays' shape requirement and the returned array's dimension number.

**<center>Table1: Compare Stack With Concatenate</center>**

| Table1 | stack | concatenate |
| ---- | ---- | ---- |
| arrays' shape requirement | exactly same shape | same shape except the corresponding axis|
| axis values | int |  int or None|
| ndim | +1 | unchanged or = 1 |

<br>

The main differences between **vstack**, **hstack**, **dstack** and **column_stack** are and the input arrays' shape requirement and the concatenate axis. They all can be implemented through **concatenate**. Table2 describes how to implement them through **concatenate**.

**<center>Table2: Implement Through Concatenate</center>**

| Table2 | vstack | hstack | dstack | column_stack | block |
| ---- | ---- | ---- | ---- | ---- |
| along axis for >= 2-D arrays | 0 | 1 | 2 | 1 | from last to first |
| along axis for 1-D arrays  | 0 | 0 or None | 2 | 1 | None |
| reshape required for 2-D arrays | No | No | Yes | No | No |
| reshape required for 1-D arrays | No | No | Yes | Yes | No |

