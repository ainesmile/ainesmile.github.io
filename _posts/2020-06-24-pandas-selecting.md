---
layout: post
title:  "How to Select Data In Pandas"
date:   2020-06-24 +1200
keywords: Pandas
categories: Machine-Learning
---

In this article, we will talk about different ways of selecting data in Pandas, including selecting by [label](#seclabel), selecting by [position](#secposition), selecting by [bool condition](#secbool),  selecting by [value](#secvalue), and selecting by [dtypes](#secdtypes). Among these, using `.loc[]` and `.iloc[]` through labels and indices are the most common ways.


## 1. Prepare Data
```python
# Series
>>> data_s = np.arange(5)
>>> index_s = list('abcde')
>>> s = pd.Series(data_s, index=index_s)
# DataFrame
>>> data_df = np.arange(20).reshape(5, 4)
>>> columns_df = list('ABCD')
>>> index_df = list('abcde')
>>> df = pd.DataFrame(data_df, index=index_df, columns=columns_df)
>>> s
a    0
b    1
c    2
d    3
e    4
dtype: int64
>>> df
	A	B	C	D
a	0	1	2	3
b	4	5	6	7
c	8	9	10	11
d	12	13	14	15
e	16	17	18	19
# multi-dtypes DataFrame
>>> df_types = pd.DataFrame({
    'int': ['a', 'b', 'c'],
    'number': [1, 2, 3],
    'bool': [True, False, True],
    'datetime': pd.to_datetime(['20200101', '20200102', '20200103']),
    'cat': pd.Categorical(['a', 'b', 'c']),
    'timedelta': [pd.Timedelta(1, unit='d'), pd.Timedelta(2, unit='d'), pd.Timedelta(3, unit='d')]
})
>>> df_types.dtypes
str                   object
int                    int64
bool                    bool
datetime      datetime64[ns]
cat                 category
timedelta    timedelta64[ns]
dtype: object
# Datatime DataFrame
>>> ts_index = pd.date_range('2020-01-01', periods=30, freq='4h30min')
>>> ts = pd.DataFrame({'A': np.arange(0, 30), 'B': np.arange(30, 60)}, index=ts_index)
```



## 2. <a name="seclabel"></a>Selecting Data By Label

In Pandas, we mainly use [`[]`](#[]), [`.loc`](#loc), [`.xs`](#xs), [`.at`](#at), [`.get`](#get), and [`.lookup`](#lookup) to select data by label. Basically, we use
	
* a single label
* a list of labels
* a slice of labels

as selecting parameters. Raise `KeyError` if a label does not exist in DataFrame or Series. In this section, we will show how to select data by label with some examples. 




### 2.1 <a name="[]"></a>`[]`

`DataFrame[]` and `Series[]` support a single label and a list of labels. `DataFrame[]` also supports a slice of row labels, both the start and the stop are included.

```python
# 1. a single label
>>> s['a']
0
>>> df['A']
a     0
b     4
c     8
d    12
e    16
Name: A, dtype: int64
# 2. a list of labels
>>> s[['a', 'b']]
a    0
b    1
dtype: int64
>>> df[['A', 'B']]
	A	B
a	0	1
b	4	5
c	8	9
d	12	13
e	16	17
# 3. a slice of row labels
>>> df['a':'c']
	A	B	C	D
a	0	1	2	3
b	4	5	6	7
c	8	9	10	11
```

### 2.2 <a name="loc"></a>`.loc[]`

`Series.loc` and `DataFrame.loc`: 

> Access a group of rows and columns by label(s) or a boolean array. Axes left out of the specification are assumed to be `:`.

Support a single label, a list of labels and a slice object with labels.

```python
# 1. a single label
# equals to df.loc['a', :]
>>> df.loc['a']
A    0
B    1
C    2
D    3
Name: a, dtype: int64
>>> df.loc['a', 'A']
0
>>> s.loc['a']
0
# 2. a list or array of labels
# equals to df.loc[['a', 'b'], :]
>>> df.loc[['a', 'b']]
	A	B	C	D
a	0	1	2	3
b	4	5	6	7
>>> df.loc[['a', 'b'], ['A', 'B']]
	A	B
a	0	1
b	4	5
>>> s.loc[['a', 'b']]
a    0
b    1
dtype: int64
# 3. a slice object with labels, both the start and the stop are included
# equals to df.loc['a': 'c', :]
>>> df.loc['a': 'c']
	A	B	C	D
a	0	1	2	3
b	4	5	6	7
c	8	9	10	11
>>> df.loc['a': 'c', 'A':'C']
	A	B	C
a	0	1	2
b	4	5	6
c	8	9	10
>>> df.loc[['a', 'b'], ['A', 'B']]
	A	B
a	0	1
b	4	5
>>> s.loc['a':'b']
a    0
b    1
dtype: int64
```

### 2.3 <a name="get"></a>`.get()`

`DataFrame.get(self, key, default=None)`:

> Get item from object for given key. Returns default value if not found.

```python
>>> s.get('a')
0
>>> s.get(key='A', default='default')
'default'
# DataFrame: column names only
>>> df.get(key='A')
a     0
b     4
c     8
d    12
e    16
Name: A, dtype: int64
>>> df.get(key='a', default='default')
'default'
```


### 2.4 <a name="at"></a>`.at()`

`DataFrame.at` and `Series.at`

> Access a single value for a row/column label pair. Use `at` if you only need to get or set a single value in a DataFrame or Series.


```python
>>> s.at['a']
0
>>> df.at['a', 'A']
0
```



### 2.5 <a name="xs"></a>`.xs()`

`DataFrame.xs(self, key, axis=0, level=None, drop_level:bool=True)`

> Return cross-section from the Series/DataFrame.

Suit for MultiIndex.

```python
>>> s.xs(key='a')
0
# equals to df.xs(key='a', axis=0)
>>> df.xs('a')
A    0
B    1
C    2
D    3
Name: a, dtype: int64
# equals to df.xs(key='A', axis=1)
>>> df.xs('A', axis=1)
a     0
b     4
c     8
d    12
e    16
Name: A, dtype: int64
```


### 2.6 <a name="lookup"></a>`.lookup()`

`DataFrame.lookup(self, row_labels, col_labels) -> numpy.ndarry`:

> Label-based "fancy indexing" function for DataFrame. Given equal-length arrays of row and column labels, return an array of the values corresponding to each (row, col) pair.

```python
>>> df.lookup(row_labels=['a', 'a'], col_labels=['A', 'C'])
array([0, 2])
```

Raise `ValueError` first if row labels and column labels are not the same size.


### 2.7 `.filter()`

`DataFrame.filter(items=None, like=None, regex=None, axis=None) -> DataFrame or Series`:

> Subset the dataframe rows or columns according to the specified index labels. The filter is applied to the labels of the index.

Keyword arguments `items`, `like`, or `regex` are mutually exclusive


```python
# 1. a single label
>>> df.filter(items='A')
	A
a	0
b	4
c	8
d	12
e	16
>>> df.filter(items='a', axis=0)
	A	B	C	D
a	0	1	2	3
>>> s.filter(items='a')
a    0
dtype: int64
# 2. a list of labels
>>> df.filter(items=['A', 'C', 'Z'])
	A	C
a	0	2
b	4	6
c	8	10
d	12	14
e	16	18
>>> df.filter(items=['a', 'c'], axis=0)
	A	B	C	D
a	0	1	2	3
c	8	9	10	11
>>> s.filter(items=['a', 'c'])
a    0
c    2
dtype: int64
# 3. containing str
>>> df_types.filter(like='date')
	datetime
0	2020-01-01
1	2020-01-02
2	2020-01-03
# 4. regular expression
>>> df_types.filter(regex='time?')
datetime	timedelta
0	2020-01-01	1 days
1	2020-01-02	2 days
2	2020-01-03	3 days
```



### 2.8 Summary Table Of Selecting By Label

<table>
	<caption>Table1: Selecting From DataFrame By Label</caption>
	<!-- Header -->
	<tr>
		<th colspan="2">DataFrame</th>
		<th><code>df[]</code></th>
		<th><code>df.loc[]</code></th>
		<th><code>df.get()</code></th>
		<th><code>df.at()</code></th>
		<th><code>df.xs()</code></th>
		<th><code>df.lookup()</code></th>
		<th><code>df.filter()</code></th>
	</tr>
	<!-- Row Label -->
	<tr>
		<td rowspan="3">Row Label</td>
		<td>a single row label<code>'a'</code></td>
		<td>&#10008;</td>
		<td>&#10004;<code>s</code></td>
		<td>&#10004;Default</td>
		<td>&#10008;</td>
		<td>&#10004;<code>s</code></td>
		<td>&#10008;</td>
		<td>&#10004;<code>axis=0, ->s</code></td>
	</tr>
	<tr>
		<td>a list of row labels <code>['a', 'b']</code></td>
		<td>&#10008;</td>
		<td>&#10004;<code>df</code></td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10004;<code>axis=0, ->s</code></td>
	</tr>
	<tr>
		<td>a slice of row labels <code>slice('a', 'c')</code></td>
		<td>&#10004;<code>df</code></td>
		<td>&#10004;<code>df</code></td>
		<td>&#10004;<code>df</code></td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10008;</td>
	</tr>
	<!-- Column Label -->
	<tr>
		<td rowspan="3">Column Label</td>
		<td>a single column label<code>'A'</code></td>
		<td>&#10004;<code>s</code></td>
		<td>&#10008;</td>
		<td>&#10004;<code>s</code></td>
		<td>&#10008;</td>
		<td>&#10004;<code>axis=1,->s</code></td>
		<td>&#10008;</td>
		<td>&#10004;<code>s</code></td>
	</tr>
	<tr>
		<td>a list of column labels <code>['A', 'B']</code></td>
		<td>&#10004;<code>df</code></td>
		<td>&#10008;</td>
		<td>&#10004;<code>df</code></td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10004;<code>s</code></td>
	</tr>
	<tr>
		<td>a slice of column labels <code>slice('A', 'B')</code></td>
		<td>&#10004;<code>empty df</code></td>
		<td>&#10004;<code>empty df</code></td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10008;</td>
	</tr>
	<!-- Pair of Row Label and Column Label (row_label, col_label) -->
	<tr>
		<td rowspan="3">Combined</td>
		<td>single label <code>'a', 'A'</code></td>
		<td>&#10008;</td>
		<td>&#10004;<code>Scalar</code></td>
		<td>&#10004;'A'</td>
		<td>&#10004;<code>Scalar</code></td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10008;</td>
	</tr>
	<tr>
		<td>list of labels <code>['a', 'b'], ['A', 'B']</code></td>
		<td>&#10008;</td>
		<td>&#10004;<code>df</code></td>
		<td>&#10004;<code>['A', 'B']</code></td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10004;<code>np.ndarray</code></td>
		<td>&#10008;</td>
	</tr>
	<tr>
		<td>slice of labels <code>slice('a','b'), slice('A','B')</code></td>
		<td>&#10008;</td>
		<td>&#10004;<code>df</code></td>
		<td>&#10004;<code>df</code></td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10008;</td>
		<td>&#10008;</td>
	</tr>
</table>



<!-- ### 2.7 `.pop`

`DataFrame.pop(col_name) -> Series`:

> Return item and drop from frame. Raise KeyError if not found.

`DataFrame.pop()` will modify the original DataFrame.

```python
>>> df.pop('A')
a     0
b     4
c     8
d    12
e    16
Name: A, dtype: int64
>>> df
	B	C	D
a	1	2	3
b	5	6	7
c	9	10	11
d	13	14	15
e	17	18	19
``` -->


<br>


## 3. <a name="secposition"></a>Selecting Data By Position


In Pandas, we mainly use [`[]`](#[]2), [`.head`](#headtail), [`.tail`](#headtail), [`.iloc`](#iloc), [`.iat`](#iat), and [`take`](#take) to select data by position. Raise `IndexError` if a requested indexer is out-of-bounds, except slice indexers which allow out-of-bounds indexing(same with python/numpy slice semantics).


### 3.1 <a name="[]2"></a>`[]`
Besides labels, `DataFrame[]` and `Series[]` also support slice object of integers, and `Series[]` supports integers. The slice object `slice(i, j, k)` has the same semantics as NumPy. Therefore, for negative values, a negative `i` equals to `n+i`, and a negative `j` equals to `n+j`, and a negative `k` means go in reverse order.

```python
# 1. a single int, Series only
>>> s[0]
0
>>> df[0]
KeyError
# 2. a list of ints, Series only
>>> s[[0, 1]]
a    0
b    1
dtype: int64
# 3. a slice object, both for DataFrame and Series
# equals to s.iloc[0:2]
>>> s[0:2]
a    0
b    1
dtype: int64
>>> df[0:4:2]
	A	B	C	D
a	0	1	2	3
c	8	9	10	11
# euqals to df[5-3:5-1]
>>> df[-3:-1]
	A	B	C	D
c	8	9	10	11
d	12	13	14	15
```

### 3.2 <a name="headtail"></a>`.head()` and `.tail()`

`DataFrame.head(n:int=5)` and `Series.head(n:int=5)`: return the first n rows.
`DataFrame.tail(n:int=5)` and `Series.head(n:int=5)`: return the last n rows.

For `n<0`, `df.head(n)` equals to `df.head(len+n)`, and `df.tail(n)` equals to `df.tail(len+n)`, where `len` is the number of rows.

For `n=0`, returns empty Series or DataFrame.


```python
>>> s.head(n=2)
a    0
b    1
dtype: int64
# equals to df.head(n=5-3)
>>> df.head(n=-3)
	A	B	C	D
a	0	1	2	3
b	4	5	6	7
```

### 3.3 <a name="iloc"></a>`.iloc[]`

`Series.iloc` and `DataFrame.iloc`

> Purely integer-location based indexing for selection by position.



```python
# 1. a single integer
# equals to df.iloc[0, :]
>>> df.iloc[0]
A    0
B    1
C    2
D    3
Name: a, dtype: int64
>>> s.iloc[0]
0
# 2. a list or array of integers
# euqals to df.iloc[[0, 2], :]
>>> df.iloc[[0, 2]]
	A	B	C	D
a	0	1	2	3
c	8	9	10	11
>>> s.iloc[[0, 1]]
a    0
b    1
dtype: int64
# 3. a slice object with ints, the stop excluded
# equals to df.iloc[0:2, :]
>>> df.iloc[0:2]
	A	B	C	D
a	0	1	2	3
b	4	5	6	7
>>> df.iloc[0:2, 0:2]
	A	B
a	0	1
b	4	5
>>> s.iloc[0:2]
a    0
b    1
dtype: int64
```

### 3.4 <a name="iat"></a> `.iat[]`

`DataFrame.iat` and `Series.iat`

> Access a single value for a row/column pair by integer position. Use `iat` if you only need to get or set a single value in a DataFrame or Series.



```python
>>> s.iat[0]
0
>>> df.iat[0, 1]
1
```


### 3.5 <a name="take"></a> `.take()`

`DataFrame.take(indices, axis=0, **kwargs)`: 

> Return the elements in the given positional indices along an axis. This means that we are not indexing according to actual values in the index attribute of the object. We are indexing according to the atual position of the element in the object.

For `n<0`, equals to `len+n`.

```python
>>> df.take([0, 2], axis=0)
	A	B	C	D
a	0	1	2	3
c	8	9	10	11
# equals to df.take([5-3, 5-1], axis=1)
>>> df.take([-3, -1], axis=1)
	B	D
a	1	3
b	5	7
c	9	11
d	13	15
e	17	19
```







## 4. <a name="secvalue"></a>Selecting Data By Value

### 4.1 Selecting By Bool

In Pandas, we can use `.[]`, `.loc`, and `.query()` to subset based on bool.

```python
>>> bool_row = pd.Series({'a': False, 'b': False, 'c': False, 'd': True, 'e': True})
>>> bool_col = pd.Series({'A': False, 'B': True, 'C': False, 'D': True})
>>> df.loc[bool_row, bool_col]
	B	D
d	13	15
e	17	19
# equals to df.loc[df['A']>10]
>>> df[df['A'] > 10]
	A	B	C	D
d	12	13	14	15
e	16	17	18	19
```

`DataFrame.query(expr, inplace=False, **kwargs)`:

> Query the columns of a DataFrame with a boolean expression. This method uses the top-level `eval()` function to evaluae the passed query. The result of the evaluation of this expression is first passed to `DataFrame.loc`.


```python
# equals to df.loc[eval('df.A>10')]
>>> df.query('A>10')
	A	B	C	D
d	12	13	14	15
e	16	17	18	19
```



### 4.2 Selecting By Value Order

`DataFrame.nlargest(n, columns, keep='first) -> DataFrame`:

> Return the first n rows ordered by columns in descending order. The columns that are not specified are returned as well, but not used for ordering. This method is equivalent to `df.sore_values(columns, ascending=False).head(n)`, but more performant.

Similarly, `DataFrame.nsmallest(n, columns, keep='first') -> DataFrame`:

> Return the first n rows ordered by columns in ascending order. This method is equivalent to `df.sort_values(columns, ascending=True).head(n)`, but more performant.


```python
>>> df.nlargest(n=2, columns=['A'])
	A	B	C	D
e	16	17	18	19
d	12	13	14	15
>>> df.nsmallest(n=2, columns=['A'])
	A	B	C	D
a	0	1	2	3
b	4	5	6	7
```

### 4.3 Selecting By Datatime

For datetime data, we can use `DataFrame.first()`, `DataFrame.last()`, `DataFrame.at_time()`, `DataFrame.bewteen_time()` to select subset based on datetime.

```python
# subset initial periods of time series data based on a date offset
>>> ts.first('12h')
	A	B
2020-01-01 00:00:00	0	30
2020-01-01 04:30:00	1	31
2020-01-01 09:00:00	2	32
# subset final periods of the time series data based on a date offset
>>> ts.last('12h')
	A	B
2020-01-06 01:30:00	27	57
2020-01-06 06:00:00	28	58
2020-01-06 10:30:00	29	59
# select values at particulat time of day
>>> ts.at_time('12:00')
	A	B
2020-01-02 12:00:00	8	38
2020-01-05 12:00:00	24	54
# select values between particular times of the day
>>> ts.between_time('10:00', '12:00')
	A	B
2020-01-02 12:00:00	8	38
2020-01-03 10:30:00	13	43
2020-01-05 12:00:00	24	54
2020-01-06 10:30:00	29	59
```


## 5. <a name="secdtypes"></a>Selection Data By Dtypes

`pd.DataFrame.select_dtypes(self, include=None, exclude=None) -> DataFrame`:

> Return a subset of the DataFrame's columns based on the column dtypes.

**<center>Table 2: Selecting Data By Dtypes</center>**

| dtype | parameters |
| ---- | ---- |
| numeric | np.number/number |
| strings | object |
| datetimes | np.datetime64/datetime/datetime64 |
| timedeltas | np.timedelta64/timedelta/timedelta64 |
| categorical | category |
| datetimetz | datetimetz/datetime64[ns, tz] |

<br>

```python
>>> df_types.select_dtypes(['object'])
	str
0	a
1	b
2	c
>>> df_types.select_dtypes(['number'])
	int	timedelta
0	1	1 days
1	2	2 days
2	3	3 days
>>> df_types.select_dtypes(['timedelta'])
	timedelta
0	1 days
1	2 days
2	3 days
>>> df_types.select_dtypes(exclude=['category', 'datetime'])
	str	int	bool	timedelta
0	a	1	True	1 days
1	b	2	False	2 days
2	c	3	True	3 days
```