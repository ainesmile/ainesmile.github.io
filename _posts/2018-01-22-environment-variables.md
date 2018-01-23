---
layout: post
title:  "修改环境变量对存储空间布局的影响"
date:   2018-01-22 +1200
keywords: unix environment varibales
categories: UNIX
---

### 1. 环境表

每个程序都会有一张环境表（environment list）。 环境表是一个字符指针数组，除最后一个指针指向 `NULL` 以外，其他每个指针指向一个以 null 结尾的 C 字符串。全局变量 `environ` 包含了指向该指针数组的地址， 被称为环境指针（environment pointer），`extern char **environ`。

![图1](/pics/environment_list.png)
<center>图1 由5个字符串组成的环境</center>
  <small>图片来源：[Advanced Programming in the UNIX Environment, 3rd Edition](https://book.douban.com/subject/11626789/)</small>

### 2. 环境变量

环境字符串的形式通常为`name=value`。UNIX内核并不查看环境变量，环境变量的解释完全取决于各个应用程序。查看、修改、新增和删除环境变量可以通过以下四个函数来实现：

```C
#include <stdlib.h>

char *getenv(const char *name);

int putenv(char *str);

int setenv(const char *name, const char *value, int rewrite);

int unsetenv(const char *name);
```

### 3. C 程序的存储空间布局

图2是典型的存储器安排。由图2可知，环境变量通常存放在进程存储空间的顶部，所以不能再向高地址扩展。同时，低地址方向是栈，我们不能移动栈帧的位置，所以也不能向低地址方向扩展。因此，我们不能在进程空间顶部增加环境变量的空间长度。当我们需要增加环境变量空间长度的时候，必须调用 `malloc` 函数在堆中分配新空间。

![图2](/pics/memory_arrangement.png)

<small>图2 典型的存储器安排</small>

<small>图片来源：[Advanced Programming in the UNIX Environment, 3rd Edition](https://book.douban.com/subject/11626789/)</small>


### 4. 修改环境变量对存储空间布局的影响

从存储空间布局的角度来看，删除一个环境字符串（envrionment strings）很简单。但修改和新增一个环境字符串就困难得多，对存储空间布局的影响较大。

* 删除一个环境字符串，不需要增加环境变量空间长度，只需要在环境表中找到该指针，然后将所有后续指针都向环境表首部顺次移动一个位置。删除一个环境变量对存储空间影响不大。

* 修改一个环境字符串，

   * `strlen("name=old_value") >= strlen("name=new_value")`：只需要在 `"name=old_value"` 空间中写入新 `"name=new_value"` 即可，并不需要增加环境变量空间长度。
  
   * `strlen("name=old_value") < strlen("name=new_value")`：则必须要增加环境变量空间长度。此时，需要调用 `malloc` 在堆中分配空间，然后将 `"name=new_value"` 写入到该空间，接着使环境表中对`name`的指针指向新分配区。

* 新增一个环境字符串，

  * 首先需要调用 `malloc` 分配空间，然后将 `"name=new_value"` 写入到该空间。

  * 第一次新增，此时原有的环境表空间不够了，需要调用 `malloc` 在堆中分配一块较大的区域来存放新的环境表。在环境表的表尾存放指向 `"name=new_value"` 的指针，并将空指针存放其后。同时，更新环境指针 `environ`，使 `environ` 指向新的环境表。注意，此时环境表中的大多数指针仍指向原来的地方，即栈顶之上。

  * 不是第一次新增，则可以知道已经有新的环境表在堆中了。这个时候只需要调用 `realloc` 来新增环境表分配区的长度，在环境表的表尾存放指向 `"name=new_value"` 的指针，并将空指针存放其后。


综上可以知道，修改和新增环境变量对存储空间布局可能有较大的影响，这主要是因为原有存放环境变量的空间不够，需要在堆中分配新空间。



### 5. 笔记来源
[Advanced Programming in the UNIX Environment, 3rd Edition](https://book.douban.com/subject/11626789/)
