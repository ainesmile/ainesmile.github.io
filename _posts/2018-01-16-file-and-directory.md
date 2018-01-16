---
layout: post
title:  "用 i-node 来理解文件访问权限和链接"
date:   2018-01-16 +1200
keywords: i-node file-access-permissions hard-links symbolic-links
categories: UNIX
---

### 1. i-node 定义

> An inode is a data structure on a filesystem on Linux and other Unix-like operation systems that stores all the information about a file except its name and its actual data.
> -- <cite>[LINFO](http://www.linfo.org/inode.html)</cite>


可见，**i-node** 是用于存储文件信息。事实是，每个 **i-node** 存储了对应该文件几乎所有的元信息，包括文件类型，文件访问权限和文件大小等，但不包括文件名、文件内容、**i-node number** 。

文件系统，是指 **directory tree**。UNIX 文件系统有多种实现，本文讨论的是 UNIX file system(UFS）。关于 **filesystem** 可以参考[LINFO, Filesystems: A Brief Introduction](http://www.linfo.org/filesystem.html)。

从磁盘的角度来说，一个磁盘可以分成一个或多个分区（**partition**），每个分区可以包含一个文件系统。**i-node** 就是磁盘分区上的长度固定的一部分，如图1所示。


![图1](/pics/partitions.png)
<center>图1 磁盘、分区和文件系统</center>
  <small>图片来源：[Advanced Programming in the UNIX Environment, 3rd Edition](https://book.douban.com/subject/11626789/)</small>


### 2. 普通文件、目录文件和符号链接

UNIX 文件类型包括普通文件（regular file）、目录文件（directory file）和符号链接（symbolic link）等7种。从 **i-node** 的角度来说，以上三种文件是一样的，主要区别在于文件内容。

当文件被创建的时候，文件名与 **i-node number** 组成一个条目，存在该文件所在目录的 **data block**。在同一个文件系统中，**i-node number** 唯一，可以通过 `stat` 函数获得，也可以通过命令 `ls -i filename` 获得。

文件由 **i-node** 和 文件内容2个部分组成。**i-node** 中包含指向文件内容的指针。

目录文件的文件内容是一张表格，表格的条目为文件名和 **i-node number**的映射；符号链接的文件内容则是被链接的源文件路径。

### 3. 文件访问权限

所有类型的文件都有访问权限（access permission）。每个文件有9个访问权限位，对应所有者、用户组和其他三组用户的读、写和执行权限。文件权限是针对文件内容而言的，**i-node** 由操作系统管理。读权限意味着可以读取文件内容，写权限意味着可以修改文件内容，执行权限意味着可以执行该文件。对目录而言，执行权限可以通过该目录。

当我们用路径来打开文件的时候，需要该路径中每一层目录的执行权限。例如，打开文件 `/Users/ainesmile/foo.md` 分别需要目录 `/`, `/Users` 和 `/Users/ainesmile` 的执行权限，当然同样需要对文件 `foo.md` 相应的权限。

当我们想要删除目录下的文件的时候，需要该目录的写权限和执行权限，并不需要该文件的权限。因为删除目录下的文件其实是在修改目录的文件内容。重命名文件同理。


### 4. hard link 与 symbolic link 的区别

一个文件只能对应一个 **i-node**，但是同一个 **i-node** 可以对应不同的文件。**hard link** 与源文件拥有相同的**i-node number**。因此，源文件和 **hard link** 是对等的，是同一个 **i-node** 的不同名字。**hard link**的文件内容与源文件相同。

而 **symbolic link** 不同， **symbolic link** 与源文件有不同的 **i-node number**。**symbolic link** 的文件内容为源文件的路径名。

```
touch foo
ln foo hard_link_foo
ln foo symbolic_link_foo
ls -i foo hard_link_foo symbolic_link_foo

8592541443 foo  8592541443 hard_link_foo    8592541449 symbolic_link_foo
```
可以看到文件 `foo` 和 `hard_link_foo` 拥有相同的 **i-node number**，而 `symbolic_link_foo` 不同。

对 **symbolic link** 的操作，如果选择 follow 这个 **symbolic link**，那么会通过源文件与源文件的文件内容关联起来。当源文件删除时，**hard link** 可以独立存在，并没有受影响。而 **symbolic link** 则无法找到源文件的文件内容。

## 笔记来源
[Advanced Programming in the UNIX Environment, 3rd Edition](https://book.douban.com/subject/11626789/)
