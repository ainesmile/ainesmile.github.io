---
layout: post
title:  "异常控制流 - 异常"
date:   2016-03-12 18:10:20 +0800
categories: OS exception
---

### 异常
**异常**（exception）是控制流的突变，用来相应处理器状态中的某些变化。每种类型的异常都被分配了一个唯一的非负整数的异常号（exception number）。在系统启动时，操作系统分配和初始化异常表（exception table），使得条目 k 包含异常号为 k 的**异常处理程序**的地址。异常表的起始地址存放在异常表基址寄存器（exception table base register）。

在任何情况下，当处理器检测到有**事件**发生时，处理器会执行间接过程调用：

1. 在跳转至异常处理程序之前，处理器将返回地址和一些额外的处理器状态压入栈中；
2. 根据异常号 k， 通过异常表，转到相应的异常处理程序；
3. 执行“从中断返回”的指令，可选的返回到被中断的程序，将状态弹回到处理器的控制和数据寄存器中。

### 异常的类别

异常可以分为四类： 中断（interrupt）、陷阱（trap）、故障（fault）和终止（abort）。其中，**中断**是异步的。
I/O 设备向处理器芯片上的一个引脚发信号，并将异常号放到系统总线上，触发中断。在当前指令执行完成之后，处理器检测到中断引脚的电压变高了，就从系统总线读取异常号，然后执行间接过程调用。当处理程序返回时，将控制返回给下一条指令。而陷阱、故障和终止都是同步的。

**陷阱**是有意的异常，陷阱最重要的用途是在用户程序和内核之间提供**系统调用**。用户程序执行“syscal n”指令，请求服务 n，syscall 指令触发陷阱。异常处理程序对参数解码，并调用适当的内核程序。系统调用运行在内核模式中。经典的示例是读文件、创建新的进程、加载一个新的程序和终止当前进程

**故障**是由错误情况引起的，它可能能够被故障处理程序修正。当故障发生时，处理器将控制转移给故障处理程序。如果故障处理程序能够修正这个错误情况，就将控制返回给当前指令，否则，返回到内核的 abort 例程。一个经典的故障示例是缺页异常。

**终止**是不可恢复的致命错误造成的结果，通常是一些硬件错误，如 DRAM 和 SRAM 位被损坏时发生的奇偶错误。终止处理程序将控制返回给一个 abort 例程，终止应用程序。

### Linux/IA32 系统中的异常

在 IA32 系统上，系统调用是通过一条称为 int n 的陷阱指令来完成的，其中 n 可能是异常表 256 个条目中的任何一个。在历史上，系统调用是通过异常 128（0x80）。

{% highlight C %}
int main() {
    write(1, "hello world", 13);
    exit(0);
}

{% endhighlight %}

{% highlight asm %}
// assemble
.section .data
string:
    .ascii "hello, world\n"
string_end:
    .equ len, string_end - string

.section .text
.global main
main:
    movl $4, %eax
    movl $1, %ebx
    movl $string, %ecx
    movl $len, %edx
    int $0x80

    movl $1, %eax
    movl $0, %ebx
    int $0x80
{% endhighlight %}

### 相关定义
{% highlight html%}
事件：在处理器中，状态被编码为不同的位和信号，状态的称为成为事件。
异常处理程序：异常处理程序（exception handler）是一个专门设计用来处理事件的操作系统子程序。
异步异常： 异步异常是指由处理器外部的 I/O 设备中的事件产生的异常，而同步异常是执行一条指令的直接产物。
{% endhighlight %}

### 笔记来源
[深入理解计算机系统（原书第2版）](http://www.amazon.cn/gp/product/B004BJ18KM/ref=as_li_ss_tl?ie=UTF8&camp=536&creative=3132&creativeASIN=B004BJ18KM&linkCode=as2&tag=soasme-23)