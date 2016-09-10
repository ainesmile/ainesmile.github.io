---
layout: post
title:  "States and Semantics"
date:   2016-09-08 17:00:00 +0800
categories: Programming predicate
---

### States

  **State space**: In this book, the **state space** is regarded as being built up as a **Cartesian product**, and the states is the points.

  **Predicate** is the formal expression of **condition**. Each predicate is assumed to be defined in each point of the state space. Given a point, the value of a predicate is either "True" or "False". Therefore, the predicate characterize the set of all points where the predicate is true. Two predicates  <code>P = Q</code> , when they characterize the same set of states.

  <code>T</code> is the predicate that is true in all points of the state space. <code>F</code> is the predicate that is false in all points of the state space.

### Semantics

  **Deterministic machines**: the happening that will take place upon activation of the mechanism is fully determined by its initial state. 

  **Nondeterministic machines**: activation in a given initial state will give rise to one out of a class of possible happenings, the initial state only fixing the class as a whole.

  **Post-condition**: In computer programming, a post-condition is a condition or predicate that must always be true just after the execution of some section of code or after an operation in a formal specification. [via wiki](https://en.wikipedia.org/wiki/Postcondition)

  **Weakest pre-condition corresponding to that post-condition**: the condition that characterizes the set of all initial states such that activation will certainly result in a properly terminating happening leaving the system in a final state satisfying a given post-condition. If the system is denoted by <code>S</code> and the desired post-condition by <code>R</code>, then we denote the corresponding weakest pre-condition by <code>wp(S, R)</code>

  **A predicate transformer**: For a fixed mechanism <code>S</code> such a rule, which is fed with the post-condition <code>R</code> and delivers the corresponding weakest pre-condition <code>wp(S, R)</code>, is called "a predicate transformer".

  **Sufficient pre-condition**: <code>P</code> is a sufficient pre-condition, denoted as <code>P => wp(S, R)</code>, requires that wherever <code>P</code> is true, <code>wp(S, R)</code> is true as well. 





### 笔记来源
[A Discipline of Programming](https://book.douban.com/subject/1762127/)