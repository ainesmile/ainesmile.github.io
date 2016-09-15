---
layout: post
title:  "The Semantic Characterization of a Programming Language"
<!-- date:   2016-09-15 17:00:00 +0800 -->
categories: Programming Semantic
---

## Mechanism

  **Mechanism**: A **program** written in a well-defined programming language can be regarded as a mechanism.

  The Semantic Characterization of a Programming Language given by the set of rules that associate the corresponding predicate transformer with each program written in that language. From that point of view we can regard the program as "a code" for a predicate transformer.


## Five Statements

This book introduces five statements, they are "skip", "abort", "assignment", "if", and "do". These five statements are described by BNF. The author used BNF to give a formal definition of the syntax of his language.

**NOTE**: BNF is an acronym for "Backus-Naur Form". In computer science, BNF is one of the two main notation techniques for context-free grammars, often used to describe the syntax of languages used in computing. A BNF specification is a set of derivation rules, written as:
 

$$ <symbol> ::= \_\_expression\_\_ $$

where \\( \<symbol\> \\) is a nonterminal, \\( ::= \\) means "is defined as".[via wiki](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_Form)

### 1. Skip

The mechanism \\( S \\) such that for any post-condition \\( R \\) we have \\(wp(S, R) = R\\). This mechanism is called "skip", for any post-condition \\( R \\): 


\\[wp(skip, R) = R\\]

The statement **"skip"** means **"leaving things as they are"**. Using BNF, viz:

$$ <statement> ::= skip $$

To be read as "An element of the syntactic category called 'statement' " is defined as "skip".

### 2. Abort

The mechanism \\( S \\) such that \\(wp(S, R) = F\\) for all \\( R \\), therefore for any post-condition \\( R \\): 

\\[wp(abort, R) = F\\]

The statement **"abort"** means it **"cannot do a thing"**. If we take \\( R = T \\) which means no further requirement upon the final state, the mechanism will still fail to reach a final state. Using BNF, viz:

$$ <statement> ::= skip | abort$$

To be read as "An element of the syntactic category called 'statement' " is defined as "skip" or "abort".

### 3. Assignment

If the variable \\( x \\) is to be replaced by the expression \\( E \\), the usual way to write such a statement is:

$$ x := E $$

where ":=" should be read as "becomes". If in a predicate \\( R \\) all occurrences of the variable \\( x \\) are replaced by the expression \\( E \\), then we denote the result of this transformer by \\( {R\_\mathsf{E \rightarrow x}} \\)


The semantic definition of the assignment mechanism is

\\[ \mathbf{wp("x := E", R)} = \mathbf{R\_\mathsf{E \rightarrow x}} \\]

 Using BNF, we can extend our formal syntax to read:

 $$ <statement> ::= skip | abort | <assignment \ statement>$$

 $$ <assignment \ statement> ::= <variable> :=  <expression>$$

If we allow the expression \\( E \\) to be a partial function of the initial state, we must sharpen our definition of the semantics of the assignment operator and write

\\[ \mathbf{wp("x := E", R)} = \{D(E) \ cand \ \mathbf{R\_\mathsf{E \rightarrow x}}\} \\]

Here \\( D(E) \\) means "in the domain of E"; the boolean expression "\\( B1 \ cand \ B2 \\)"
is called **"conditional conjunction"**, has the same value of "\\( B1 \ and \ B2 \\)", but if \\( B1 \\) is false, the latter regardless of the question whether \\( B2 \\) is defined.

Using **concurrent assignment** statement, a number of different variables can be substituted simultaneously. Here is the example:

$$ x1, x2 := E1, E2 $$


### Composition of Mechanisms

To construct elaborate mechanism we follow the pattern that can be described recursively by

$$<mechanism> ::= <primitive \ mechanism> | <proper \  composition \ of \ <mechanism>'s>$$

Two conditions must be satisfied: we must have **"primitive mechanism"** to start with and, secondly, we must know how to **"compose properly"**.

The "skip", "abort", and "assignment" are primitive mechanisms. The composition must define how the properties of the whole follow from the properties of the parts.

Given two mechanisms \\( S1 \\) and \\( S2 \\), whose predicate transformers are known, if we can think of a rule for deriving a new predicate transformer, we regard this resulting predicate transformer as describing the properties of a composite object.

**Functional composition**, supplying the value of the one as argument to the other. The semantic definition of semicolon is defined:

$$ wp("S1;S2", R) = wp(S1, wp(S2,R))$$



With the aid of semicolon we can write programs as a concatenation of \\( n \ (n > 0) \\)
 statements: \\( "S1;S2;S3;...;Sn" \\).

### Guarded Command

To make the activation of a (sub)mechanism co-dependent on the current state of the system, we introduce **"guarded command"** in two steps:

$$ <guarding \ head > ::= <boolean \ expression> \rightarrow <statement> $$

$$ <guarded \ command > ::= <guarding \ head > \lbrace ;<statement> \rbrace$$

where the braces "{" and "}" should be read as "followed by zero or more instances of the enclosed".

An alternative syntax for a guarded command would have been:

$$ <statement \ list > ::= <statement> \lbrace ;<statement> \rbrace $$

$$ <guarded \ command > ::= < boolean \ expression >  \rightarrow <statement \ list > $$


The boolean expression preceding the arrow is called "a guard". The statement list following the arrow will only be executed only if the boolean expression is true.


**"IF"** and **"DO"** are two statements we can build from a guarded command set.

### 4. If

Suppose we are requested to construct a mechanism such that, if the initial state satisfies \\( Q \\), the finial state will satisfy \\( R \\). Suppose furthermore that we cannot find a single statement list that will do the job in all cases. In this situation, we need a set of statement lists, each of these statement lists satisfies a number of initial states, defined as:

$$ <guarded \ command \ set > ::= <guarded \ command> \lbrace ▯ <guarded \ command> \rbrace $$


where the symbol \\( ▯ \\) acts as a separator between otherwise unordered alternatives.

The **"if"** statement written as:

$$ <statement> ::= if <guarded \ command \ set> fi $$

One of the guarded commands whose guard is true is selected and its statement list is activated.

Let **"IF"** be the name of the statement

$$if \ B_{1} \rightarrow SL_{1} ▯ B_{2} \rightarrow SL_{2} ▯ ... ▯ B_{n} \rightarrow SL_{n} \ fi $$

where \\( B_{j} \\) are guards, then for any post-condition \\( R \\)

\begin{align}
wp(IF, R) = & (E j: 1 \leq j \leq n: B_{j}) \ and \\\\ & (A j: 1 \leq j \leq n: B_{j} \Rightarrow wp(SL_{j}, R))
\end{align}

This formula should be read as \\( wp(IF, R) \\) is true for every point in state space where there exists \\( j \\) in the range \\( ( 1 \leq j \leq n ) \\) such that \\( B_{j} \\) is true and for all \\( j \\) such that \\( B_{j} \\) is true, \\( wp(SL_{j}, R) \\) is true.

With "If" mechanism, the initial state satisfies \\( wp(IF, R) \\), the finial state will satisfy \\( R \\).

**NOTE**: A state in which all guards are false leads to **abortion**;

### 5. Do

Let "DO" be the name of the statement

$$do \ B_{1} \rightarrow SL_{1} ▯ B_{2} \rightarrow SL_{2} ▯ ... ▯ B_{n} \rightarrow SL_{n} \ od $$

The condition \\( H_{k}(R)\\) are given by

$$ H_{0}(R) = R \ and \ non (Ej: 1 \leq j \leq n: B_{j}) $$

and for \\( k > 0 \\):

$$ H_{k}(R) = wp(IF, H_{k-1}(R)) \ or \ H_{0}(R) $$

then

$$wp(DO, R) = (Ek: 0 \leq k \leq n: H_{k}(R))$$

Here  \\( H_{k}(R)\\) is the weakest precondition such that the do-od-construct will terminate after at most \\( k \\) selections of a guarded command, leaving the system in a final state satisfying the post-condition \\( R \\).

**NOTE1:**

\\( wp(DO, R) \\) characters all possible initial states that satisfy \\( H_{k}(R) \\). \\( H_{k}(R) \\) is true means \\( H_{0}(R)\\) is true or \\( wp(IF, H_{k-1}(R)) \\) is true.

\\( H_{0}(R)\\) is true means that all the guards are false, and terminate with the final state satisfying \\( R \\), semantically equivalent with "skip".

\\( wp(IF, H_{k-1}(R)) \\) is true means that for every point in state space where there exists \\( j \\) in the range \\( ( 1 \leq j \leq n ) \\) such that \\( B_{j} \\) is true and for all \\( j \\) such that \\( B_{j} \\) is true, \\( wp(IF, H_{k-1}(R)) \\) satisfied, that is \\( H_{k-1}(R) \\) satisfied.

In this way, \\( wp(IF, H_{k-1}(R)) \\) is true means that
in \\( k - 1 \\) steps, each step are required that there exists \\( j \\) such that \\( B_{j} \\) is true and in the end \\( H_{1}(R)\\) is true. \\( wp(IF, H_{0}(R)) \\) is true requires there exists \\( j \\) such that \\( B_{j} \\) is true and \\( j \\) such that \\( B_{j} \\) is true not exists in the same time, which is impossible, makes itself false and \\( H_{1}(R) = H_{0}(R)\\). That is, in \\( k - 1 \\) steps, the **do-od-construct** will terminate and leaving the system in a final state satisfying the post-condition \\( R \\).

**NOTE2:** If we allow the empty guarded command set, the **"do od"** is semantically equivalent with "skip".

### Functional Composition's Four Properties with Proof


**PROPERTY 1.**

For any mechanism \\( S1 \ and \ S2 \\), we have

$$ wp("S1;S2", F) = F$$

Prove:

$$ wp("S1;S2", F) = wp(S1, wp(S2,F)) = wp(S1, F) = F$$

**PROPERTY 2.**

For any mechanism \\( S1 \ and \ S2 \\), and any post-condition \\( Q \\), and \\( R \\) such that

$$ Q \Rightarrow R \ \ \ \ for \ all \ the \ states $$

we have 

$$ wp("S1;S2", Q) \Rightarrow wp("S1;S2", R) $$

Prove:

$$ wp("S1;S2", Q) = wp(S1, wp(S2,Q)) \Rightarrow wp(S1, wp(S2, R)) = wp("S1;S2", R) $$

**PROPERTY 3.**

For any mechanism \\( S1 \ and \ S2 \\), and any post-condition \\( Q \\), and \\( R \\) we have

$$ (wp("S1;S2", Q) \ and \ wp("S1;S2", R)) = wp("S1;S2",  Q \ and \ R) $$

Prove:

On the one hand,

\begin{align}
wp("S1;S2", Q) & = wp(S1, wp(S2,Q)) \\\\ & \Rightarrow
wp(S1, wp(S2, Q \ and \ R)) \\\\ & = wp("S1;S2",  Q \ and \ R)
\end{align}


similarly,

\begin{align}
wp("S1;S2", R) & = wp(S1, wp(S2,R)) \\\\ & \Rightarrow
wp(S1, wp(S2, Q \ and \ R)) \\\\ & = wp("S1;S2",  Q \ and \ R)
\end{align}

so we have 

$$ (wp("S1;S2", Q) \ and \ wp("S1;S2", R)) \Rightarrow wp("S1;S2",  Q \ and \ R)$$

that is 

$$ left \Rightarrow right $$

On the other hand,

\begin{align}
wp("S1;S2",  Q \ and \ R) & = wp(S1, wp(S2, Q \ and \ R)) \\\\ & = 
wp(S1, wp(S2, Q) \ and  \ wp(S2, R)) \\\\ & \Rightarrow wp(S1, wp(S2, Q)) \\\\ & = wp("S1;S2", Q)
\end{align}

Similarly,

\begin{align}
wp("S1;S2",  Q \ and \ R) & = wp(S1, wp(S2, Q \ and \ R)) \\\\ & = 
wp(S1, wp(S2, Q) \ and  \ wp(S2, R)) \\\\ & \Rightarrow wp(S1, wp(S2, R)) \\\\ & = wp("S1;S2", R)
\end{align}

so we have

$$ wp("S1;S2",  Q \ and \ R) \Rightarrow wp("S1;S2", Q) \ and \ wp("S1;S2", R)$$

that is 

$$ right \Rightarrow left $$

Proved.



**PROPERTY 4.**

For any mechanism \\( S1 \ and \ S2 \\), and any post-condition \\( Q \\), and \\( R \\) we have

$$ wp("S1;S2", Q) \ or \ wp("S1;S2", R) \Rightarrow  wp("S1;S2",  Q \ or \ R)\ \ \ \ for \ all \ the \ states$$

Prove:

Proprety 2 allows us to conclude

$$wp("S1;S2", Q) \Rightarrow wp("S1;S2",  Q \ or \ R)$$

Similarly,

$$wp("S1;S2", R) \Rightarrow wp("S1;S2",  Q \ or \ R)$$

so we have,

$$ wp("S1;S2", Q) \ or \ wp("S1;S2", R) \Rightarrow  wp("S1;S2",  Q \ or \ R)\ \ \ \ for \ all \ the \ states$$


### 笔记来源
[A Discipline of Programming](https://book.douban.com/subject/1762127/)