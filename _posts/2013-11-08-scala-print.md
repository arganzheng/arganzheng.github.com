---
layout: post
title: scala的字符串格式化输出
---


如下：

    D:\code\api_sdk_proj\meta-idl>scala
    Welcome to Scala version 2.10.2 (Java HotSpot(TM) Client VM, Java 1.6.0_25).
    Type in expressions to have them evaluated.
    Type :help for more information.
    
    scala> case class A(s:String)
    defined class A
    
    scala> val a=A("argan")
    a: A = A(argan)
    
    scala> a.s
    res0: String = argan
    
    scala> println(f"a.s=$a.s")
    a.s=A(argan).s
    
    scala> a
    res2: A = A(argan)
    
    scala> println(f"a.s=${a.s}")
    a.s=argan
    
    scala>