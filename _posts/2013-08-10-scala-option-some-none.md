---
layout: post
title: Scala的Option、Some和None
---


在java中，我们经常会遇到NPE异常，看到一个方法，我们无法确定它会不会返回一个null对象。方法本身也无法保证接收到的参数一定是不是null，所以有了防御式编程的说法。返回值与参数的取值说明，我们只能从代码注释或者文档中得知，更差情况，从代码实现得知，无法从函数签名直观得到。Scala引入Option关键字解决了这个问题。

必然下面这个函数定义：

    def computeFoo: Option[Foo] = { … }

表示computeFoo方法可能返回Foo对象，也可以返回null。

配合scala的pattern match，使用起来非常的优雅：

    computeFoo match {
        case Some(foo) => ... // got a Foo, do something with it
        case None => ... // did not get a Foo, deal with it
    }

如果看到一个方法声明为
    
    def computeFoo: Foo = { … }
    
那么我们可以假定这个方法不会返回null。所以我们可以放心的使用。

**NOTE** 这只是一个约定(规范)，不是一个保证。即使你没有声明Option[Foo]，但是仍然返回null，Scala不会报错。可以类比Ruby中的方法签名中的问号（表示返回boolean）和感叹号（表示有状态修改）。


## 参考文章

1. [The beauty of Scala's Option/Some/None](http://blog.orbeon.com/2011/04/scalas-optionsomenone.html)
2. [Scala Options](http://www.tutorialspoint.com/scala/scala_options.htm)