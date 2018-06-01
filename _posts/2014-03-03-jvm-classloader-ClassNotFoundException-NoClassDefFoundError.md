---
title: JVM类加载器与ClassNotFoundException和NoClassDefFoundError
layout: post
category: [技术]
tags: [jvm, classloader]
catelog: true
---

**说明** 本文内容基本来着参考文章。

在我们日常的项目开发中，会经常碰到`ClassNotFoundException`和`NoClassDefFoundError`这两种异常，对于经验足够的工程师而言，可能很轻松的就可以解决，但是却不一定明白为何要去这么做，本博客将从java虚拟机类加载的角度让大家彻底理解`ClassNotFoundException`和`NoClassDefFoundError`这两种异常及一些重用的解决方案。

程序员定义的java类要被java虚拟机运行首先要做的就是类加载过程，而类加载过程的第一步就是“加载”过程（注意加载只是类加载的其中一个阶段而已，不要混淆这两个不同的概念）。在”加载“阶段，虚拟机需要完成以下三件事：

1.通过一个类的全限定名来获取此类的二进制字节流。
2.将字节流代表的静态存储结构转换为方法区的运行时数据结构
3.在内存中创建一个代表此类的`java.lang.Class`对象，作为方法区此类的各种数据的访问入口。

其中完成第一步动作的代码模块就是java类加载器，顾名思义：类加载器（class loader）用来加载 Java 类到 Java 虚拟机中。

类加载器会去获取类的信息，但是类加载器并没有规定类的格式。具体的格式通常有一下几种：

* 文件格式： class文件、jar文件、war文件等
* 动态生成：程序在执行过程中，动态生成类的字节码，主要表现为动态代理
* 有其他文件生成：比如说由JSP文件生成对应的Class文件
* 网络获取，Applet

加载阶段完成后，虚拟机外部的二进制字节流就按照虚拟机所需的格式存储在方法区之中。然后在内存中实例化一个`java.lang.Class`类的对象，这个对象将作为程序访问方法区中的这些类型数据的外部接口。

这个阶段，相比于C语言转载的过程，都是把可执行文件装入装入内存的过程。

下面对类加载器进行详细讲解。

### 一、`java.lang.ClassLoader`类

基本上java中的类加载器都是继承自`java.lang.ClassLoader`类的，所以首先我们来介绍一下`java.lang.ClassLoader`类，然后再介绍java中的几种类加载器。

`java.lang.ClassLoader`类的作用就是根据一个指定的类的名称，找到或者生成其对应的字节代码，然后从这些字节代码中定义出一个 Java 类，即 `java.lang.Class`类的一个实例。除此之外，ClassLoader还可以用于在classpath中加载 java 应用所需的资源，如图像文件和配置文件等。我们在此只关心和类加载相关的功能，首先我们来看一下`java.lang.ClassLoader`类中和类加载相关的重要方法：

* ClassLoader getParent() : 返回该类加载器的父类加载器
* Class<?> loadClass(String name) throws ClassNotFoundException : 加载名称为name的类
* Class<?> findClass(String name) throws ClassNotFoundException : 查找名称为name的类
* Class<?> findLoadedClass(String name) : 查找名称为name的已经被加载过的类
* Class<?> defineClass(String name, byte[] b, int off, int len) throws ClassFormatError : 将class二进制内容(byte[]b)转换成Class对象
* void resolveClass(Class<?> c) : 链接指定的Class对象

注意`defineClass`方法存在多个重载版本，其中一个会抛出`NoClassDefFoundError`异常，而loadClass()抛出的是 `java.lang.ClassNotFoundException` 异常。

### 二、类加载器的树状组织结构

Java 中的类加载器大致可以分成两类，一类是系统提供的，另外一类则是用户自定义的。系统提供的类加载器主要包括以下三个：

1. 启动加载类(`Bootstrap ClassLoader`)：C++实现，是JVM的一部分。负责加载`$JAVAHOME/lib`目录下的类。或者被`-Xbootclasspath`参数所指定的路径中的，并且是JVM所识别的（仅安装文件名称识别，如rt.jar，名称不符合的类库即使放在lib目录也不会被加载）类库加载到JVM内存中。启动加载类无法被Java程序直接引用。
2. 扩展类加载器(`Extension ClassLoader`)：这个ClassLoader由`sun.misc.Launcher$ExtClassLoader`实现，它负责加载`$JAVAHOME/lib/ext`目录中的，或者被`java.ext.dirs`系统变量所指定的路径中的所有类库，开发者可以直接使用扩展类加载器。
3. 应用程序类加载器(`Application ClassLoader`)：这个类加载器由`sun.misc.Launcher$AppClassLoader`来实现。由于这个类加载器是`ClassLoader.getSystemClassLoader()`方法的返回值，所以一般也称之为系统类加载器(`SystemClassLoader`)。它负责加载用户类路径(通过`$classpath`环境变量指定)上所指定的类库，开发者可以直接使用这个类加载器，如果应用程序中没有自定义过自己的类加载器，一般情况下这个就是程序中默认的类加载器。

除了引导类加载器之外，所有的类加载器都有一个父类加载器。通过 `getParent()` 方法可以得到。对于系统提供的类加载器来说，系统类加载器的父类加载器是扩展类加载器，而扩展类加载器的父类加载器是引导类加载器；对于用户自定义的类加载器来说，其父类加载器是加载此类加载器 Java 类的类加载器。因为类加载器 Java 类如同其它的 Java 类一样，也是要由类加载器来加载的。一般来说，用户自定义的类加载器的父类加载器是系统类加载器。类加载器通过这种方式组织起来，形成树状结构。树的根节点就是引导类加载器。用图表示如下：

![类加载器双亲委派模型](/img/in-post/classloader.png)

在上述图示中，给人的直观感觉是系统类加载器的父类加载器是标准扩展类加载器，标准扩展类加载器的父类加载器是启动类加载器，真的是这样吗？我们通过代码来测试一下：

```java
public class LoaderTest {  

    public static void main(String[] args) {  
        try {  
            System.out.println(ClassLoader.getSystemClassLoader());  
            System.out.println(ClassLoader.getSystemClassLoader().getParent());  
            System.out.println(ClassLoader.getSystemClassLoader().getParent().getParent());  
        } catch (Exception e) {  
            e.printStackTrace();  
        }  
    }  
}  
```

程序的运行结果如下：

```
sun.misc.Launcher$AppClassLoader@6d06d69c  
sun.misc.Launcher$ExtClassLoader@70dea4e  
null 
```

我们知道通过`java.lang.ClassLoader.getSystemClassLoader()`可以直接获取到系统类加载器，那么从运行结果来看，系统类加载器的父加载器是标准扩展类加载器，但是我们试图获取标准扩展类加载器的父类加载器时确得到了null，不是预期的启动类加载器，这是为何呢？这就需要我们从`java.lang.ClassLoader`中的`loadClass（String name`方法的源代码就找答案了：

```java
public Class<?> loadClass(String name) throws ClassNotFoundException {  
    return loadClass(name, false);  
}  
  
protected synchronized Class<?> loadClass(String name, boolean resolve)  
        throws ClassNotFoundException {  
  
    // 首先判断该类型是否已经被加载  
    Class c = findLoadedClass(name);  
    if (c == null) {  
        //如果没被加载，就委托给父类加载或者委派给启动类加载器加载  
        try {  
            if (parent != null) {  
                //如果存在父类加载器，就委派给父类加载器加载  
                c = parent.loadClass(name, false);  
            } else {  
                //如果不存在父类加载器，就通过启动类加载器加载该类，  
                //通过调用本地方法native findBootstrapClass0(String name)  
                c = findBootstrapClass0(name);  
            }  
        } catch (ClassNotFoundException e) {  
            // 如果父类加载器和启动类加载器都不能完成加载任务，才调用自身的加载功能  
            c = findClass(name);  
        }  
    }  
    if (resolve) {  
        resolveClass(c);  
    }  
    return c;  
}  
```

从上述代码可以很清楚的看到如果该加载器的父类为null则会调用启动类加载器类加载类，除非启动类加载器不能完成加载任务，才会调用自定义的加载器，因此可知将扩展类的父加载器设置为null与将其父加载器设置为启动类加载器效果是一样的。

### 三、类加载器的双亲委派规则（代理模式）

双亲委派规则指的是如果一个类加载器收到了类加载的请求，它首先不会自己尝试去加载这个类，而是把这个请求委派给父类加载器去完成，每一个层次的类加载器都是如此，因此所有的加载请求最终都会传递到顶层的启动类加载器去加载，如果父加载器不能完成这个加载请求（在它的搜索范围内找不到所需的类），子加载器才会自己尝试去加载。

那么java中为何会采取双亲委派原则呢？（其实这种原则在各种程序设计中很常见，如安卓中的事件分派机制），要知道这个原因，首先我们需要知道在java中虚拟机是如何判断两个类为同一个类的，Java 虚拟机不仅要看类的全名是否相同，还要看加载此类的类加载器是否一样。当且仅当两者都相同的情况，才认为两个类是相同的。即便是同样的字节代码，被不同的类加载器加载之后所得到的类，也是不同的。比如一个 Java 类 com.example.Sample，编译之后生成了字节代码文件 Sample.class。两个不同的类加载器 ClassLoaderX 和 ClassLoaderY 分别读取了这个 Sample.class文件，然后定义出两个 java.lang.Class类的实例来表示这个类。这两个实例是不相同的。对于 Java 虚拟机来说，它们是不同的类。如果试图对这两个类的对象进行相互赋值，会抛出运行时异常 ClassCastException。

了解了这一点之后，就可以理解采用双亲委派规则的原因了。采用双亲委派规则是为了保证 Java 核心库的类型安全。所有 Java 应用都至少需要引用 java.lang.Object 类，也就是说在运行的时候，java.lang.Object 这个类需要被加载到 Java 虚拟机中。如果这个加载过程由 Java 应用自己的类加载器来完成的话，很可能就存在多个版本的 java.lang.Object 类，而且这些类之间是不兼容的。通过采用双亲委派规则，对于 Java 核心库的类的加载工作由引导类加载器来统一完成，保证了 Java 应用所使用的都是同一个版本的 Java 核心库的类，是互相兼容的。

### 四、类加载器与ClassNotFoundException和NoClassDefFoundError

通过前面的讲解我们已经知道类加载器采用的是双亲委派原则，类加载器会首先代理给其父类加载器来尝试加载某个类。这就意味着真正完成类的加载工作的类加载器和启动这个加载过程的类加载器，可能不是同一个。真正完成类的加载工作是通过调用 defineClass 来实现的；而启动类的加载过程是通过调用 loadClass 来实现的。前者称为一个类的定义加载器（defining loader），后者称为初始加载器（initiating loader）。在 Java 虚拟机判断两个类是否相同的时候，使用的是类的定义加载器。也就是说，哪个类加载器启动类的加载过程并不重要，重要的是最终定义这个类的加载器。两种类加载器的关联之处在于：一个类的定义加载器是它引用的其它类的初始加载器。如类 com.example.Outer 引用了类 com.example.Inner，则由类 com.example.Outer 的定义加载器负责启动类 com.example.Inner 的加载过程。 `java.lang.ClassNotFoundException`是被`loadClass()`抛出的， `java.lang.NoClassDefFoundError`是被`defineClass()`抛出。

类加载器在成功加载某个类之后，会把得到的 java.lang.Class 类的实例缓存起来。下次再请求加载该类的时候，类加载器会直接使用缓存的类的实例，而不会尝试再次加载。也就是说，对于一个类加载器实例来说，相同全名的类只加载一次，即 loadClass 方法不会被重复调用。

通过前面的讲解我们知道loadClass()是来启动类的加载过程的，其源代码在前面我们已经分析过，即当父加载器不为null同时不能完成加载请求的情况下会抛出ClassNotFoundException异常，而导致父加载器无法完成加载的一个原因很简单就是找不到这个类，通常这种情况是传入的类的字符串名称书写错误，如调用class.forName(String name)或者loadClass(String name)时传入了一个错误的类的名称导致类加载器无法找到这个类。

> **java.lang.ClassNotFoundException:**
>
> Thrown when an application tries to load in a class through its string name using:
> 1. The `forName` method in class Class.
> 2. The `findSystemClass` method in class ClassLoader .
> 3. The `loadClass` method in class ClassLoader.
> 
> but no definition for the class with the specified name could be found.


从规范说明看，`java.lang.ClassNotFoundException`异常抛出的根本原因是类文件找不到，缺少了`.class`文件，比如少引了某个 jar，解决方法通常需要检查一下 classpath 下能不能找到包含缺失 `.class` 文件的 jar。

defineClass是用来完成类的加载工作的，此时已经表明类加载的启动已经完成了，即当前执行的类已经编译，但运行时找不到它的定义时（class definition existed when the currently executing class was compiled, but the definition can no longer be found.）就会抛出NoClassDefFoundError异常，这种情况通常出现在创建一个对象实例的时候（creating a new instance），如在类X中定义了一个类Y，如在类X中定义如下语句`ClassY y=new ClassY`;程序运行成功之后（此时X与Y的字节码文件已经存在），如果将类Y的字节码文件删除了重新运行上述代码，则会在运行时候抛出NoClassDefFoundError异常。当然这只是为了说明抛出这种异常的原因，一般不会出现删除该类字节码情况，实际上是其它原因导致类似删除的效果导致的，如JAR重复引入，版本不一致导至。因为jar中都是一些已经编译好的Class文件，如果存在多个版本那么在加载的时候就不知道应该调用哪一个版本（相当于删除字节码的效果），此种情况一般出现在引入第三方SDK的时候。

再如类X引用类Y时，类Y初始化失败时也会导致以上的错误出现。其实上面说的重复引入第三方SDK就属于这种情况的一种，一般在我们自己的java文件中import第三方的java类，如果存在多个版本那么在加载的时候就不知道应该调用哪一个版本（相当与引入的类Y初始化失败），当然除了第三方SDK外，自己定义的java文件可能也会出现这种情况。

> **java.lang.NoClassDefFoundError:**
> 
> Thrown if the Java Virtual Machine or a ClassLoader instance tries to load in the definition of a class (as part of a normal method call or as part of creating a new instance using the new expression) and no definition of the class could be found.
> 
> The searched-for class definition existed when the currently executing class was compiled, but the definition can no longer be found.

出现这个异常，并不是说这个`.class`类文件不存在，而是类加载器试图加载类的定义时却找不到这个类的定义，实际上`.class`文件是存在的。

> You are getting a `java.lang.NoClassDefFoundError` which does NOT mean that your class is missing (in that case you'd get a `java.lang.ClassNotFoundException`). The ClassLoader ran into an error while reading the class definition when trying to read the class.

而碰到这个异常的解决办法，一般需要检查这个类定义中的初始化部分（如类属性定义、static 块等）的代码是否有抛异常的可能。这个一般在日志中也会体现出来：`java.lang.NoClassDefFoundError: Could not initialize class xxx`。

例如下面这个例子：

```
public class Foo { 
    private static Logger logger = LoggerFactory.getLogger(Foo.class); 
    // ... 
} 
```

在另一个类里创建 Foo 实例:

```
public class Bar { 
    public void someMethod() { 
        Foo foo = new Foo(); 
        // ... 
    } 
} 
```

在执行 new Foo() 时抛异常 `java.lang.NoClassDefFoundError: Could not initialize class Foo`。经过一番排查,这个异常最后的原因出在了 `LoggerFactory.getLogger(Foo.class)` 调用抛错，原因是由于 `slf4j-api.jar` 和 slf4j 的某个 binding jar 版本不兼容所致。

另外,从两个异常的定义看，java.lang.NoClassDefFoundError 是一种 unchecked exception(也称 runtime exception),而 java.lang.ClassNotFoundException 是一种 checked exception。


参考文章
-------

1. [【java虚拟机系列】JVM类加载器与ClassNotFoundException和NoClassDefFoundError](https://blog.csdn.net/htq__/article/details/51753883)
2. [https://www.aliyun.com/jiaocheng/529511.html](https://www.aliyun.com/jiaocheng/529511.html)
3. [ClassLoader 学习笔记](https://www.cnblogs.com/kakaxisir/p/5723260.html)
