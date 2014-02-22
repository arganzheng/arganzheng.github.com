---
title: 巧用TheadLocal
layout: post
---


应用场景
--------

1. 需要增加参数，但是不想或者不能修改接口。
2. 放置一些线程不安全，但是反复构建会比较耗内存或者CPU的对象，同步访问又会导致锁竞争。比如SimpleDateFormat。


实现原理
--------

本质上就是一个以threadId为key的Map。


实战例子——Spring Transaction Manager
------------------------------------

事务管理可能会跨越好几个Service方法，如何在这些Service调用之间保存签名service的事务定义和数据库链接信息呢(需要这些信息决定事务的行为，比如determines if there is an existing transaction; 
)，特别是事务id？Spring就是通过ThreadLocal来实现的（@see TransactionSynchronizationManager`）  。具体参见这篇文章：写的非常详细：[Spring @Transactional explained](http://doanduyhai.wordpress.com/2011/11/20/spring-transactional-explained/)


InheritableThreadLocal
----------------------

    package java.lang;
    import java.lang.ref.*;

    /**
     * This class extends <tt>ThreadLocal</tt> to provide inheritance of values
     * from parent thread to child thread: when a child thread is created, the
     * child receives initial values for all inheritable thread-local variables
     * for which the parent has values.  Normally the child's values will be
     * identical to the parent's; however, the child's value can be made an
     * arbitrary function of the parent's by overriding the <tt>childValue</tt>
     * method in this class.
     * 
     * <p>Inheritable thread-local variables are used in preference to
     * ordinary thread-local variables when the per-thread-attribute being
     * maintained in the variable (e.g., User ID, Transaction ID) must be
     * automatically transmitted to any child threads that are created.
     *
     * @author  Josh Bloch and Doug Lea
     * @version %I%, %G%
     * @see     ThreadLocal
     * @since   1.2
     */

    public class InheritableThreadLocal<T> extends ThreadLocal<T> {
        /**
         * Computes the child's initial value for this inheritable thread-local
         * variable as a function of the parent's value at the time the child
         * thread is created.  This method is called from within the parent
         * thread before the child is started.
         * <p>
         * This method merely returns its input argument, and should be overridden
         * if a different behavior is desired.
         *
         * @param parentValue the parent thread's value
         * @return the child thread's initial value
         */
        protected T childValue(T parentValue) {
            return parentValue;
        }

        /**
         * Get the map associated with a ThreadLocal. 
         *
         * @param t the current thread
         */
        ThreadLocalMap getMap(Thread t) {
           return t.inheritableThreadLocals;
        }

        /**
         * Create the map associated with a ThreadLocal. 
         *
         * @param t the current thread
         * @param firstValue value for the initial entry of the table.
         * @param map the map to store.
         */
        void createMap(Thread t, T firstValue) {
            t.inheritableThreadLocals = new ThreadLocalMap(this, firstValue);
        }
    }



参考文档
--------

1. [Understanding the concept behind ThreadLocal](http://javarecipes.com/2012/07/11/understanding-the-concept-behind-threadlocal/) 
2. [Java Thread Local – How to use and code sample](http://veerasundar.com/blog/2010/11/java-thread-local-how-to-use-and-code-sample/)
3. [Spring @Transactional explained](http://doanduyhai.wordpress.com/2011/11/20/spring-transactional-explained/)
4. [使用Java构造高可扩展应用——如何实现一个高效且多线程安全的队列](http://www.ibm.com/developerworks/cn/java/j-lo-scalbility/index.html)