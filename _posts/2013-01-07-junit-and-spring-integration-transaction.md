---
layout: post
title: JUnit与Spring的整合——JUnit中的TestCase如何拥有spring的事务管理机制
---


## 问题

很多时候我们不希望单元测试的结果玷污了数据库，玷污了我们也不希望人肉去check，我们希望它无痕迹的悄悄执行，告诉我最终结果即可。那么怎样让你的UT也具有事务功能呢？

## 预期效果

我们希望能够达到这样的效果：

    package me.arganzheng.study;
	
	import static org.junit.Assert.*;
	import org.junit.Test;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.test.annotation.Rollback;
	
	/**
	 *  
	 * @author arganzheng
	 */
	public class FooServiceTest{
	
		@Autowired
		private FooService fooService;
	
		@Test
		@Rollback(true)
		public void testSaveFoo() {
			Foo foo = new Foo();
			// ...
			long id = fooService.saveFoo(foo);
	
			assertTrue(id > 0);
		}
	}
	
虽然我们插入了一条数据，但是在UT后，已经雁过无痕了。

	
## 解决方案

在前面一篇文章[JUnit与Spring的整合——JUnit的TestCase如何自动注入Spring容器托管的对象](http://blog.arganzheng.me/posts/junit-and-spring-integration-ioc-autowire.html)中我们详细的介绍了Spring与Junit的整合，提到了Spring3提供了`SpringJUnit4ClassRunner`基类让我们可以很方便的接入JUnit4。    

    public class org.springframework.test.context.junit4.SpringJUnit4ClassRunner extends org.junit.runners.BlockJUnit4ClassRunner {
        ...
    }
    
它是基于事件监听机制（the listener-based test context framework），并且允许用户自定义事件监听器。默认是`org.springframework.test.context.support.DependencyInjectionTestExecutionListener`、`org.springframework.test.context.support.DirtiesContextTestExecutionListener`和`org.springframework.test.context.transaction.TransactionalTestExecutionListener`这三个事件监听器。

其中`org.springframework.test.context.support.DependencyInjectionTestExecutionListener`是实现自动注入Spring托管的bean的。而`org.springframework.test.context.transaction.TransactionalTestExecutionListener`则是实现事务管理的事件处理器：

>
org.springframework.test.context.transaction.TransactionalTestExecutionListener
>
>
TestExecutionListener which provides support for executing tests within transactions by using @Transactional and @NotTransactional annotations. 
>
Changes to the database during a test run with @Transactional will be run within a transaction that will, by default, be automatically rolled back after completion of the test; whereas, changes to the database during a test run with @NotTransactional will not be run within a transaction. Similarly, test methods that are not annotated with either @Transactional (at the class or method level) or @NotTransactional will not be run within a transaction. 
>
Transactional commit and rollback behavior can be configured via the class-level @TransactionConfiguration and method-level @Rollback annotations. @TransactionConfiguration also provides configuration of the bean name of the PlatformTransactionManager that is to be used to drive transactions. 
>
When executing transactional tests, it is sometimes useful to be able execute certain set up or tear down code outside of a transaction. TransactionalTestExecutionListener provides such support for methods annotated with @BeforeTransaction and @AfterTransaction. 

其实我发现Spring的javadoc非常的简单明了又全面，根本不需要到处查找其他的东东，直接看javadoc和源码就可以了。这里简单概括一下：

+ `TransactionalTestExecutionListener`
    + @Transactional (class-level & method-level)
    + @NotTransactional (method-level)        
    + @TransactionConfiguration (class-level)        
    + @Rollback (method-level)
    + @BeforeTransaction (method-level)
    + @AfterTransaction (method-level)

注：其中 class-level即`Target(ElementType.TYPE)`，method-level即`Target(ElementType.METHOD)`。

这样我们可以如下配置：

    ackage me.arganzheng.study;
	
	import org.junit.runner.RunWith;
    import org.springframework.test.context.ContextConfiguration;
    import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
    import org.springframework.transaction.annotation.Transactional;  
	
	/**
	 *  
	 * @author arganzheng
	 */
	@RunWith(SpringJUnit4ClassRunner.class)
    @ContextConfiguration({
	    "classpath:conf-spring/spring-dao.xml",
	    "classpath:conf-spring/spring-service.xml",
	    "classpath:conf-spring/spring-controller.xml"
    })
    @TransactionConfiguration(transactionManager="transactionManager") //可选，默认就是这个
    @Transactionnal
	public class BaseSpringTestCase{
	
    }

然后我们的FooServiceTest这样写就可以拥有事务处理能力了：

    package me.arganzheng.study;
	
	import static org.junit.Assert.*;
	import org.junit.Test;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.test.annotation.Rollback;
	
	/**
	 *  
	 * @author arganzheng
	 */
	public class FooServiceTest extends BaseSpringTestCase{
	
		@Autowired
		private FooService fooService;
	
		@Test
		// @Rollback(true) 默认就是true
		public void testSaveFoo() {
			Foo foo = new Foo();
			// ...
			long id = fooService.saveFoo(foo);
	
			assertTrue(id > 0);
		}
	}

当然前提是你在Spring的配置文件中正确的配置了transactionManager。关于如何在Spring中配置事务，请参考笔者另一篇文章 [Spring事务配置](http://blog.arganzheng.me/posts/spring-transaction.html) .

