---
layout:post
title: JUnit与Spring的整合——JUnit中的TestCase如何自动注入spring容器托管的对象
---


## 问题

在Java中，一般使用JUnit作为单元测试框架，测试的对象一般是Service和DAO，也可能是RemoteService和Controller。所有这些测试对象基本都是Spring托管的，不会直接new出来。如何在每个UT中注入这些依赖呢？

## 预期效果

我们希望能够达到这样的效果：

    package me.arganzheng.study;
	
	import static org.junit.Assert.*;
	import org.junit.Test;
	import org.springframework.beans.factory.annotation.Autowired;
	
	/**
	 *  
	 * @author arganzheng
	 */
	public class FooServiceTest{
	
		@Autowired
		private FooService fooService;
	
		@Test
		public void testSaveFoo() {
			Foo foo = new Foo();
			// ...
			long id = fooService.saveFoo(foo);
	
			assertTrue(id > 0);
		}
	}
	
## 解决思路

其实在我前面的文章：[Quartz与Spring的整合-Quartz中的job如何自动注入spring容器托管的对象
](http://blog.arganzheng.me/posts/quartz-and-spring-integration-ioc-autowire.html)，已经详细的讨论过这个问题了。Quartz是一个框架，Junit同样是个框架，Spring对于接入外部框架，采用了非常一致的做法。对于依赖注入，不外乎就是这个步骤：

1. 首先，找到外部框架创建实例的地方（类或者接口），比如Quartz的jobFactory，默认为`org.quartz.simpl.SimpleJobFactory`,也可以配置为`org.quartz.simpl.PropertySettingJobFactory`。这两个类都是实现了`org.quartz.spi.JobFactory`接口。对于JUnit4.5+，则是`org.junit.runners.BlockJUnit4ClassRunner`类中的`createTest`方法。

        /**
         * Returns a new fixture for running a test. Default implementation executes
         * the test class's no-argument constructor (validation should have ensured
         * one exists).
         */
        protected Object createTest() throws Exception {
            return getTestClass().getOnlyConstructor().newInstance();
        }

2. 继承或者组合这些框架类，如果需要使用他们封装的一些方法的话。如果这些类是有实现接口的，那么也可以直接实现接口，与他们并行。然后对创建出来的对象进行依赖注入。

比如在Quartz中，Spring采用的是直接实现`org.quartz.spi.JobFactory`接口的方式：

        public class SpringBeanJobFactory extends AdaptableJobFactory implements SchedulerContextAware {
            ...
        }
        
        public class AdaptableJobFactory implements JobFactory {
            ...
	    }

但是Spring提供的`org.springframework.scheduling.quartz.SpringBeanJobFactory`并没有自动依赖注入，它其实也是简单的根据job类名直接创建类：
 
        /**
	     * Create an instance of the specified job class.
	     * <p>Can be overridden to post-process the job instance.
	     * @param bundle the TriggerFiredBundle from which the JobDetail
	     * and other info relating to the trigger firing can be obtained
	     * @return the job instance
	     * @throws Exception if job instantiation failed
	     */
	    protected Object createJobInstance(TriggerFiredBundle bundle) throws Exception {
		    return bundle.getJobDetail().getJobClass().newInstance();
	    }

不过正如它注释所说的，**`Can be overridden to post-process the job instance.`**，我们采也正是继承了`org.springframework.scheduling.quartz.SpringBeanJobFactory`，然后覆盖它的这个方法：
        
    public class OurSpringBeanJobFactory extends org.springframework.scheduling.quartz.SpringBeanJobFactory{
        @Autowire
        private AutowireCapableBeanFactory beanFactory;

        /**
         * 这里我们覆盖了super的createJobInstance方法，对其创建出来的类再进行autowire。
         */
        @Override
        protected Object createJobInstance(TriggerFiredBundle bundle) throws Exception {
            Object jobInstance = super.createJobInstance(bundle);
            beanFactory.autowireBean(jobInstance);
            return jobInstance;
        }
    ｝

由于`OurSpringBeanJobFactory`是配置在Spring容器中，默认就具备拿到ApplicationContext的能力。当然就可以做ApplicationContext提供的任何事情的。

#### 题外话
    这里体现了框架设计一个很重要的原则：针对修改关闭，针对拓展开放。
    除非是bug，否者框架的源码不会直接拿来修改，但是对于功能性的个性化需求，框架应该允许用户进行扩展。
    这也是为什么所有的框架基本都是面向接口和多态实现的，并且支持应用通过配置项注册自定义实现类，
    比如Quartz的`org.quartz.scheduler.jobFactory.class`和`org.quartz.scheduler.instanceIdGenerator.class`配置项。

----

## 解决方案

回到JUnit，其实也是如此。

Junit4.5+是通过`org.junit.runners.BlockJUnit4ClassRunner`中的`createTest`方法来创建单元测试类对象的。

    /**
     * Returns a new fixture for running a test. Default implementation executes
     * the test class's no-argument constructor (validation should have ensured
     * one exists).
     */
    protected Object createTest() throws Exception {
       return getTestClass().getOnlyConstructor().newInstance();
    }
    
那么根据前面的讨论，我们只要extends`org.junit.runners.BlockJUnit4ClassRunner`类，覆盖它的`createTest`方法就可以了。如果我们的这个类能够方便的拿到ApplicationContext（这个只要简单的将这个类配置在Spring容器中就可以了），那么就可以很方便的实现依赖注入功能了。JUnit没有专门定义创建UT实例的接口，但是它提供了`@RunWith`的注解，让我们可以指定我们自定义的ClassRunner。于是，解决方案就出来了。

## Spring内建的解决方案

Spring3提供了`SpringJUnit4ClassRunner`基类让我们可以很方便的接入JUnit4。    

    public class org.springframework.test.context.junit4.SpringJUnit4ClassRunner extends org.junit.runners.BlockJUnit4ClassRunner {
        ...
    }
    
思路跟我们上面讨论的一样，不过它采用了更灵活的设计：

1. 相对于ApplicationContextAware接口，它允许指定要加载的配置文件位置，实现更细粒度的控制。这个是通过`@ContextConfiguration`注解暴露给用户的。
2. 基于事件监听机制（the listener-based test context framework），并且允许用户自定义事件监听器。默认是`org.springframework.test.context.support.DependencyInjectionTestExecutionListener`、`org.springframework.test.context.support.DirtiesContextTestExecutionListener`和`org.springframework.test.context.transaction.TransactionalTestExecutionListener`这三个事件监听器。

> TIPS:
>   对于JUnit3，Spring提供了另外几种方式接入。不过不建议使用。
>   
    * AbstractDependencyInjectionSpringContextTests
    * AbstractTransactionalSpringContextTests
    * AbstractTransactionalDataSourceSpringContextTests

采用这种方式，我们可以这样写我们的UT：


    package me.arganzheng.study;
	
	import static org.junit.Assert.*;
	import org.junit.Test;
	import org.springframework.beans.factory.annotation.Autowired;
	
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
	public class FooServiceTest{
	
		@Autowired
		private FooService fooService;
	
		@Test
		public void testSaveFoo() {
			Foo foo = new Foo();
			// ...
			long id = fooService.saveFoo(foo);
	
			assertTrue(id > 0);
		}
	}

当然，每个UT类都要配置这么多anotation配置是很不方便的，搞成一个基类会好很多：

    ackage me.arganzheng.study;
	
	import static org.junit.Assert.*;
	import org.junit.Test;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.test.annotation.Rollback;
	
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
    @Transactional
	public class BaseSpringTestCase{
	
		@Autowired
	    private ConfigurableBeanFactory beanFactory;

	    @Before
	    public void setUp(){
		    beanFactory.registerScope("request", new RequestScope());
		    RequestContextHolder.setRequestAttributes(new ServletRequestAttributes(new MockHttpServletRequest()));
	    }

	    @After
	    public void tearDown(){
		    ((AbstractRequestAttributes)RequestContextHolder.currentRequestAttributes()).requestCompleted();
		    RequestContextHolder.resetRequestAttributes();
	    }
    }

注：关于RequestContextHolder的原因，是为了解决web层测试的一个问题。@Transactional是解决事务管理问题。后面有时间会讨论到。

然后我们的FooServiceTest就可以简化为：

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

## 单元测试的其他问题

上面只是简单解决了依赖注入和事务管理问题，其实单元测试还有很多。如

1. 事务管理
2. Mock掉外界依赖
3. 测试数据准备和结果验证

等等。

--EOF--


