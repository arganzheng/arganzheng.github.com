---
layout: post
title: JUnit与Spring的整合——JUnit的TestCase如何自动注入Spring容器托管的对象
---


## 问题

在Java中，一般使用JUnit作为单元测试框架，测试的对象一般是Service和DAO，也可能是RemoteService和Controller。所有这些测试对象基本都是Spring托管的，不会直接new出来。而每个TestCase类却是由JUnit创建的。如何在每个TestCase实例中注入这些依赖呢？

## 预期效果

我们希望能够达到这样的效果：

    package me.arganzheng.study;
	
	import static org.junit.Assert.*;
	import org.junit.Test;
	import org.springframework.beans.factory.annotation.Autowired;
	
	/**
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

不过正如它注释所说的，**`Can be overridden to post-process the job instance`**，我们的做法也正是继承了`org.springframework.scheduling.quartz.SpringBeanJobFactory`，然后覆盖它的这个方法：
        
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

由于`OurSpringBeanJobFactory`是配置在Spring容器中，默认就具备拿到ApplicationContext的能力。当然就可以做ApplicationContext能够做的任何事情。

#### 题外话
    这里体现了框架设计一个很重要的原则：开闭原则——针对修改关闭，针对扩展开放。
    除非是bug，否者框架的源码不会直接拿来修改，但是对于功能性的个性化需求，框架应该允许用户进行扩展。
    这也是为什么所有的框架基本都是面向接口和多态实现的，并且支持应用通过配置项注册自定义实现类，
    比如Quartz的`org.quartz.scheduler.jobFactory.class`和`org.quartz.scheduler.instanceIdGenerator.class`配置项。


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
    
那么根据前面的讨论，我们只要extends`org.junit.runners.BlockJUnit4ClassRunner`类，覆盖它的`createTest`方法就可以了。如果我们的这个类能够方便的拿到ApplicationContext（这个其实很简单，比如使用`ClassPathXmlApplicationContext`），那么就可以很方便的实现依赖注入功能了。JUnit没有专门定义创建UT实例的接口，但是它提供了`@RunWith`的注解，可以让我们指定我们自定义的ClassRunner。于是，解决方案就出来了。

## Spring内建的解决方案

Spring3提供了`SpringJUnit4ClassRunner`基类让我们可以很方便的接入JUnit4。    

    public class org.springframework.test.context.junit4.SpringJUnit4ClassRunner extends org.junit.runners.BlockJUnit4ClassRunner {
        ...
    }
    
思路跟我们上面讨论的一样，不过它采用了更灵活的设计：

1. 引入[Spring TestContext Framework](http://static.springsource.org/spring/docs/3.0.0.M3/reference/html/ch10s03.html#testcontext-framework)，允许接入不同的UT框架（如JUnit3.8-，JUnit4.5+，TestNG，etc.）.
2. 相对于ApplicationContextAware接口，它允许指定要加载的配置文件位置，实现更细粒度的控制，同时缓存application context per Test Feature。这个是通过`@ContextConfiguration`注解暴露给用户的。（其实由于`SpringJUnit4ClassRunner`是由JUnit创建而不是Spring创建的，所以这里ApplicationContextAware should not work。但是笔者发现`AbstractJUnit38SpringContextTests`是实现`ApplicationContextAware`接口的，但是其ApplicationContext又是通过`org.springframework.test.context.support.DependencyInjectionTestExecutionListener`加载的。感觉实在没有必要实现`ApplicationContextAware`接口。）
3. 基于事件监听机制（the listener-based test context framework），并且允许用户自定义事件监听器，通过`@TestExecutionListeners`注解注册。默认是`org.springframework.test.context.support.DependencyInjectionTestExecutionListener`、`org.springframework.test.context.support.DirtiesContextTestExecutionListener`和`org.springframework.test.context.transaction.TransactionalTestExecutionListener`这三个事件监听器。

其中依赖注入就是在`org.springframework.test.context.support.DependencyInjectionTestExecutionListener`完成的：
     
    /**
	 * Performs dependency injection and bean initialization for the supplied
	 * {@link TestContext} as described in
	 * {@link #prepareTestInstance(TestContext) prepareTestInstance()}.
	 * <p>The {@link #REINJECT_DEPENDENCIES_ATTRIBUTE} will be subsequently removed
	 * from the test context, regardless of its value.
	 * @param testContext the test context for which dependency injection should
	 * be performed (never <code>null</code>)
	 * @throws Exception allows any exception to propagate
	 * @see #prepareTestInstance(TestContext)
	 * @see #beforeTestMethod(TestContext)
	 */
	protected void injectDependencies(final TestContext testContext) throws Exception {
		Object bean = testContext.getTestInstance();
		AutowireCapableBeanFactory beanFactory = testContext.getApplicationContext().getAutowireCapableBeanFactory();
		beanFactory.autowireBeanProperties(bean, AutowireCapableBeanFactory.AUTOWIRE_NO, false);
		beanFactory.initializeBean(bean, testContext.getTestClass().getName());
		testContext.removeAttribute(REINJECT_DEPENDENCIES_ATTRIBUTE);
	}

这里面ApplicationContext在Test类创建的时候就已经根据@ContextLocation标注的位置加载存放到TestContext中了：
   
    /**
     * TestContext encapsulates the context in which a test is executed, agnostic of
     * the actual testing framework in use.
     * 
     * @author Sam Brannen
     * @author Juergen Hoeller
     * @since 2.5
     */
    public class TestContext extends AttributeAccessorSupport {

    	TestContext(Class<?> testClass, ContextCache contextCache, String defaultContextLoaderClassName) {
	        ...
	    	
		    if (!StringUtils.hasText(defaultContextLoaderClassName)) {
			    defaultContextLoaderClassName = STANDARD_DEFAULT_CONTEXT_LOADER_CLASS_NAME;
		    }

		    ContextConfiguration contextConfiguration = testClass.getAnnotation(ContextConfiguration.class);
		    String[] locations = null;
		    ContextLoader contextLoader = null;

	    	...

			Class<? extends ContextLoader> contextLoaderClass = retrieveContextLoaderClass(testClass,
				defaultContextLoaderClassName);
			contextLoader = (ContextLoader) BeanUtils.instantiateClass(contextLoaderClass);
			locations = retrieveContextLocations(contextLoader, testClass);
		

		    this.testClass = testClass;
		    this.contextCache = contextCache;
		    this.contextLoader = contextLoader;
		    this.locations = locations;
	    }
	}


#### ** *说明* **:

Spring3使用了[Spring TestContext Framework](http://static.springsource.org/spring/docs/3.0.0.M3/reference/html/ch10s03.html#testcontext-framework)框架，支持多种接入方式：[10.3.5.5 TestContext support classes](http://static.springsource.org/spring/docs/3.0.0.M3/reference/html/ch10s03.html#testcontext-support-classes)。非常不错的官方文档，强烈推荐阅读。简单概括如下：

+ JUnit3.8：package `org.springframework.test.context.junit38`
    + `AbstractJUnit38SpringContextTests`
        + applicationContext
    + `AbstractTransactionalJUnit38SpringContextTests`
        + applicationContext
        + simpleJdbcTemplate
+ JUnit4.5：package `org.springframework.test.context.junit4`
    + `AbstractJUnit4SpringContextTests`
        + applicationContext
    + `AbstractTransactionalJUnit4SpringContextTests`
        + applicationContext
        + simpleJdbcTemplate
    + Custom JUnit 4.5 Runner：`SpringJUnit4ClassRunner`
        + @Runwith
        + @ContextConfiguration
        + @TestExecutionListeners
+ TestNG: package ` org.springframework.test.context.testng`
    + `AbstractTestNGSpringContextTests`
        + applicationContext
    + `AbstractTransactionalTestNGSpringContextTests`
        + applicationContext
        + simpleJdbcTemplate
       
补充：对于JUnit3，Spring2.x原来提供了三种接入方式：

+ AbstractDependencyInjectionSpringContextTests
+ AbstractTransactionalSpringContextTests
+ AbstractTransactionalDataSourceSpringContextTests

不过从Spring3.0开始，这些了类都被`org.springframework.test.context.junit38.AbstractJUnit38SpringContextTests`和`AbstractTransactionalJUnit38SpringContextTests`取代了:
> @deprecated as of Spring 3.0, in favor of using the listener-based test context framework（不过由于JUnit3.x不支持`beforeTestClass`和`afterTestClass`，所以这两个事件是无法监听的。）

> ({@link org.springframework.test.context.junit38.AbstractJUnit38SpringContextTests})

---

采用Spring3.x提供的SpringJUnit4ClassRunner接入方式，我们可以这样写我们的UT：


    package me.arganzheng.study;
	
	import static org.junit.Assert.*;
	import org.junit.Test;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.test.context.ContextConfiguration;
    import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
	
	/** 
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

    import org.junit.runner.RunWith;
    import org.springframework.test.context.ContextConfiguration;
    import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
    import org.springframework.transaction.annotation.Transactional;	
	/**  
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
	
    }
    
然后我们的FooServiceTest就可以简化为：

    package me.arganzheng.study;
	
	import static org.junit.Assert.*;
	import org.junit.Test;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.test.annotation.Rollback;
	
	/** 
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

上面只是简单解决了依赖注入问题，其实单元测试还有很多。如

1. 事务管理
2. Mock掉外界依赖
3. web层测试
4. 接口测试
3. 静态和私有方法测试
4. 测试数据准备和结果验证

等等。

--EOF--


