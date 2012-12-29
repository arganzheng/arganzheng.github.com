---
layout: post
title: Quartz与Spring的整合-Quartz中的job如何自动注入spring容器托管的对象
---

## 问题

Quartz如果使用数据库配置方式，则业务job并不在xml配置文件中配置，而是在数据库中指定classname，Quartz会根据classname实例化该job，但是没有办法定义与注入该job依赖的其他bean。

## 预期效果

我们希望达到这样的效果：

    /**
     * 
     * 取消超时未支付订单的任务。 
     *
     * @author arganzheng
     *
     */
    public class CancelUnpaidOrderTask implements Job {
	    private final static Logger log = Logger.getLogger(CancelUnpaidOrderTask.class);

	    @Autowired
	    private AppOrderService orderService;


	    @Override
	    public void execute(JobExecutionContext ctx) throws JobExecutionException {
		    ...
    }

关键在这一行：

    @Autowired
	private AppOrderService orderService;

orderService是配置在spring容器中的，而CancelUnpaidOrderTask则是配置在Quarzt数据库中，由`org.springframework.scheduling.quartz.SpringBeanJobFactory` 运行时调用`
	protected Object createJobInstance(TriggerFiredBundle bundle) throws Exception;` 方法创建的。

## 解决方案

Spring提供了一种机制让你可以获取ApplicationContext。就是`ApplicationContextAware`接口。对于一个实现了`ApplicationContextAware`接口的类，Spring会实例化它的同时，调用它的`public void setApplicationContext(ApplicationContext applicationContext) throws BeansException;`接口，将该bean所属上下文传递给它。

一般利用这个来做ServicesLocator:

    public class FooServicesLocator implemnts ApplicationContextAware{
        private static ApplicationContext applicationContext;

        @Override
        public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
            this.applicationContext = applicationContext;
        }

        public static FooService getFooService() {
            return applicationContext.getBean(FooService.class);
        }
    }
    
    
然后在需要引用FooService的地方，这样子获取FooService：`FooServicesLocator.getFoobarServic();` 得到Spring托管的FooService。

不过这样是依赖查询，不是注入，要实现DI，可以使用`AutowireCapableBeanFactory`进行autowire。     

    applicationContext.getAutowireCapableBeanFactory().autowireBean(existingBean);

于是对于上面的那个问题，就有了如下的解决方案：

    package me.arganzheng.study.quarzt.task.SpringBeanJobFactory;
    
    import org.quartz.spi.TriggerFiredBundle;
    import org.springframework.beans.BeansException;
    import org.springframework.context.ApplicationContext;
    import org.springframework.context.ApplicationContextAware;

    /**
     * 自定义SpringBeanJobFactory，用于对Job注入ApplicationContext等。
     * 
     * @author arganzheng
     */
    public class SpringBeanJobFactory extends org.springframework.scheduling.quartz.SpringBeanJobFactory implements ApplicationContextAware {
    	private ApplicationContext applicationContext;

	    @Override
	    public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
		    this.applicationContext = applicationContext;
	    }
	   
	    /**
	     * 这里我们覆盖了super的createJobInstance方法，对其创建出来的类再进行autowire。
	     */
	    @Override
	    protected Object createJobInstance(TriggerFiredBundle bundle) throws Exception {
		    Object jobInstance = super.createJobInstance(bundle);
		    applicationContext.getAutowireCapableBeanFactory().autowireBean(jobInstance);
	    	return jobInstance;
	    }
    }

然后在Spring中配置Quarzt的入口：

	<?xml version="1.0" encoding="GBK"?>
	<beans xmlns="http://www.springframework.org/schema/beans"
		xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
		xsi:schemaLocation="
	       http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-2.0.xsd">
	 
		<bean class="org.springframework.scheduling.quartz.SchedulerFactoryBean">
			<property name="jobFactory">
			    <bean class="me.arganzheng.study.quarzt.task.SpringBeanJobFactory" />
			</property>
			<property name="quartzProperties">
				<bean class="com.paipai.api.appstore.task.QuartzSchedulerInstanceIdGenerator" factory-method="generateInstanceIfForCurrentMachine">
					<constructor-arg>
						<props>
							<prop key="org.quartz.jobStore.dataSource">FuwuQuartzDataSource</prop>
							<prop key="org.quartz.dataSource.FuwuQuartzDataSource.driver">${jdbc.driverClassName}</prop>
							<prop key="org.quartz.dataSource.FuwuQuartzDataSource.URL">${jdbc.url}</prop>
							<prop key="org.quartz.dataSource.FuwuQuartzDataSource.user">${jdbc.username}</prop>
							<prop key="org.quartz.dataSource.FuwuQuartzDataSource.password">${jdbc.password}</prop>
							<prop key="org.quartz.jobStore.tablePrefix">T_APP_QRTZ_</prop>
							<prop key="org.quartz.jobStore.isClustered">true</prop>
							<prop key="org.quartz.jobStore.driverDelegateClass">org.quartz.impl.jdbcjobstore.MSSQLDelegate</prop>
							<prop key="org.quartz.jobStore.class">org.quartz.impl.jdbcjobstore.JobStoreTX</prop>
						</props>
					</constructor-arg>
					<constructor-arg>
						<value>org.quartz.scheduler.instanceId</value>
					</constructor-arg>
				</bean>
			</property>
			<property name="globalJobListeners">
				<list>
					<bean class="me.arganzheng.study.quarzt.task.QuartzEventListener" />
				</list>
			</property>
		</bean>
	</beans>

Quarzt的配置非常简单，就一个入口类`org.springframework.scheduling.quartz.SchedulerFactoryBean`。我们这里通过配置它的jobFactory为我们自定义的JobFactory来实现自动注入功能：

    <bean class="org.springframework.scheduling.quartz.SchedulerFactoryBean">
			<property name="jobFactory">
			    <bean class=”me.arganzheng.study.quarzt.task.SpringBeanJobFactory" />
			</property>
			…
	</bean>

spring的AutowireCapableBeanFactory其实非常强大，而且貌似任何通过Spring配置的bean都可以自动注入，这样就不用实现ApplicationContextAware接口了：
[How to inject dependencies into a self-instantiated object in Spring?](http://stackoverflow.com/questions/3813588/how-to-inject-dependencies-into-a-self-instantiated-object-in-spring)

    public class SpringBeanJobFactory extends org.springframework.scheduling.quartz.SpringBeanJobFactory{
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
    }

说明：对于Quarzt与Spring的整合问题，有其他的解决方案，比如这个：[Quartz and Spring Integration](http://techo-ecco.com/blog/quartz-and-spring-integration/)。

------------------------------------------------------------------

对于Spring配置的类，如果想要使用注解简化依赖bean配置，还可以使用AspectJ的 `Configurable`注解来实现依赖注入：[Aspect Oriented Programming with Spring](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/aop.html#aop-atconfigurable)
>
    @Configurable
    public class MyClass {
       @Autowired 
       private AnotherClass instance;
    }
    
> Then at creation time it will automatically inject its dependencies. You also should have `<context:spring-configured/>` in your application context xml.
** 说明：其实还需要MyClass注册在application context xml文件中。**

不用AspectJ的注解，其实Spring3也有类似的注解，主要用于Spring MVC：

** *注意* **：这里面有一个非常重要的原则，就是入口类（如上面的`MyClass`类）必须已经在spring中配置，而该入口类依赖的bean（如上面的`antherClass`），可以通过注解(如`autowired`)实现自动注入。而且要让spring在根据配置文件创建该入口类的时候，还额外的去解析该入口类的注解并且注入注解声明的类，需要在配置文件中额外的配置来告诉spring。比如上面的`<context:spring-configured/>`就是做这样的事情。  

一个完整的Configurable例子见这篇文档：[Spring, Aspects, @Configurable and Compile Time Weaving using maven](http://www.chrissearle.org/blog/technical/spring_aspects_configurable_and_compile_time_weaving_using_maven)

如果入口类也不想使用Spring配置文件，那么就需要额外的配置告诉Spring哪些入口类是需要你托管的，一般是包扫描：`<context:component-scan>`和特殊的类注解如@Controller，@Component, etc. [15. Web MVC framework-15.3.1 Defining a controller with @Controller](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/mvc.html)：
> ### 15.3.1 Defining a controller with @Controller

> The @Controller annotation indicates that a particular class serves the role of a controller. Spring does not require you to extend any controller base class or reference the Servlet API. However, you can still reference Servlet-specific features if you need to.

> The @Controller annotation acts as a stereotype for the annotated class, indicating its role. The dispatcher scans such annotated classes for mapped methods and detects @RequestMapping annotations (see the next section).

> You can define annotated controller beans explicitly, using a standard Spring bean definition in the dispatcher's context. However, the @Controller stereotype also allows for autodetection, aligned with Spring general support for detecting component classes in the classpath and auto-registering bean definitions for them.

> To enable autodetection of such annotated controllers, you add component scanning to your configuration. Use the spring-context schema as shown in the following XML snippet:

>     <?xml version="1.0" encoding="UTF-8"?>
      <beans xmlns="http://www.springframework.org/schema/beans" 
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xmlns:p="http://www.springframework.org/schema/p" 
        xmlns:context="http://www.springframework.org/schema/context"
        xsi:schemaLocation="
          http://www.springframework.org/schema/beans 
          http://www.springframework.org/schema/beans/spring-beans-3.0.xsd
          http://www.springframework.org/schema/context 
          http://www.springframework.org/schema/context/spring-context-3.0.xsd">
>          
         <context:component-scan base-package="org.springframework.samples.petclinic.web"/>
>         
         // ...
>         
    </beans>

stackoverflow上有一个非常详细讲解`<context:annotation-config>`和`<context:component-scan>`的帖子：
[Difference between <context:annotation-config> vs <context:component-scan>](http://stackoverflow.com/questions/7414794/difference-between-contextannotation-config-vs-contextcomponent-scan)。很长，这里就不quote了。简单来说就是两个步骤：

1. 扫描入口类：`<context:component-scan>` + `@Controller`, `@Component`, etc.
2. 通过注解方式注入入口类的依赖：`<context:annotation-config/>` 

*如果配置了1，那么自动包含2.*

当然，回到我们的主题，如果连入口类都不是Spring托管（不是xml配置，也不是anotation注解+包路径扫描），而是由框架或者应用创建的，那么就需要使用我们一开始介绍的方法来处理了。

*--EOF--*
