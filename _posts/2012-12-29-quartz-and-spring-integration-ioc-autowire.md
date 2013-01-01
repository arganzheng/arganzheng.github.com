---
layout: post
title: Quartz与Spring的整合-Quartz中的job如何自动注入spring容器托管的对象
---


## 问题

Quartz中的job是由Quartz框架动态创建的（配置该job的classname，通过反射创建），而job一般会依赖到配置在spring中的bean，怎样获取或者更好的自动注入这些依赖bean呢？

## 预期效果

我们希望达到这样的效果：

    /**
     * 
     * 取消超时未支付订单的任务。 
     *
     * @author arganzheng
     */
    public class CancelUnpaidOrderTask implements Job {
	    @Autowired
	    private AppOrderService orderService;

	    @Override
	    public void execute(JobExecutionContext ctx) throws JobExecutionException {
		    ...
    }

关键在这一行：

    @Autowired
	private AppOrderService orderService;

orderService是配置在spring容器中的，而CancelUnpaidOrderTask则是配置在Quartz数据库中，由`org.springframework.scheduling.quartz.SpringBeanJobFactory` 运行时调用`
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
    
当然，你需要在你的xml配置文件中定义FooServicesLocator和FooService。然后在需要引用FooService的地方，就可以这样子获取FooService了：`FooServicesLocator.getFoobarServic();` 得到Spring托管的FooService。

不过这样是依赖查询，不是注入，要实现DI，可以使用`AutowireCapableBeanFactory`进行autowire。     

    applicationContext.getAutowireCapableBeanFactory().autowireBean(existingBean);

于是对于上面的那个问题，就有了如下的解决方案：

    package me.arganzheng.study.quartz.task.SpringBeanJobFactory;
    
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

然后在Spring中配置Quartz的入口：

	<?xml version="1.0" encoding="GBK"?>
	<beans xmlns="http://www.springframework.org/schema/beans"
		xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
		xsi:schemaLocation="
	       http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-2.0.xsd">
	 
		<bean class="org.springframework.scheduling.quartz.SchedulerFactoryBean">
			<property name="jobFactory">
			    <bean class="me.arganzheng.study.quartz.task.SpringBeanJobFactory" />
			</property>
			
		    ...
			
		</bean>
	</beans>

对于数据库配置方式的Quartz，配置非常简单，就一个入口类`org.springframework.scheduling.quartz.SchedulerFactoryBean`。我们这里通过配置它的jobFactory为我们自定义的JobFactory来实现自动注入功能：

    <bean class="org.springframework.scheduling.quartz.SchedulerFactoryBean">
			<property name="jobFactory">
			    <bean class=”me.arganzheng.study.quartz.task.SpringBeanJobFactory" />
			</property>
			...
	</bean>

** *注意* **：上面的XML配置采用了直接配置jobFactory属性的方式将jobFactory配置为我们自定义的jobFactory类，默认是`org.springframework.scheduling.quartz.SpringBeanJobFactory`。虽然Quartz允许我们通过`org.quartz.scheduler.jobFactory.class`配置项配置自定义的jobFactory:

> #### org.quartz.scheduler.jobFactory.class 
> 
>The class name of the JobFactory to use. The default is `org.quartz.simpl.SimpleJobFactory`, you may like to try `org.quartz.simpl.PropertySettingJobFactory`. A JobFatcory is responsible for producing instances of JobClasses. SimpleJobFactory simply calls newInstance() on the class. PropertySettingJobFactory does as well, but also reflectively sets the job's bean properties using the contents of the JobDataMap.

但是注意到我们配置的是Spring封装的是`org.springframework.scheduling.quartz.SchedulerFactoryBean`，它并不认这个配置项目。因为它已经将它的jobFactory由`org.quartz.simpl.SimpleJobFactory`改为`org.springframework.scheduling.quartz.SpringBeanJobFactory`，所以只能采用配置jobFactory属性的方式修改jobFactory为我们的自定义factory。

spring的AutowireCapableBeanFactory其实非常强大，Spring3.0允许任何通过Spring配置的bean都可以自动注入它所属的上下文，也就是说默认所有的bean都自动实现了ApplicationContextAware接口，这样就不用显示实现ApplicationContextAware接口了（ 是不是更POJO:) ）：
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

关于使用`ApplicationContextAware`和`AutowireCapableBeanFactory`实现@Autowired功能，在stackoverflow上这个帖子有很详细的说明，可以参考一下：[How to get beans created by FactoryBean spring managed?](http://stackoverflow.com/questions/4970297/how-to-get-beans-created-by-factorybean-spring-managed)

## 其他解决方案

对于Quartz与Spring的整合问题，spring其实提供了很多内建方案：

1. 使用`org.springframework.scheduling.quartz.JobDetailBean`+`jobDataAsMap`：比如这个：[Spring 3 + Quartz 1.8.6 Scheduler Example](http://www.mkyong.com/spring/spring-quartz-scheduler-example/)。不过貌似不推荐.
2. 使用`org.springframework.scheduling.quartz.SchedulerFactoryBean`+`schedulerContextAsMap`：比如这个：[Quartz and Spring Integration](http://techo-ecco.com/blog/quartz-and-spring-integration/)。
3. 使用`org.springframework.scheduling.quartz.MethodInvokingJobDetailFactoryBean`：这个可以让任何定义在spring中的类成为Quartz要求的job。比如这个：[25.6 Using the OpenSymphony Quartz Scheduler](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/scheduling.html#scheduling-quartz)
4. 使用`org.springframework.scheduling.quartz.SchedulerFactoryBean`+`applicationContextSchedulerContextKey`：比如这个：[Accessing Spring beans from Quartz jobs](http://blog.mark-mclaren.info/2007/06/accessing-spring-beans-from-quartz-jobs_1652.html)

每种方法笔者都认真的看过，而且找的例子都是非常不错的例子。个人感觉3和4不错，尤其是4。3使用起来有点像spring的事务配置，4使用起来有点像在web层通过`WebApplicationContextUtils`得到spring的`ApplicationContext`。不过这几种方式都不是依赖注入，而且配置信息比较多。所以还是推荐上面的`org.springframework.scheduling.quartz.SchedulerFactoryBean`+AutowireCapableBeanFactory的`SpringBeanJobFactory`解决方案:)

------------------------------------------------------------------

`@Autowired`注解大大节省了Spring的xml配置，将bean依赖关系声明转移到类文件和运行期。即：
原来需要这样的配置：
    <bean id="thisClass" class="me.arganzheng.study.MyClass" />
      <property name="anotherClass" ref="anotherClass" />
    </bean>
    <bean id="anotherClass" class="me.arganzheng.study.AnotherClass">
    </bean>
    
    package me.arganzheng.study;
    
    public class MyClass { 
      private Another anotherClass;
     
      public void setAnotherClass(AnotherClass anotherClass) {
        this.anotherClass = anotherClass; 
      }
    }

使用`@Autowired`注解可以简化为：

    <bean id="thisClass" class="me.arganzheng.study.MyClass" />
    </bean>
    <bean id="anotherClass" class="me.arganzheng.study.AnotherClass">
    </bean>
    
    package me.arganzheng.study;
    
    public class MyClass { 
      @Autowired
      private Another anotherClass;
    }

不过这样MyClass本身在Spring配置文件中定义，而它的依赖又是在自身类文件通过`@Autowired`注解声明，需要有种方式告诉Spring说当你根据配置文件创建我的时候，麻烦也扫描一下我的注解，把通过注解声明的依赖也注入进来。这可以通过Spring的`<context:spring-configured/>`配置+AspectJ的 `Configurable`注解来实现运行期依赖注入：[Aspect Oriented Programming with Spring](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/aop.html#aop-atconfigurable)
>
    @Configurable
    public class MyClass {
       @Autowired 
       private AnotherClass instance;
    }
    
> Then at creation time it will automatically inject its dependencies. You also should have `<context:spring-configured/>` in your application context xml.
** 说明：其实还需要MyClass注册在application context xml文件中。**

不用AspectJ的注解，其实Spring3也有类似的注解，主要用于Spring MVC：

** *注意* **：这里面有一个非常重要的前提，就是所有的类（如上面的`MyClass`和`AnotherClass`）都必须已经在spring中配置，只是这些bean直接的依赖关系（如上面的`MyClass`依赖于`AntherClass`），可以通过注解(如`@autowired`)实现运行期自动注入。而且要让spring在根据配置文件创建该这些bean的时候，还额外的去解析该bean的注解并且注入通过注解声明的依赖bean，需要在配置文件中额外的配置来告诉spring。比如上面的`<context:spring-configured/>`就是做这样的事情。  

一个完整的Configurable例子见这篇文档：[Spring, Aspects, @Configurable and Compile Time Weaving using maven](http://www.chrissearle.org/blog/technical/spring_aspects_configurable_and_compile_time_weaving_using_maven)

如果bean本身（不即使依赖关系）也不想使用Spring配置文件注册，那么就需要额外的配置告诉Spring哪些类是需要你托管的，一般是包扫描：`<context:component-scan>`+特殊的类注解如@Controller，@Component, etc. [15. Web MVC framework-15.3.1 Defining a controller with @Controller](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/mvc.html)：
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

1. 扫描类：`<context:component-scan>` + `@Controller`, `@Component`, etc.
2. 通过注解方式注入该类的依赖：`<context:annotation-config/>` 

*如果配置了1，那么自动包含2.*

当然，回到我们的主题，如果有些bean不应该由Spring托管（不是xml配置，也不是anotation注解+包路径扫描），而是由框架或者应用创建的，那么就需要使用我们一开始介绍的方法来处理了。

*--EOF--*

