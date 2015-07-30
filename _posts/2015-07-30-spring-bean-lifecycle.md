---
title: Spring的Bean生命周期
layout: post
---


Bean扩展点
---------

1、JSR-250注解(推荐方式)

* @PostConstruct
* @PreDestroy

2、实现Spring接口

* InitializingBean: afterPropertiesSet()
* DisposableBean: destroy()

3、XML配置

* init-method, default-init-method
* destroy-method, default-destroy-method 


容器扩展点
---------

### 1、Customizing beans using a BeanPostProcessor

BeanPostProcessor提供了一个机制让你可以修改创建的Bean实例，比如AOP，比如依赖注入。

	package org.springframework.beans.factory.config;

	import org.springframework.beans.BeansException;

	/**
	 * Factory hook that allows for custom modification of new bean instances,
	 * e.g. checking for marker interfaces or wrapping them with proxies.
	 *
	 * <p>ApplicationContexts can autodetect BeanPostProcessor beans in their
	 * bean definitions and apply them to any beans subsequently created.
	 * Plain bean factories allow for programmatic registration of post-processors,
	 * applying to all beans created through this factory.
	 *
	 * <p>Typically, post-processors that populate beans via marker interfaces
	 * or the like will implement {@link #postProcessBeforeInitialization},
	 * while post-processors that wrap beans with proxies will normally
	 * implement {@link #postProcessAfterInitialization}.
	 *
	 * @author Juergen Hoeller
	 * @since 10.10.2003
	 * @see InstantiationAwareBeanPostProcessor
	 * @see DestructionAwareBeanPostProcessor
	 * @see ConfigurableBeanFactory#addBeanPostProcessor
	 * @see BeanFactoryPostProcessor
	 */
	public interface BeanPostProcessor {

		/**
		 * Apply this BeanPostProcessor to the given new bean instance <i>before</i> any bean
		 * initialization callbacks (like InitializingBean's {@code afterPropertiesSet}
		 * or a custom init-method). The bean will already be populated with property values.
		 * The returned bean instance may be a wrapper around the original.
		 * @param bean the new bean instance
		 * @param beanName the name of the bean
		 * @return the bean instance to use, either the original or a wrapped one; if
		 * {@code null}, no subsequent BeanPostProcessors will be invoked
		 * @throws org.springframework.beans.BeansException in case of errors
		 * @see org.springframework.beans.factory.InitializingBean#afterPropertiesSet
		 */
		Object postProcessBeforeInitialization(Object bean, String beanName) throws BeansException;

		/**
		 * Apply this BeanPostProcessor to the given new bean instance <i>after</i> any bean
		 * initialization callbacks (like InitializingBean's {@code afterPropertiesSet}
		 * or a custom init-method). The bean will already be populated with property values.
		 * The returned bean instance may be a wrapper around the original.
		 * <p>In case of a FactoryBean, this callback will be invoked for both the FactoryBean
		 * instance and the objects created by the FactoryBean (as of Spring 2.0). The
		 * post-processor can decide whether to apply to either the FactoryBean or created
		 * objects or both through corresponding {@code bean instanceof FactoryBean} checks.
		 * <p>This callback will also be invoked after a short-circuiting triggered by a
		 * {@link InstantiationAwareBeanPostProcessor#postProcessBeforeInstantiation} method,
		 * in contrast to all other BeanPostProcessor callbacks.
		 * @param bean the new bean instance
		 * @param beanName the name of the bean
		 * @return the bean instance to use, either the original or a wrapped one; if
		 * {@code null}, no subsequent BeanPostProcessors will be invoked
		 * @throws org.springframework.beans.BeansException in case of errors
		 * @see org.springframework.beans.factory.InitializingBean#afterPropertiesSet
		 * @see org.springframework.beans.factory.FactoryBean
		 */
		Object postProcessAfterInitialization(Object bean, String beanName) throws BeansException;

	}


**注意**

> BeanPostProcessors operate on bean (or object) instances; that is to say, the Spring IoC container instantiates a bean instance and then BeanPostProcessors do their work.
> BeanPostProcessors are scoped per-container. This is only relevant if you are using container hierarchies. If you define a BeanPostProcessor in one container, it will only post-process the beans in that container. In other words, beans that are defined in one container are not post-processed by a BeanPostProcessor defined in another container, even if both containers are part of the same hierarchy.
> To change the actual bean definition (i.e., the blueprint that defines the bean), you instead need to use a BeanFactoryPostProcessor as described in Section 6.8.2, “Customizing configuration metadata with a BeanFactoryPostProcessor”.


### 2、Customizing configuration metadata with a BeanFactoryPostProcessor


BeanFactoryPostProcessor对bean的配置元数据进行操作，可以通过BeanFactoryPostProcessor读取配置元信息，并且可以在Spring容器实例化bean之前修改这些元数据。 

最常见的用法就是配置文件占位符的替换。PropertyOverrideConfigurer 和 PropertyPlaceholderConfigurer，就是典型的例子。我们也可以自定义一个PropertyPlaceholderConfigurer，让占位符不是从properties文件读取，而是从配置中心读取。


### 3、Customizing instantiation logic with a FactoryBean

The FactoryBean interface is a point of pluggability into the Spring IoC container’s instantiation logic. If you have complex initialization code that is better expressed in Java as opposed to a (potentially) verbose amount of XML, you can create your own FactoryBean, write the complex initialization inside that class, and then plug your custom FactoryBean into the container.

Spring框架本身大量使用了FactoryBean来隐藏对象的复杂构建逻辑，大概有50+个实现。比如我们前面在介绍Quartz与Spring的整合的时候提到的[SchedulerFactoryBean](http://blog.arganzheng.me/posts/quartz-and-spring-integration-dynamic-instanceid-with-factorybean.html)。把初始化逻辑放在FactoryBean中实现，然后在XML中配置FactoryBean，但是最后却是得到FactoryBean生产的Bean，有点狸猫换太子的味道。

	package org.springframework.beans.factory;

	
	public interface FactoryBean<T> {

		
		T getObject() throws Exception;

		
		Class<?> getObjectType();

		
		boolean isSingleton();

	}



参考文档
-------

1. [6.8.2 Customizing configuration metadata with a BeanFactoryPostProcessor](http://docs.spring.io/spring-framework/docs/4.2.x/spring-framework-reference/htmlsingle/#beans-factory-extension-factory-postprocessors)


