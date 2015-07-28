---
title: Spring各种依赖注入注解的区别
layout: post
---


Spring对于Bean的依赖注入，支持多种注解方式：


1. @Resource
	* javax.annotation	
	* JSR250 (Common Annotations for Java)
2. @Inject	
	* javax.inject	
	* JSR330 (Dependency Injection for Java)
3. @Autowired	
	* org.springframework.bean.factory	
	* Spring


直观上看起来，@Autowired是Spring提供的注解，其他几个都是JDK本身内建的注解，Spring对这些注解也进行了支持。但是使用起来这三者到底有什么区别呢？笔者经过方法的测试，发现一些有意思的特性。总结如下：


#### 1. @Autowired有个required属性，可以配置为false，这种情况下如果没有找到对应的bean是不会抛异常的。@Inject和@Resource没有提供对应的配置，所以必须找到否则会抛异常。

#### 2. @Autowired和@Inject基本是一样的，因为两者都是使用`AutowiredAnnotationBeanPostProcessor`来处理依赖注入。但是@Resource是个例外，它使用的是`CommonAnnotationBeanPostProcessor`来处理依赖注入。当然，两者都是`BeanPostProcessor`。

@Autowired和@Inject

* 默认 autowired by type
* 可以 通过@Qualifier 显式指定 autowired by qualifier name。
* 如果 autowired by type 失败（找不到或者找到多个实现），则退化为autowired by field name

@Resource

* 默认 autowired by field name
* 如果 autowired by field name失败，会退化为 autowired by type
* 可以 通过@Qualifier 显式指定 autowired by qualifier name
* 如果 autowired by qualifier name失败，会退化为 autowired by field name。但是这时候如果 autowired by field name失败，就不会再退化为autowired by type了。


##### **TIPS** Qualified name VS Bean name

在Spring设计中，Qualified name并不等同于Bean name，后者必须是唯一的，但是前者类似于tag或者group的作用，对特定的bean进行分类。可以达到getByTag(group)的效果。对于XML配置的bean，可以通过id属性指定bean name（如果没有指定，默认使用类名首字母小写），通过`<qualifier>`标签指定qualifier name：


	<bean id="lamborghini" class="me.arganzheng.study.spring.autowired.Lamborghini">
		<qualifier value="luxury"/>
	    <!-- inject any dependencies required by this bean -->
	</bean>

如果是通过注解方式，那么可以通过`@Qualifier`注解指定qualifier name，通过`@Named`或者@Component（@Service，@Repository等）的value值指定bean name：


	@Component("lamborghini")
	@Qualifier("luxury")
	public class Lamborghini implements Car {

	}

或者

	@Component
	@Named("lamborghini")
	@Qualifier("luxury")
	public class Lamborghini implements Car {

	}	


同样，如果没有指定bean name，那么Spring会默认是用类名首字母小写(Lamborghini=>lamborghini)。


#### 3. 通过Anotation注入依赖的方式在XML注入方式之前进行。如果对同一个bean的依赖同时使用了两种注入方式，那么XML的优先。但是不同担心通过Anotation注入的依赖没法注入XML中配置的bean，依赖注入是在bean的注册之后进行的。

#### 4. 目前的autowired by type方式（笔者用的是3.2.3.RELEASE版本），Spring的`AutowiredAnnotationBeanPostProcessor`实现都是有"bug"的，也就是说@Autowired和@Inject都是有坑的（称之为坑，不称之为bug是因为貌似是故意的。。）。这是来源于线上的一个bug，也是这边文章的写作原因。现场如下：


application-context.xml中有如下定义：

	<?xml version="1.0" encoding="UTF-8"?>
	<beans xmlns="http://www.springframework.org/schema/beans"
		xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:aop="http://www.springframework.org/schema/aop"
		xmlns:context="http://www.springframework.org/schema/context"
		xmlns:util="http://www.springframework.org/schema/util"
		xsi:schemaLocation="
			http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-3.0.xsd
			http://www.springframework.org/schema/aop http://www.springframework.org/schema/aop/spring-aop-3.0.xsd
			http://www.springframework.org/schema/context http://www.springframework.org/schema/context/spring-context-3.0.xsd
			http://www.springframework.org/schema/util http://www.springframework.org/schema/util/spring-util-2.5.xsd">

		<context:annotation-config />

		<context:component-scan base-package="me.arganzheng.study" />

		<util:constant id="en"
			static-field="me.arganzheng.study.spring.autowired.Constants.Language.EN" />
		<util:constant id="ja"
			static-field="me.arganzheng.study.spring.autowired.Constants.Language.JP" />
		<util:constant id="ind"
			static-field="me.arganzheng.study.spring.autowired.Constants.Language.IND" />
		<util:constant id="pt"
			static-field="me.arganzheng.study.spring.autowired.Constants.Language.PT" />
		<util:constant id="th"
			static-field="me.arganzheng.study.spring.autowired.Constants.Language.TH" />
		<util:constant id="ar"
			static-field="me.arganzheng.study.spring.autowired.Constants.Language.AR" />
		<util:constant id="en-rIn"
			static-field="me.arganzheng.study.spring.autowired.Constants.Language.EN_RIN" />

		<util:map id="languageChangesMap" key-type="java.lang.String"
			value-type="java.lang.String">
			<entry key="pt" value="pt" />
			<entry key="br" value="pt" />
			<entry key="jp" value="ja" />
			<entry key="ja" value="ja" />
			<entry key="ind" value="ind" />
			<entry key="id" value="ind" />
			<entry key="en-rin" value="en-rIn" />
			<entry key="in" value="en-rIn" />
			<entry key="en" value="en" />
			<entry key="gb" value="en" />
			<entry key="th" value="th" />
			<entry key="ar" value="ar" />
			<entry key="eg" value="ar" />
		</util:map>

	</beans>	

其中static-field应用的常量定义在如下类中：

	package me.arganzheng.study.spring.autowired;

	public interface Constants {

		public interface Language {
			public static final String EN = "CommonConstants.LANG_ENGLISH";
			public static final String JP = "CommonConstants.LANG_JAPANESE";
			public static final String IND = "CommonConstants.LANG_INDONESIAN";
			public static final String PT = "CommonConstants.LANG_PORTUGUESE";
			public static final String TH = "CommonConstants.LANG_THAI";
			public static final String EN_RIN = "CommonConstants.LANG_ENGLISH_INDIA";
			public static final String AR = "CommonConstants.LANG_Arabic";
		}

	}

然后如果我们在代码中如下声明依赖：


	public class AutowiredTest extends BaseSpringTestCase {

		@Autowired
		private Map<String, String> languageChangesMap;

		@Test
		public void testAutowired() {
			notNull(languageChangesMap);
			System.out.println(languageChangesMap.getClass().getSimpleName());
			System.out.println(languageChangesMap);
		}

	}

Guess what，诡异的事情发生了！

运行结果如下：

	LinkedHashMap
	{en=CommonConstants.LANG_ENGLISH, ja=CommonConstants.LANG_JAPANESE, ind=CommonConstants.LANG_INDONESIAN, pt=CommonConstants.LANG_PORTUGUESE, th=CommonConstants.LANG_THAI, ar=CommonConstants.LANG_Arabic, en-rIn=CommonConstants.LANG_ENGLISH_INDIA}

也就是说Map<String, String> languageChangesMap注入了`<util:constant>`变量。
如果把application-context中的那一串`<util:constant>`声明给删掉，则会报错：


	严重: Caught exception while allowing TestExecutionListener [org.springframework.test.context.support.DependencyInjectionTestExecutionListener@5c51ee0a] to prepare test instance [me.arganzheng.study.spring.autowired.AutowiredTest@6e301e0]
	org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'me.arganzheng.study.spring.autowired.AutowiredTest': Injection of autowired dependencies failed; nested exception is org.springframework.beans.factory.BeanCreationException: Could not autowire field: private java.util.Map me.arganzheng.study.spring.autowired.AutowiredTest.languageChangesMap; nested exception is org.springframework.beans.factory.NoSuchBeanDefinitionException: No qualifying bean of type [java.lang.String] found for dependency [map with value type java.lang.String]: expected at least 1 bean which qualifies as autowire candidate for this dependency. Dependency annotations: {@org.springframework.beans.factory.annotation.Autowired(required=true)}
		...
	Caused by: org.springframework.beans.factory.NoSuchBeanDefinitionException: No qualifying bean of type [java.lang.String] found for dependency [map with value type java.lang.String]: expected at least 1 bean which qualifies as autowire candidate for this dependency. Dependency annotations: {@org.springframework.beans.factory.annotation.Autowired(required=true)}
		at org.springframework.beans.factory.support.DefaultListableBeanFactory.raiseNoSuchBeanDefinitionException(DefaultListableBeanFactory.java:986)
		at org.springframework.beans.factory.support.DefaultListableBeanFactory.doResolveDependency(DefaultListableBeanFactory.java:843)
		at org.springframework.beans.factory.support.DefaultListableBeanFactory.resolveDependency(DefaultListableBeanFactory.java:768)
		at org.springframework.beans.factory.annotation.AutowiredAnnotationBeanPostProcessor$AutowiredFieldElement.inject(AutowiredAnnotationBeanPostProcessor.java:486)
		... 28 more


debug了一下，发现确实是Spring的一个bug。在DefaultListableBeanFactory的这个方法出问题了：

	protected Object doResolveDependency(DependencyDescriptor descriptor, Class<?> type, String beanName,
				Set<String> autowiredBeanNames, TypeConverter typeConverter) throws BeansException {
			...		
			
			else if (Map.class.isAssignableFrom(type) && type.isInterface()) {
				Class<?> keyType = descriptor.getMapKeyType();
				if (keyType == null || !String.class.isAssignableFrom(keyType)) {
					if (descriptor.isRequired()) {
						throw new FatalBeanException("Key type [" + keyType + "] of map [" + type.getName() +
								"] must be assignable to [java.lang.String]");
					}
					return null;
				}
				Class<?> valueType = descriptor.getMapValueType();
				if (valueType == null) {
					if (descriptor.isRequired()) {
						throw new FatalBeanException("No value type declared for map [" + type.getName() + "]");
					}
					return null;
				}
				Map<String, Object> matchingBeans = findAutowireCandidates(beanName, valueType, descriptor);
				if (matchingBeans.isEmpty()) {
					if (descriptor.isRequired()) {
						raiseNoSuchBeanDefinitionException(valueType, "map with value type " + valueType.getName(), descriptor);
					}
					return null;
				}
				if (autowiredBeanNames != null) {
					autowiredBeanNames.addAll(matchingBeans.keySet());
				}
				return matchingBeans;
			}
			...
		}

关键在这一句：`Map<String, Object> matchingBeans = findAutowireCandidates(beanName, valueType, descriptor);`。这里居然是用valueType作为类型去查找bean了，当然拿到所有为String类型的bean了。


这是autowired by type的问题，但是更让人诡异的是如果通过`@Qualifier("languageChangesMap")`显式指定autowired by name，还是报错了：


	严重: Caught exception while allowing TestExecutionListener [org.springframework.test.context.support.DependencyInjectionTestExecutionListener@9476189] to prepare test instance [me.arganzheng.study.spring.autowired.AutowiredTest@2d546e21]
	...
	Caused by: org.springframework.beans.factory.NoSuchBeanDefinitionException: No qualifying bean of type [java.lang.String] found for dependency [map with value type java.lang.String]: expected at least 1 bean which qualifies as autowire candidate for this dependency. Dependency annotations: {@org.springframework.beans.factory.annotation.Autowired(required=true), @org.springframework.beans.factory.annotation.Qualifier(value=languageChangesMap)}
		at org.springframework.beans.factory.support.DefaultListableBeanFactory.raiseNoSuchBeanDefinitionException(DefaultListableBeanFactory.java:986)
		at org.springframework.beans.factory.support.DefaultListableBeanFactory.doResolveDependency(DefaultListableBeanFactory.java:843)
		at org.springframework.beans.factory.support.DefaultListableBeanFactory.resolveDependency(DefaultListableBeanFactory.java:768)
		at org.springframework.beans.factory.annotation.AutowiredAnnotationBeanPostProcessor$AutowiredFieldElement.inject(AutowiredAnnotationBeanPostProcessor.java:486)
		... 28 more	


debug了一下，发现跟没有指定qualifie name是一样的执行路径。不是指定了bean name了吗？为什么还是autowired by type呢？仔细查看了一下才发现。DefaultListableBeanFactory的doResolveDependency方法对首先对类型做了区别：


	protected Object doResolveDependency(DependencyDescriptor descriptor, Class<?> type, String beanName,
				Set<String> autowiredBeanNames, TypeConverter typeConverter) throws BeansException {

			Object value = getAutowireCandidateResolver().getSuggestedValue(descriptor);
			if (value != null) {
				if (value instanceof String) {
					String strVal = resolveEmbeddedValue((String) value);
					BeanDefinition bd = (beanName != null && containsBean(beanName) ? getMergedBeanDefinition(beanName) : null);
					value = evaluateBeanDefinitionString(strVal, bd);
				}
				TypeConverter converter = (typeConverter != null ? typeConverter : getTypeConverter());
				return (descriptor.getField() != null ?
						converter.convertIfNecessary(value, type, descriptor.getField()) :
						converter.convertIfNecessary(value, type, descriptor.getMethodParameter()));
			}

			if (type.isArray()) {
				Class<?> componentType = type.getComponentType();
				Map<String, Object> matchingBeans = findAutowireCandidates(beanName, componentType, descriptor);
				if (matchingBeans.isEmpty()) {
					if (descriptor.isRequired()) {
						raiseNoSuchBeanDefinitionException(componentType, "array of " + componentType.getName(), descriptor);
					}
					return null;
				}
				if (autowiredBeanNames != null) {
					autowiredBeanNames.addAll(matchingBeans.keySet());
				}
				TypeConverter converter = (typeConverter != null ? typeConverter : getTypeConverter());
				return converter.convertIfNecessary(matchingBeans.values(), type);
			}
			else if (Collection.class.isAssignableFrom(type) && type.isInterface()) {
				Class<?> elementType = descriptor.getCollectionType();
				if (elementType == null) {
					if (descriptor.isRequired()) {
						throw new FatalBeanException("No element type declared for collection [" + type.getName() + "]");
					}
					return null;
				}
				Map<String, Object> matchingBeans = findAutowireCandidates(beanName, elementType, descriptor);
				if (matchingBeans.isEmpty()) {
					if (descriptor.isRequired()) {
						raiseNoSuchBeanDefinitionException(elementType, "collection of " + elementType.getName(), descriptor);
					}
					return null;
				}
				if (autowiredBeanNames != null) {
					autowiredBeanNames.addAll(matchingBeans.keySet());
				}
				TypeConverter converter = (typeConverter != null ? typeConverter : getTypeConverter());
				return converter.convertIfNecessary(matchingBeans.values(), type);
			}
			else if (Map.class.isAssignableFrom(type) && type.isInterface()) {
				Class<?> keyType = descriptor.getMapKeyType();
				if (keyType == null || !String.class.isAssignableFrom(keyType)) {
					if (descriptor.isRequired()) {
						throw new FatalBeanException("Key type [" + keyType + "] of map [" + type.getName() +
								"] must be assignable to [java.lang.String]");
					}
					return null;
				}
				Class<?> valueType = descriptor.getMapValueType();
				if (valueType == null) {
					if (descriptor.isRequired()) {
						throw new FatalBeanException("No value type declared for map [" + type.getName() + "]");
					}
					return null;
				}
				Map<String, Object> matchingBeans = findAutowireCandidates(beanName, valueType, descriptor);
				if (matchingBeans.isEmpty()) {
					if (descriptor.isRequired()) {
						raiseNoSuchBeanDefinitionException(valueType, "map with value type " + valueType.getName(), descriptor);
					}
					return null;
				}
				if (autowiredBeanNames != null) {
					autowiredBeanNames.addAll(matchingBeans.keySet());
				}
				return matchingBeans;
			}
			else {
				Map<String, Object> matchingBeans = findAutowireCandidates(beanName, type, descriptor);
				if (matchingBeans.isEmpty()) {
					if (descriptor.isRequired()) {
						raiseNoSuchBeanDefinitionException(type, "", descriptor);
					}
					return null;
				}
				if (matchingBeans.size() > 1) {
					String primaryBeanName = determinePrimaryCandidate(matchingBeans, descriptor);
					if (primaryBeanName == null) {
						throw new NoUniqueBeanDefinitionException(type, matchingBeans.keySet());
					}
					if (autowiredBeanNames != null) {
						autowiredBeanNames.add(primaryBeanName);
					}
					return matchingBeans.get(primaryBeanName);
				}
				// We have exactly one match.
				Map.Entry<String, Object> entry = matchingBeans.entrySet().iterator().next();
				if (autowiredBeanNames != null) {
					autowiredBeanNames.add(entry.getKey());
				}
				return entry.getValue();
			}
		}

如果是Array，Collection或者Map，则根据集合类中元素的类型来进行autowired by type（Map使用value的类型）。为什么这么特殊处理呢？原来，Spring是为了达到这样的目的：让你可以一次注入所有符合类型的实现，也就是说可以这样子注入：

	@Autowired
	private List<Car> cars;

如果你的car有多个实现，那么都会注入进来，不会再报

	org.springframework.beans.factory.NoSuchBeanDefinitionException: 
	No unique bean of type [me.arganzheng.study.spring.autowired.Car] is defined: 
	expected single matching bean but found 2: [audi, toyota].

然而，上面的情况如果你用@Resource则不会有这个问题：

	public class AutowiredTest extends BaseSpringTestCase {

		@Resource
		@Qualifier("languageChangesMap")
		private Map<String, String> languageChangesMap;

		@Test
		public void testAutowired() {
			assertNotNull(languageChangesMap);
			System.out.println(languageChangesMap.getClass().getSimpleName());
			System.out.println(languageChangesMap);
		}
	}


正常运行：

	LinkedHashMap
	{pt=pt, br=pt, jp=ja, ja=ja, ind=ind, id=ind, en-rin=en-rIn, in=en-rIn, en=en, gb=en, th=th, ar=ar, eg=ar}

当然，你如果不指定`@Qualifier("languageChangesMap")`，同时field name不是languageChangesMap，那么还是一样报错的。
	
	Caused by: org.springframework.beans.factory.NoSuchBeanDefinitionException: No qualifying bean of type [java.lang.String] found for dependency [map with value type java.lang.String]: expected at least 1 bean which qualifies as autowire candidate for this dependency. Dependency annotations: {@javax.annotation.Resource(shareable=true, mappedName=, description=, name=, type=class java.lang.Object, authenticationType=CONTAINER, lookup=)}
	at org.springframework.beans.factory.support.DefaultListableBeanFactory.raiseNoSuchBeanDefinitionException(DefaultListableBeanFactory.java:986)
	at org.springframework.beans.factory.support.DefaultListableBeanFactory.doResolveDependency(DefaultListableBeanFactory.java:843)
	at org.springframework.beans.factory.support.DefaultListableBeanFactory.resolveDependency(DefaultListableBeanFactory.java:768)
	at org.springframework.context.annotation.CommonAnnotationBeanPostProcessor.autowireResource(CommonAnnotationBeanPostProcessor.java:438)
	at org.springframework.context.annotation.CommonAnnotationBeanPostProcessor.getResource(CommonAnnotationBeanPostProcessor.java:416)
	at org.springframework.context.annotation.CommonAnnotationBeanPostProcessor$ResourceElement.getResourceToInject(CommonAnnotationBeanPostProcessor.java:550)
	at org.springframework.beans.factory.annotation.InjectionMetadata$InjectedElement.inject(InjectionMetadata.java:150)
	at org.springframework.beans.factory.annotation.InjectionMetadata.inject(InjectionMetadata.java:87)
	at org.springframework.context.annotation.CommonAnnotationBeanPostProcessor.postProcessPropertyValues(CommonAnnotationBeanPostProcessor.java:303)
	... 26 more


而且，@Resource也可以实现上面的List接收所有实现：

	public class AutowiredTest extends BaseSpringTestCase {

		@Resource
		@Qualifier("languageChangesMap")
		private Map<String, String> languageChangesMap;

		@Resource
		private List<Car> cars;

		@Test
		public void testAutowired() {
			assertNotNull(languageChangesMap);
			System.out.println(languageChangesMap.getClass().getSimpleName());
			System.out.println(languageChangesMap);

			assertNotNull(cars);
			System.out.println(cars.getClass().getSimpleName());
			System.out.println(cars);
		}
	}


运行的妥妥的：

	LinkedHashMap
	{pt=pt, br=pt, jp=ja, ja=ja, ind=ind, id=ind, en-rin=en-rIn, in=en-rIn, en=en, gb=en, th=th, ar=ar, eg=ar}
	ArrayList
	[me.arganzheng.study.spring.autowired.Audi@579584da, me.arganzheng.study.spring.autowired.Toyota@19453122]


这是因为@Resource注解使用的是`CommonAnnotationBeanPostProcessor`处理器，跟`AutowiredAnnotationBeanPostProcessor`不是同一个作者[/偷笑]。这里就不分析了，感兴趣的同学可以自己看代码研究一下。


最终结论如下：

@Autowired和@Inject

* autowired by type
* 可以 通过@Qualifier 显式指定 autowired by qualifier name（非集合类。注意：不是autowired by bean name！）
* 如果 autowired by type 失败（找不到或者找到多个实现），则退化为autowired by field name（非集合类）

@Resource

* 默认 autowired by field name
* 如果 autowired by field name失败，会退化为 autowired by type
* 可以 通过@Qualifier 显式指定 autowired by qualifier name
* 如果 autowired by qualifier name失败，会退化为 autowired by field name。但是这时候如果 autowired by field name失败，就不会再退化为autowired by type了。


测试工程保存在[GitHub](https://github.com/arganzheng/quick-test)上，是标准的maven工程，感兴趣的同学可以clone到本地运行测试一下。


补充
----

有同事指出Spring官方文档上有这么一句话跟我的结有点冲突：

> However, although you can use this convention to refer to specific beans by name, @Autowired is fundamentally about type-driven injection with optional semantic qualifiers. This means that qualifier values, even with the bean name fallback, always have narrowing semantics within the set of type matches; they do not semantically express a reference to a unique bean id.

也就是说@Autowired即使加了@Qualifier注解，其实也是autowired by type。@Qualifier只是一个限定词，过滤条件而已。重新跟进了一下代码，发现确实是这样子的。Spring设计的这个 @Qualifier name 并不等同于 bean name。他有点类似于一个tag。不过如果这个tag是唯一的化，那么其实效果上等同于bean name。实现上，Spring是先getByType，得到list candicates，然后再根据qualifier name进行过滤。


再定义一个兰博基尼，这里使用@Qualifier指定：


	package me.arganzheng.study.spring.autowired;

	import org.springframework.beans.factory.annotation.Qualifier;
	import org.springframework.stereotype.Component;

	@Component
	@Qualifier("luxury")
	public class Lamborghini implements Car {

	}


再定义一个劳斯莱斯，这里故意用@Named指定：


	package me.arganzheng.study.spring.autowired;

	import javax.inject.Named;

	import org.springframework.stereotype.Component;

	@Component
	@Named("luxury")
	public class RollsRoyce implements Car {

	}


测试一下注入定义的豪华车：

	package me.arganzheng.study.spring.autowired;

	import static junit.framework.Assert.assertNotNull;

	import java.util.List;

	import me.arganzheng.study.BaseSpringTestCase;

	import org.junit.Test;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.beans.factory.annotation.Qualifier;

	/**
	 * 
	 * @author zhengzhibin
	 * 
	 */
	public class AutowiredTest extends BaseSpringTestCase {

		@Autowired
		@Qualifier("luxury")
		private List<Car> luxuryCars;

		@Test
		public void testAutowired() {

			assertNotNull(luxuryCars);
			System.out.println(luxuryCars.getClass().getSimpleName());
			System.out.println(luxuryCars);
		}

	}


运行结果如下：

	ArrayList
	[me.arganzheng.study.spring.autowired.Lamborghini@66b875e1, me.arganzheng.study.spring.autowired.RollsRoyce@58433b76]


补充：[Autowiring modes](http://docs.spring.io/spring-framework/docs/4.2.x/spring-framework-reference/htmlsingle/#beans-factory-autowire)
---------

Spring支持四种autowire模式，当使用XML配置方式时，你可以通过autowire属性指定。

1. no. (Default) No autowiring. Bean references must be defined via a ref element. Changing the default setting is not recommended for larger deployments, because specifying collaborators explicitly gives greater control and clarity. To some extent, it documents the structure of a system.
2. byName. Autowiring by property name. Spring looks for a bean with the same name as the property that needs to be autowired. For example, if a bean definition is set to autowire by name, and it contains a master property (that is, it has a setMaster(..) method), Spring looks for a bean definition named master, and uses it to set the property.
3. byType. Allows a property to be autowired if exactly one bean of the property type exists in the container. If more than one exists, a fatal exception is thrown, which indicates that you may not use byType autowiring for that bean. If there are no matching beans, nothing happens; the property is not set.
4. constructor. Analogous to byType, but applies to constructor arguments. If there is not exactly one bean of the constructor argument type in the container, a fatal error is raised.

如果使用@Autowired、@Inject或者@Resource注解的时候，则稍微复杂一些，会有一个失败退化过程，并且引入了Qualifier。不过基本原理是一样。

参考文章
--------

1. [Spring Injection with @Resource, @Autowired and @Inject](http://blogs.sourceallies.com/2011/08/spring-injection-with-resource-and-autowired/#more-2350)
2. [@Resource vs @Autowired](http://stackoverflow.com/questions/4093504/resource-vs-autowired)
3. [5.9 beans-autowired-annotation](http://docs.spring.io/spring/docs/3.2.9.BUILD-SNAPSHOT/spring-framework-reference/htmlsingle/#beans-autowired-annotation)
