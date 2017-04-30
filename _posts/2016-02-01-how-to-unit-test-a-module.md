---
title: 如何单元测试二方库
layout: post
catalog: true
---

问题
----

为了模块化，我们需要将通用的模块拆分出来，作为一个单独的二方库，在多个工程中引用。比如product-biz, product-dataaccess等。这些模块一般会使用到其他框架，比如Spring，mybatis，并且可能还有一些配置项根据不同的工程有不同的配置，所有这些依赖（配置文件和配置项）一般都是在引用方提供，然后整个工程才能够跑起来。这样会带来一个严重的问题，就是这些二方库不具有自治性。他们的代码和配置是分离的，而没有配置，单元测试也没法跑起来。


解决方案
-------

要让单元测试可以run起来，配置是必须提供的，包括框架需要的配置，比如Spring的bean配置，mybatis配置，还有相关的配置项。这样该二方库就是自治的了。他依赖的东西都可以到位，为他编写单元测试是一件简单自然的事情：

比如以前的B2B就是这样子做的：

	arganzhengs-MacBook-Pro:src argan$ tree -L 5
	.
	├── conf
	│   ├── META-INF
	│   │   ├── autoconf
	│   │   │   ├── auto-config.xml
	│   │   │   └── mlproduct.malanda.properties.vm
	│   │   └── services
	│   │       └── com.alibaba.intl.sourcing.biz.mlproduct.uri.InfoBuilder
	│   └── biz
	│       ├── ibatis
	│       │   └── sql-map-mlproduct.xml
	│       ├── mlCommon_services_locator.xml
	│       ├── mlProduct_services_locator.xml
	│       ├── product_attribute_custom_provider.properties
	│       └── spring
	│           ├── ml-html-filter.xml
	│           ├── mlproduct_dal.xml
	│           ├── mlproduct_service.xml
	│           ├── spring_mlcommons_service.xml
	│           ├── spring_mlproduct_attribute.xml
	│           └── spring_mlproduct_verify.xml
	├── java
	│   └── com
	│       └── alibaba
	│           └── intl
	│               └── sourcing
	└── java.test
	    └── com
	        └── alibaba
	            └── intl
	                └── sourcing

	17 directories, 13 files


可以看到它是通过ServicesLocator模式来解决这个问题:

	$ cat mlProduct_services_locator.xml 

	<?xml version="1.0" encoding="UTF-8"?>

	<configuration>
		<services>
			<service name="MLProductServicesLocator" class="com.alibaba.service.spring.DefaultBeanFactoryService">
				<property name="bean.parent" value="DatasourceMLProductPostingServicesLocator" />
				<property name="bean.descriptors">
					<value>/biz/spring/mlproduct_dal.xml</value>
					<value>/biz/spring/mlproduct_service.xml</value>
					<value>/biz/spring/spring_mlproduct_attribute.xml</value>
					<value>/biz/spring/spring_mlproduct_verify.xml</value>
				</property>
			</service>
		</services>
	</configuration>

而这个MLProductServicesLocator的定义其实很简单：

	package com.alibaba.intl.commons.service;

	import org.springframework.context.ApplicationContext;

	import com.alibaba.service.Service;
	import com.alibaba.service.ServiceManager;
	import com.alibaba.service.resource.ResourceLoaderService;
	import com.alibaba.service.spring.BeanFactoryService;

	public class AbstractServiceLocator {
	    private static ServiceManager        manager;
	    private static ResourceLoaderService resourceLoaderService;
	    private static RuntimeException      initException;

	    static {
	        System.setProperty("sun.net.client.defaultConnectTimeout", "30000");
	        System.setProperty("sun.net.client.defaultReadTimeout", "30000");

	        try {
	            manager = SingletonServiceManagerLoader.getInstance(new String[] { "classpath*:/biz/*_services_locator.xml" });
	            resourceLoaderService = (ResourceLoaderService) manager.getService(ResourceLoaderService.SERVICE_NAME);
	        } catch (RuntimeException e) {
	            initException = e;
	        } catch (Throwable t) {
	            initException = new RuntimeException("AbstractServiceLocator init error", t);
	        }
	    }

	    protected static ApplicationContext getApplicationContext(String key) {
	        if (initException != null) {
	            throw initException;
	        }

	        BeanFactoryService beanFactoryService = (BeanFactoryService) manager.getService(key);

	        return beanFactoryService.getBeanFactory();
	    }

	    protected static ResourceLoaderService getResourceLoaderService() {
	        if (initException != null) {
	            throw initException;
	        }
	        return resourceLoaderService;
	    }

	    protected static Service getService(String key) {
	        if (initException != null) {
	            throw initException;
	        }
	        return manager.getService(key);
	    }

	    public static void initAllService() {
	        manager.initAll();
	    }
	}

关键在于静态初始化块中这句话：
	
	manager = SingletonServiceManagerLoader.getInstance(new String[] { "classpath*:/biz/*_services_locator.xml" });

通过启动一个SingletonServiceManagerLoader来加载所有`/biz/*_services_locator.xml`文件，每个`*_services_locator.xml`配置文件定义了一个BeanFactoryService，例如对于前面的MLProductServiceLocator，其相应的配置文件为mlProduct_services_locator.xml。

具体可以参见: [Webx Framework](http://www.openwebx.org/) 及其实例项目 [petstore](https://github.com/webx/citrus-sample/tree/master/petstore)。

当然对应到maven的标准目录结构下，我们可以这样子安排我们的目录结构：

	arganzhengs-MacBook-Pro:biz argan$ tree
	.
	├── pom.xml
	└── src
	    ├── main
	    │   ├── java
	    │   └── resources
	    │       └── petstore
	    │           └── petstore-biz.xml
	    └── test
	        ├── java
	        └── resources

	9 directories, 2 files

其中配置文件存放在resources目录下。

但是还有一个问题没有解决，就是配置项的注入。不同的环境(dev、test、idc)，不同的工程对同一个二方库的配置是不一样的。所以petstore-biz.xml需要用到PropertyPlaceholder，这些properties文件是否也需要放在二方库中呢？这个问题比较难处理，因为具体的配置确实是因上层应用而异。如果把应用相关的配置存放在二方库中，那么会导致二方库对上层有感知，这个反向依赖关系有点不优雅。这里有个折中的方案，就是二方库只包含dev和test环境的配置，正式环境的配置交给引用该二方库的应用来配置。以前B2B的做法是采用auto-config的方式，把占位符在编译期间替换掉。滔滔写的单元测试组件也会在启动单元测试的时候做auto-config这件事，从而保证单元测试也能够正常运行。不过具体的配置项值也并不是跟着二方库走的，以前是在home目录下的一个autoconfig.properties文件（SVN版本管理），后面也是改成统一的线上配置中心。不过因为采用的是auto-config方式，最终编译出来的jar或者war包，并不需要包含properties文件，因为配置项的值已经在编译期间打到配置文件中了。也就是说虽然源码级别由于确实配置项值运行不了，但是编译后的二方库是可以直接运行的。

总结一下，二方库一般包含如下几个部分：

1. 代码
2. 配置文件（Spring，myBatis，etc.，其中Spring的配置通常可以通过 注解+包扫描 去除，myBatis也是可以通过注解替代XML配置）
3. 配置项 （properties文件，或者配置中心）

因此有如下几种组织方式：

1. 纯代码，配置文件和配置项由引用的工程提供
2. 代码+配置文件，配置项由引用的工程提供
3. 代码+配置文件+配置项

个人觉得第一种方案简单，但是无法自治，每个工程都需要配置不说了，关键是二方库无法单元测试，因为缺失配置文件和配置项。第三种方案会导致二方库反向依赖上层应用，并不合理。综合考虑，第二种方案相对比较折中，二方库可以包含dev和test环境的配置文件，其他环境的由应用提供。

二方库自治化还带来一个额外的好处，就是为服务化奠定了坚实的基础。因为服务化的前提就是自治化，能够运行起来对外提供服务。而自治化意味着可以独立运行，只要使用一个RPC框架（比如dubbo、thrift、g-rpc或者Restful）对外暴露服务就可以了。从这个概念上讲跟现在流行的[微服务架构(MSA)](http://dockone.io/article/947)是非常类似的。

更多微服务架构的介绍，可以参考如下文章：

1. [基于微服务的软件架构模式](http://dockone.io/article/877)
2. [Spring Boot与Docker（一）：微服务架构和容器化概述](http://dockone.io/article/879)
3. [实施微服务，我们需要哪些基础框架？](http://www.infoq.com/cn/articles/basis-frameworkto-implement-micro-service)


补记: Spring Boot come into rescue
----------------------------------

最近仔细将Spring Boot的官方文档看了一遍，发现Spring Boot就是为了解决这个问题的。通过Spring Boot构建的服务(executable jar or war)就是一个自包含的服务(`self-contained`)，就是可以跑起来提供服务的。不过他提倡的是微服务架构，也就是说这个服务不是通过jar包直接依赖，而是通过REST或者其他RPC方式对外提供服务。

1. Spring配置：推荐java-based configuration + package scanning
2. properties：支持properties文件和yaml

其实解决方案就是Spring Boot最核心的功能——自动配置，而自动配置的基础其实就是java-based配置。他很好的解决了以前bean配置和代码分离的状况。这样，一个应用依赖一个jar包的时候，就不需要再配置spring bean了。Spring会在启动的时候扫描标有`@configuration`注解的类，这些类其实就相当于一个个独立的XML beans配置。为了防止bean的重复注册或者错误配置，Spring4引入了`conditional configuration`，可以对bean进行条件配置。当然，property配置还是留给了应用。但是在java-based配置中，推荐使用`@EnableConfigurationProperties(XXXXProperties.class)`来讲properties属性配置注入到相应的类中。

补记: 如何单元测试一个模块@2016-03-06
------------------------------------

关于模块的单元测试问题我们在前面的文章中已经详细分析过。现在我们来看看解决方案。

我们知道要把一个单元测试跑起来，单单只有代码是不够的，还需要spring配置和相应的properties配置项。那么怎样提供Spring配置文件和Properties配置文件呢？

### 方案一

在二方库中配置Spring和相应的properties。但是我们不能简单的把这些配置文件放在src/main/resources目录，因为这样子会被打包到最终生成的jar包中。可能会
导致冲突，或者bean重复配置。这时候maven的src/test/resources就发挥作用了。放在这里面，单元测试的时候是可以正确加载到，但是打包的时候又会skip掉。


优点：传统的应用测试方式，但是又不需要在上层应用中测试

缺点：Spring bean配置不能被上层应用直接使用，每个依赖的应用都需要单独配置一次。还有一个状况就是二方库的spring配置可能会跟应用的配置不一样（比如dataSource和事务配置），这样可能是好事也可能是坏事。

具体实现如下：

pom.xml

	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
		xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">
		<modelVersion>4.0.0</modelVersion>
		<parent>
			<groupId>me.arganzheng.study.unittest</groupId>
			<artifactId>modules</artifactId>
			<version>1.0.0</version>
		</parent>

		<groupId>me.arganzheng.study.unittest.modules</groupId>
		<artifactId>user-module</artifactId>
		<packaging>jar</packaging>
		<name>${project.artifactId}</name>
		<description>The user biz module project</description>

		<properties>
			<skip_maven_deploy>false</skip_maven_deploy>
			<druid-version>1.0.14</druid-version>
		</properties>

		<dependencies>
			<!-- 二方库依赖 -->
			<dependency>
				<groupId>me.arganzheng.study.unittest.modules.common</groupId>
				<artifactId>common-utils</artifactId>
				<version>1.0.0</version>
			</dependency>

			<!-- 三方库依赖 -->
			<dependency>
				<groupId>mysql</groupId>
				<artifactId>mysql-connector-java</artifactId>
				<version>5.1.34</version>
			</dependency>

			<!-- Database -->
			<dependency>
				<groupId>com.alibaba</groupId>
				<artifactId>druid</artifactId>
				<version>${druid-version}</version>
			</dependency>
			<!-- mybatis-spring -->
			<dependency>
				<groupId>org.mybatis</groupId>
				<artifactId>mybatis-spring</artifactId>
				<version>1.2.4</version>
			</dependency>
			<dependency>
				<groupId>org.mybatis</groupId>
				<artifactId>mybatis</artifactId>
				<version>3.3.1</version>
			</dependency>

			<!-- Test scope dependencies -->
			<dependency>
				<groupId>junit</groupId>
				<artifactId>junit</artifactId>
				<version>4.8.2</version>
				<scope>test</scope>
			</dependency>
			<dependency>
				<groupId>org.springframework</groupId>
				<artifactId>spring-test</artifactId>
				<version>${spring.version}</version>
				<scope>test</scope>
			</dependency>
		</dependencies>
	</project>


src/test/resources/applicationContext.xml:

	<?xml version="1.0" encoding="UTF-8"?>
	<beans xmlns="http://www.springframework.org/schema/beans"
		xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:aop="http://www.springframework.org/schema/aop"
		xmlns:context="http://www.springframework.org/schema/context" xmlns:tx="http://www.springframework.org/schema/tx"
		xmlns:p="http://www.springframework.org/schema/p"
		xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-3.0.xsd
	          http://www.springframework.org/schema/aop http://www.springframework.org/schema/aop/spring-aop-3.0.xsd
	          http://www.springframework.org/schema/context http://www.springframework.org/schema/context/spring-context-3.0.xsd
	          http://www.springframework.org/schema/tx http://www.springframework.org/schema/tx/spring-tx-3.0.xsd"
		default-autowire="byName">

		<bean
			class="org.springframework.beans.factory.config.PropertyPlaceholderConfigurer">
			<property name="locations">
				<list>
					<value>classpath:application.properties</value>
				</list>
			</property>
		</bean>

		<context:component-scan base-package="me.arganzheng.study.unittestmodules.user" />

		<bean id="stat-filter" class="com.alibaba.druid.filter.stat.StatFilter">
			<property name="slowSqlMillis" value="${jdbc.slowSqlMillis}" />
			<property name="logSlowSql" value="${jdbc.logSlowSql}" />
			<property name="mergeSql" value="${jdbc.mergeSql}" />
		</bean>

		<bean id="log-filter" class="com.alibaba.druid.filter.logging.Log4jFilter">
			<!-- 输出可执行的SQL -->
			<property name="statementExecutableSqlLogEnable" value="${jdbc.statementExecutableSqlLogEnable}" />
			<property name="resultSetLogEnabled" value="false" />
		</bean>

		<bean id="dataSource" class="com.alibaba.druid.pool.DruidDataSource"
			destroy-method="close">
			<property name="url" value="${jdbc.url}" />
			<property name="username" value="${jdbc.username}" />
			<property name="password" value="${jdbc.password}" />

			<property name="initialSize" value="${jdbc.initialSize}" />
			<property name="minIdle" value="${jdbc.minIdle}" />  <!-- 可以和initialSize保持一致 -->
			<property name="maxActive" value="${jdbc.maxActive}" />

			<!-- 配置获取连接等待超时的时间 -->
			<property name="maxWait" value="${jdbc.maxWait}" />

			<!-- 配置removeAbandoned对性能会有一些影响，建议怀疑存在泄漏之后再打开。 -->
			<property name="removeAbandoned" value="true" /> <!-- 打开removeAbandoned功能 -->
			<property name="removeAbandonedTimeout" value="1800" /> <!-- 1800秒，也就是30分钟 -->
			<property name="logAbandoned" value="true" /> <!-- 关闭abanded连接时输出错误日志 -->

			<!-- 配置间隔多久才进行一次检测，检测需要关闭的空闲连接，单位是毫秒 -->
			<property name="timeBetweenEvictionRunsMillis" value="60000" />
			<!-- 配置一个连接在池中最小生存的时间，单位是毫秒 -->
			<property name="minEvictableIdleTimeMillis" value="300000" />

			<property name="validationQuery" value="SELECT @@SQL_MODE" />
			<property name="testWhileIdle" value="true" />
			<property name="testOnBorrow" value="false" />
			<property name="testOnReturn" value="false" />

			<!-- 配置监控统计拦截的filters -->
			<property name="proxyFilters">
				<list>
					<ref bean="stat-filter" />
					<ref bean="log-filter" />
				</list>
			</property>
			<!-- 配置timeBetweenLogStatsMillis>0之后，DruidDataSource会定期把监控数据输出到日志中。 -->
			<property name="timeBetweenLogStatsMillis" value="${jdbc.timeBetweenLogStatsMillis}" />
		</bean>

		<bean id="sqlSessionFactory" class="org.mybatis.spring.SqlSessionFactoryBean">
			<property name="configLocation" value="classpath:mybatis-config.xml" />
			<property name="dataSource" ref="dataSource" />
			<property name='mapperLocations' value='classpath*:mybatis-mapper/**/*.xml' />
			<property name="typeAliasesPackage" value="me.arganzheng.study.unittest.model;me.arganzheng.study.unittest.criteria" />
		</bean>

		<bean id="sqlSession" class="org.mybatis.spring.SqlSessionTemplate">
			<constructor-arg ref="sqlSessionFactory" />
		</bean>

	</beans>


src/test/resources/application.properties:


	jdbc.driverClassName=com.mysql.jdbc.Driver
	jdbc.url=jdbc:mysql://127.0.0.1:3306/mp_user?useUnicode=true&characterEncoding=utf8&zeroDateTimeBehavior=convertToNull
	jdbc.username=xxxx
	jdbc.password=xxxxx

	jdbc.minIdle=1 
	jdbc.initialSize=1 
	jdbc.maxActive=2

	jdbc.maxWait=30000

	jdbc.slowSqlMillis=1000
	jdbc.logSlowSql=true
	jdbc.mergeSql=true

	jdbc.statementExecutableSqlLogEnable=true

	jdbc.timeBetweenLogStatsMillis=600000


单元测试类：

	package me.arganzheng.study.unittest.modules.user.service.impl;

	import org.junit.Assert;
	import org.junit.Test;
	import org.junit.runner.RunWith;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.test.context.ContextConfiguration;
	import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

	import me.arganzheng.study.unittest.modules.user.domain.User;
	import me.arganzheng.study.unittest.modules.user.service.interfaces.UserService;

	@RunWith(SpringJUnit4ClassRunner.class)
	@ContextConfiguration({ "classpath:applicationContext.xml" })
	public class UserServiceImplTest {
	    @Autowired
	    private UserService userService;

	    @Test
	    public void testSelectAccountByName() {
	        User user = userService.getVistorUserInfo("arganzheng", "USA");
	        Assert.assertNotNull(user);
	    }
	}


### 方案二

可以看到使用传统的Spring配置方式，需要配置很多东东。这时候Spring Boot的优势就起作用了。可以利用Spring Boot的自动配置和starer pom，达到配置简化，甚至无配置的效果。

pom.xml

	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
		xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">
		<modelVersion>4.0.0</modelVersion>
		<parent>
			<groupId>me.arganzheng.study.unittest</groupId>
			<artifactId>modules</artifactId>
			<version>1.0.0</version>
		</parent>

		<groupId>me.arganzheng.study.unittest.modules</groupId>
		<artifactId>user-module</artifactId>
		<packaging>jar</packaging>
		<name>${project.artifactId}</name>
		<description>The user biz module project</description>

		<properties>
			<skip_maven_deploy>false</skip_maven_deploy>
		</properties>

		<dependencies>
			<!-- 二方库依赖 -->
			<dependency>
				<groupId>me.arganzheng.study.unittest.modules.common</groupId>
				<artifactId>common-utils</artifactId>
				<version>1.0.0</version>
			</dependency>

			<!-- 三方库依赖 -->
			<dependency>
				<groupId>org.mybatis.spring.boot</groupId>
				<artifactId>mybatis-spring-boot-starter</artifactId>
				<version>1.0.0</version>
			</dependency>

			<!--  
				注意：如果不显示的添加这个依赖，会报：
				Caused by: java.lang.ClassNotFoundException: org.springframework.context.event.GenericApplicationListener 
			  	错误。具体原因不明。。
			-->
			<dependency>
				<groupId>org.springframework</groupId>
				<artifactId>spring-context</artifactId>
				<version>4.2.5.RELEASE</version>
			</dependency>

			<dependency>
				<groupId>mysql</groupId>
				<artifactId>mysql-connector-java</artifactId>
				<version>5.1.34</version>
			</dependency>

			<dependency>
				<groupId>org.springframework.boot</groupId>
				<artifactId>spring-boot-starter-test</artifactId>
				<version>1.3.2.RELEASE</version>
				<scope>test</scope>
			</dependency>

		</dependencies>

	</project>


单元测试类：

	package me.arganzheng.study.unittest.modules.user.mapper;

	import org.junit.Assert;
	import org.junit.Test;
	import org.junit.runner.RunWith;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.boot.test.SpringApplicationConfiguration;
	import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
	import org.springframework.transaction.annotation.Transactional;

	import me.arganzheng.study.unittest.modules.user.ApplicationContext;
	import me.arganzheng.study.unittest.modules.user.domain.User;

	@RunWith(SpringJUnit4ClassRunner.class)
	@SpringApplicationConfiguration(classes = ApplicationContext.class)
	@Transactional
	public class UserMapperTest {

	    @Autowired
	    private UserMapper userMapper;

	    @Test
	    public void testSelectVistorUserByDeviceId() {
	        User user = userMapper.selectVistorUserByDeviceId("arganzheng");
	        Assert.assertNotNull(user);
	    }
	}

其中 ApplicationContext 类是java-based configuration类：

	package me.arganzheng.study.unittest.modules.user;

	import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
	import org.springframework.context.annotation.ComponentScan;
	import org.springframework.context.annotation.Configuration;

	@Configuration
	@EnableAutoConfiguration
	@ComponentScan("me.arganzheng.study.unittest.modules.user")
	public class ApplicationContext {

	}

因为使用了Spring Boot mybatis和DataSource，所以配置项有所不同：

src/test/resources/application.properties:

	mybatis.config=classpath:mybatis-config.xml

	spring.datasource.driver-class-name=com.mysql.jdbc.Driver
	spring.datasource.url=jdbc:mysql://127.0.0.1:3306/mp_user
	spring.datasource.username=xxx
	spring.datasource.password=xxxx

是不是整体简单很多？


但是后来发现在测试service的时候就出错了：


	package me.arganzheng.study.unittest.modules.user.service.impl;

	import org.junit.Assert;
	import org.junit.Test;
	import org.junit.runner.RunWith;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.boot.test.SpringApplicationConfiguration;
	import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
	import org.springframework.transaction.annotation.Transactional;

	import me.arganzheng.study.unittest.modules.user.ApplicationContext;
	import me.arganzheng.study.unittest.modules.user.domain.User;
	import me.arganzheng.study.unittest.modules.user.service.interfaces.UserService;

	@RunWith(SpringJUnit4ClassRunner.class)
	@SpringApplicationConfiguration(classes = ApplicationContext.class)
	@Transactional
	public class UserServiceImplTest {

	    @Autowired
	    private UserService userService;

	    @Test
	    public void testSelectAccountByName() {
	        User user = userService.getVistorUserInfo("arganzheng", "USA");
	        Assert.assertNotNull(user);
	    }
	}

会报如下异常：

	org.apache.ibatis.binding.BindingException: Invalid bound statement (not found): com.baidu.mobojoy.mobopay.modules.user.service.interfaces.UserService.getVistorUserInfo
		at org.apache.ibatis.binding.MapperMethod$SqlCommand.<init>(MapperMethod.java:196)
		at org.apache.ibatis.binding.MapperMethod.<init>(MapperMethod.java:44)
		at org.apache.ibatis.binding.MapperProxy.cachedMapperMethod(MapperProxy.java:59)
		at org.apache.ibatis.binding.MapperProxy.invoke(MapperProxy.java:52)
		at com.sun.proxy.$Proxy46.getVistorUserInfo(Unknown Source)
		at com.baidu.mobojoy.mobopay.modules.user.service.impl.UserServiceImplTest.testSelectAccountByName(UserServiceImplTest.java:24)
		at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)

把Autowired改成UserServiceImpl就没有问题：

	@Autowired
    private UserServiceImpl userService;

显然，myBatis错误的为service接口生成代理了。原因就是在于mybatis-spring-boot的自动配置类中，做了一个默认配置项。[MybatisAutoConfiguration](
https://github.com/mybatis/mybatis-spring-boot/blob/master/mybatis-spring-boot-autoconfigure/src/main/java/org/mybatis/spring/boot/autoconfigure/MybatisAutoConfiguration.java):

	/**
	 * This will just scan the same base package as Spring Boot does. If you want more
	 * power, you can explicitly use {@link org.mybatis.spring.annotation.MapperScan} but
	 * this will get typed mappers working correctly, out-of-the-box, similar to using
	 * Spring Data JPA repositories.
	 */
	public static class AutoConfiguredMapperScannerRegistrar implements BeanFactoryAware,
			ImportBeanDefinitionRegistrar, ResourceLoaderAware {
		...
	}

如果没有配置@MapperScan，那么默认会使用跟Spring Boot一样的base package。而我们Spring boot配置的scan范围是根package，所以service自然而然也被当作mapper扫描进来了。
但是神奇的是加上 @MapperScan 之后还是一样报错。。晕死。。所以说自动配置很容易出问题，还是手动配置靠谱啊。。



### 方案三

前面两种方案（本质上是一种方案），主要的缺点就是上层应用需要重新配置一遍Spring。而我们知道绝大多数情况下，上层应用不需要对二方库进行特殊的bean配置。所以为了避免每个依赖的上层应用都配置一次spring，是不是可以直接复用二方库中的spring配置呢？

答案当然是可以的，最简单的方案，就是将二方库中的Spring配置放在src/main/resources下，或者使用java-based Configuration放在java类中。这样二方库Spring配置就会被打包进入jar包，也就是在应用的classpath中，上层应用就可以直接指定二方库的配置文件（通过include）或者配置类（通过@Import注解）了。

但是这样会带来两个问题：

1、上层应用需要知道二方库的Spring配置文件路径和名称，或者配置类的名称。

	<import resource="classpath*:spring-conf/user-module-applicationContext.xml"/>
	@Import({ UserModuleConfiguration.class })

2、多个二方库可能对同一个bean重复配置，比如data source。

第一个问题的解决方案，就是可以使用约定高于配置的方式，比如所有的二方库的配置文件都以xxx-applicationContext.xml命名，那么在import的时候就可以使用通配符一次性导入：

		<import resource="classpath*:spring-conf/*-applicationContext.xml"/>

而对于配置类的方式，则可以通过包扫描的方式，扫描整个classpath。Spring Boot的自动扫描就是如此。

第二个问题相对不大好解决，原来阿里的做法是容器的隔离和共享机制，不过会带来很大的复杂性。最新版的spring 4引入的`conditional configuration`提供了另一种解决思路。它可以让你在配置bean的时候先检查某些条件是否满足，如果不满足就跳过该bean的装配。具体参见: [Creating your own auto-configuration](http://docs.spring.io/spring-boot/docs/current/reference/html/boot-features-developing-auto-configuration.html)。

但是条件装配对bean的加载顺序有要求，具体参见: [Why We Do Not Use Spring Boot Auto Configuration](http://dev-blog.xoom.com/2015/03/15/use-spring-boot-auto-configuration/)。所以是否可行，还是要测试一下。

总之，我们希望达到如下目的：代码和Spring配置在二方库中，同时包含UT的配置项可以跑单元测试。上层应用只需要依赖二方库，然后配置相应环境的配置项就可以了。

另外，关于二方库和应用配置文件的隔离，除了上面提到的src/test/resources目录，还可以考虑统一使用配置中心

---

这是一个大话题，未完待续。。

