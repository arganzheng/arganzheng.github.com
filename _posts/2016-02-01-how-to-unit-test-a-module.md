---
title: 如何单元测试二方库
layout: post
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
