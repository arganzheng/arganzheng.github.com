---
title: Spring Boot学习
layout: post
catalog: true
---


Spring Boot简化了传统的Spring应用开发，主要四个特性：

1. 自动配置：进一步简化XML配置
2. Starter依赖：简化pom.xml依赖配置
3. The Actuator: Gives you insight into what’s going on inside of a running Spring Boot application.
4. 命令行接口：This optional feature of Spring Boot lets you write complete applications with just application code, but no need for a traditional project build.


 conditional configuration features from Spring 4, along with transitive dependency resolution offered by Maven and Gradle, to automatically configure beans in the Spring application context.


 In a nutshell, Spring Boot auto-configuration is a runtime (more accurately, applica- tion startup-time) process that considers several factors to decide what Spring configu- ration should and should not be applied. 


 When you add Spring Boot to your application, there’s a JAR file named spring- boot-autoconfigure that contains several configuration classes. Every one of these configuration classes is available on the application’s classpath and has the opportunity to contribute to the configuration of your application. There’s configuration for Thyme- leaf, configuration for Spring Data JPA, configuration for Spring MVC, and configura- tion for dozens of other things you might or might not want to take advantage of in your Spring application.
What makes all of this configuration special, however, is that it leverages Spring’s support for conditional configuration, which was introduced in Spring 4.0. Conditional configuration allows for configuration to be available in an application, but to be ignored unless certain conditions are met.


### 如何覆盖自动配置？


Although Spring Boot’s auto-configuration and @ConditionalOnMissingBean make it possible for you to explicitly override any of the beans that would otherwise be auto- configured, it’s not always necessary to go to that extreme. Let’s see how you can set a few simple configuration properties to tweak the auto-configured components.


properties配置：

SpringBoot默认加载的配置文件是application.properties。
你可以加载其他的配置文件


@Configuration
@PropertySource("classpath:/com/soundsystem/app.properties")
 public class ExpressiveConfig {
  @Autowired
  Environment env;
@Bean
Declare a property source


When relying on component-scanning and autowiring to create and initialize your application components, there’s no configuration file or class where you can specify the placeholders. Instead, you can use the @Value annotation in much the same way as you might use the @Autowired annotation. In the BlankDisc class, for example, the constructor might be written like this:

public BlankDisc(
      @Value("${disc.title}") String title,
      @Value("${disc.artist}") String artist) {
  this.title = title;
  this.artist = artist;
}

In order to use placeholder values, you must configure either a PropertyPlaceholder- Configurer bean or a PropertySourcesPlaceholderConfigurer bean. Starting with Spring 3.1, PropertySourcesPlaceholderConfigurer is preferred because it resolves placeholders against the Spring Environment and its set of property sources.
The following @Bean method configures PropertySourcesPlaceholderConfigurer in Java configuration:

@Bean
public
static PropertySourcesPlaceholderConfigurer placeholderConfigurer() {
  return new PropertySourcesPlaceholderConfigurer();
}


SpringApplicationContextLoader



2. The @SpringApplicationConfiguration annotation is similar to the @ContextConfiguration annotation in that it is used to specify which application context(s) that should be used in the test. Additionally, it will trigger logic for reading Spring Boot specific configurations, properties, and so on.

相当于 @ContextConfiguration(classes={TestConfiguration.class}, loader = SpringApplicationContextLoader.class)

如果测试 Spring MVC工程，那么必须指定 @WebAppConfiguration 注解。

3. @WebAppConfiguration must be present in order to tell Spring that a WebApplicationContext should be loaded for the test. It also provides an attribute for specifying the path to the root of the web application.


排除默写配置类：

@EnableAutoConfiguration(exclude = { SecurityAutoConfiguration.class, ManagementSecurityAutoConfiguration.class })


### 配置类 [Configuration classes](http://docs.spring.io/spring-boot/docs/current-SNAPSHOT/reference/htmlsingle/#using-boot-configuration-classes)


Spring Boot倾向于java-based配置：

	@Configuration
	@EnableAutoConfiguration
	@ComponentScan(basePackages = "me.arganzheng.stduy.springboot")
	public class SpringBootApplication {

	    public static void main(String[] args) {
	        SpringApplication.run(SpringBootApplication.class, args);
	    }

	}

但是我们也可以使用让`SpringApplication.run()`接受XML配置文件。

**TIPS**

可以使用`@SpringBootApplication`注解，它相当于同时注解了`@Configuration @EnableAutoConfiguration @ComponentScan`

**配置导入**

1. 使用`@Import`注解导入其他的配置类。你也可以使用`@ComponentScan`自动扫描所有的Spring组件，包括`@Configuration`配置类。
2. 你也可以导入XML配置，你可以使用`@ImportResource`注解加载其他XML配置文件。



### [自动配置](http://docs.spring.io/spring-boot/docs/current-SNAPSHOT/reference/htmlsingle/#using-boot-auto-configuration)

**TIPS**

1、可以通过`--debug`启动开关，讲自动配置信息打印到控制台。

2、排除某个自动配置类

	@Configuration
	@EnableAutoConfiguration(exclude={DataSourceAutoConfiguration.class})
	public class MyConfiguration {
	}


### [自动配置原理](http://docs.spring.io/spring-boot/docs/current-SNAPSHOT/reference/htmlsingle/#boot-features-developing-auto-configuration)

#### 1、定义配置类

这个前面已经提到过，Spring Boot一般使用java-based配置，也就是以`Configuration`注解的类。

为了避免重复配置，Spring 4提供了条件配置注解，可以根据需要配置。最常用的是`@ConditionalOnClass`和`@ConditionalOnMissingBean`注解。

#### 2、加载自动配置类

Spring Boot在启动的时候会检查jar包中的`META-INF/spring.factories`文件，这个文件中会配置所有标注了`@EnableAutoConfiguration`注解的配置类：




### 创建自己的自动配置类

If you work in a company that develops shared libraries, or if you work on an open-source or commercial library, you might want to develop your own auto-configuration. Auto-configuration classes can be bundled in external jars and still be picked-up by Spring Boot.


如何启动Spring Boot程序
---------------------

有三种方式：

1、从IDE启动 

2、jar包方式启动
 	
 	java -jar target/myproject-0.0.1-SNAPSHOT.jar

也可以指定调试模式：

	java -Xdebug -Xrunjdwp:server=y,transport=dt_socket,address=8000,suspend=n -jar target/myproject-0.0.1-SNAPSHOT.jar

3、使用maven插件启动

	mvn spring-boot:run

同样可以调试：

	set MAVEN_OPTS=-Xrunjdwp:transport=dt_socket,server=y,suspend=n,address=8000
	# export MAVEN_OPTS=-Xmx1024m -XX:MaxPermSize=128M 
	mvn spring-boot:run

**TIPS**

如果你想查看Spring Boot自动配置了什么信息，可以使用`--debug`启动项。		


推荐阅读
-------

1. [Spring Boot Reference Guide：41. Creating your own auto-configuration](http://docs.spring.io/spring-boot/docs/current-SNAPSHOT/reference/htmlsingle/#boot-features-developing-auto-configuration) 关于Spring Boot自动配置原理的官方介绍






