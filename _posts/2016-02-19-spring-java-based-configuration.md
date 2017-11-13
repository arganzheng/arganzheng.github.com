---
title: Spring Java-based配置
layout: post
catalog: true
---

Spring目前为止支持如下三种配置方式：

1. XML-based configuration: Spring 1.0
2. Annotation-based configuration (with component scanning): Spring 2.0
3. Java-based configuration Spring 3.0

这里探讨一下第三种方式：Java-based configuation。


WHY
---

首先我们思考一下为什么Spring要推出这种方式。从XML配置到注解配置对生产效率的提升是显而易见的，我相信每一个使用了注解配置的java开发人员都不愿意回到原始的XML配置方式了。那么Spring3.0引入的基于java的配置方式又给我们带来什么样的惊喜呢？

使用XML最大的问题在我看来就是使用方需要自己去配置错综复杂的组装逻辑（依赖关系），类库定义方(提供方)确实是使用Spring无关（或者说框架无关）的方式定义和实现了POJO。但是使用方如果要使用Spring托管的依赖注入功能，就必须配置一个复杂的XML文件，否则就自己用java代码组装(通过set方法或者构造函数)。

所以，为了减少使用方庞大的bean配置工作，Spring2.0引入了基于注解的方式，做了一个折中：类库提供方来定义bean和依赖(通过Component和Autowired等注解)，然后服务使用方只需要简单的配置ComponentScan的路径就可以将这些Bean组织起来了。

这当然大大的简化了使用方的工作，但是也把原来XML配置的好处抛弃了。而且留下有一个尴尬的问题：这种方式，提供方和使用方都需要依赖于Spring框架，而且都需要做一定的工作让其work起来。

而且这两种方式还有一个问题，就是类库在单元测试的时候很不方便。因为仍然有组织的逻辑需要处理，也就是说类库提供方并不能避免通过XML组织bean逻辑的操作，因为单元测试要用到这些bean，但是这个配置又不能直接给真正的业务使用方使用。

这就是Spring3.0引入Java-based Configuration的原因了。让Bean的定义和依赖关系，以及最后的组装生产配置都封装在类库中。这样使用方只需要简单的引用jar包就可以直接使用了，不需要配置XML文件了（即使是简单的ComponentScan）。

需要注意的是，Java-based Configuration并没有替代Annotation-based Configuration，但是确实引入了一个新的注解——`@Bean`注解，来表示一个工厂方法返回的bean将被Spring托管。Java-based Configuration更多的完全抛弃XML配置，毕竟XML不方便打包和加载，而且XML配置没有工厂方法那么灵活，例如在运行时期根据一些信息返回不同的bean实例（策略模式）。

总而言之，XML配置的缺点（编译期间类型校验，类定义跳转）都变成java-based Configuration的优点了，哈哈。


**TIPS** 

当然使用XML配置有个好处，就是类库中的java类定义并不与Spring容器耦合。而使用基于注解的方式，Bean的定义和使用（注入）都依赖于Spring，It makes sense if you build up your own architecture relying on Spring’s component scanning / autowiring injection style。

另一个相对的好处就是java配置方式虽然有强类型检查，但是并不利于线上修改，每次修改都要编译发布，而XML是文本配置文件，直接修改重启就可以了，这个特性在单体应用流行的时候其实是非常重要的一个优势的。

servlet 3之后，web.xml中的配置也可以通过java类来配置了：[WebApplicationInitializer](http://docs.spring.io/spring/docs/3.1.x/javadoc-api/org/springframework/web/WebApplicationInitializer.html)


HOW
---


具体配置其实蛮简单的。关键就两个注解：

* @Component：Annotating a class with the @Configuration indicates that the class can be used by the Spring IoC container as a source of bean definitions. 简单来说就是一个 @Component注解类 相对于一个XML配置文件。
* @Bean: The @Bean annotation tells Spring that a method annotated with @Bean will return an object that should be registered as a bean in the Spring application context. 简单来说就是一个 @Bean注解方法 相对于XML配置文件中的 `<bean/>` 元素。 
	* name
	* init-method
	* destroy-method
	* autowiring 

举个简单的例子：

```java
package life.arganzheng.study.spring;
import org.springframework.context.annotation.*;

@Configuration
public class HelloJavaBasedConfig {

   @Bean 
   public Foo foo(){
      return new foo();
   }

}
```

对应如下XML配置：

```xml
<beans>
   <bean id = "foo" class = "life.arganzheng.study.spring.Foo" />
</beans>
```

加载配置类的方式跟加载XML也是类似的，XML使用的是`ClassPathXmlApplicationContext`，这里使用的是`AnnotationConfigApplicationContext`:

```java
public static void main(String[] args) {
   ApplicationContext ctx = new AnnotationConfigApplicationContext(HelloJavaBasedConfig.class);
   
   Foo foo = ctx.getBean(Foo.class);
   ...
}
```

配置类也可以互相import：

```java
@Configuration
public class ConfigA {
  public @Bean A a() { return new A(); }
}

@Configuration
@Import(ConfigA.class)
public class ConfigB {
  public @Bean B b() { return new B(); }
}
```

### 依赖注入

配置类跟其他的注解类没有区别，所以可以彼此通过`Autowired`互相注入。如：

```java
@Configuration
public class ServiceConfig {
  private @Autowired AccountRepository accountRepository;

  public @Bean TransferService transferService() {
      return new TransferServiceImpl(accountRepository);
  }
}

@Configuration
public class RepositoryConfig {
  private @Autowired DataSource dataSource;

  public @Bean AccountRepository accountRepository() {
      return new JdbcAccountRepository(dataSource);
  }
}

@Configuration
@Import({ServiceConfig.class, RepositoryConfig.class})
public class SystemTestConfig {
  public @Bean DataSource dataSource() { /* return new DataSource */ }
}

public static void main(String[] args) {
  ApplicationContext ctx = new AnnotationConfigApplicationContext(SystemTestConfig.class);
  // everything wires up across configuration classes...
  TransferService transferService = ctx.getBean(TransferService.class);
  transferService.transfer(100.00, "A123", "C456");
}
```

也可以引用另一个配置类：

```java
@Configuration
public class ServiceConfig {
  private @Autowired RepositoryConfig repositoryConfig;

  public @Bean TransferService transferService() {
      return new TransferServiceImpl(repositoryConfig.accountRepository());
  }
}

@Configuration
public interface RepositoryConfig {
  @Bean AccountRepository accountRepository();
}

@Configuration
public class DefaultRepositoryConfig implements RepositoryConfig {
  public @Bean AccountRepository accountRepository() {
      return new JdbcAccountRepository(...);
  }
}

@Configuration
@Import({ServiceConfig.class, DefaultRepositoryConfig.class}) // import the concrete config!
public class SystemTestConfig {
  public @Bean DataSource dataSource() { /* return DataSource */ }
}

public static void main(String[] args) {
  ApplicationContext ctx = new AnnotationConfigApplicationContext(SystemTestConfig.class);
  TransferService transferService = ctx.getBean(TransferService.class);
  transferService.transfer(100.00, "A123", "C456");
}
```

### Combining Java and XML configuration

如果想在一个XML配置为主的项目中使用java-based Configuration，只需要将这个配置类在XML文件中配置就可以了：

```xml
<beans>
  <!-- enable processing of annotations such as @Autowired and @Configuration -->
  <context:annotation-config/>
  <context:property-placeholder location="classpath:/com/acme/jdbc.properties"/>

  <bean class="com.acme.AppConfig"/>

  <bean class="org.springframework.jdbc.datasource.DriverManagerDataSource">
      <property name="url" value="${jdbc.url}"/>
      <property name="username" value="${jdbc.username}"/>
      <property name="password" value="${jdbc.password}"/>
  </bean>
</beans>
```

```java
public static void main(String[] args) {
  ApplicationContext ctx = new ClassPathXmlApplicationContext("classpath:/com/acme/system-test-config.xml");
  TransferService transferService = ctx.getBean(TransferService.class);
  // ...
}
```

因为`@Configuration`跟`@Component`、`@Service`这些注解是一样的，所以你也可以使用`context:component-scan/>`，自动扫描配置类。


反过来，如果你想在一个java配置为主的项目中使用XML配置，那也是支持的。通过`@ImportResource`注解可以导入xml配置：

```
@Configuration
@ImportResource("classpath:/com/acme/properties-config.xml")
public class AppConfig {
  private @Value("${jdbc.url}") String url;
  private @Value("${jdbc.username}") String username;
  private @Value("${jdbc.password}") String password;

  public @Bean DataSource dataSource() {
      return new DriverManagerDataSource(url, username, password);
  }
}
```

properties-config.xml

```xml
<beans>
  <context:property-placeholder location="classpath:/com/acme/jdbc.properties"/>
</beans>
```

jdbc.properties

``` 
jdbc.url=jdbc:hsqldb:hsql://localhost/xdb
jdbc.username=sa
jdbc.password=
```

Main.java

```java
public static void main(String[] args) {
  ApplicationContext ctx = new AnnotationConfigApplicationContext(AppConfig.class);
  TransferService transferService = ctx.getBean(TransferService.class);
  // ...
}
```

### 其他的一些注解

* @Scope



参考文章
-------

1. [Consider Replacing Spring XML Configuration with JavaConfig](https://dzone.com/articles/consider-replacing-spring-xml)
2. [Spring App Migration: From XML to Java-based Config](http://www.robinhowlett.com/blog/2013/02/13/spring-app-migration-from-xml-to-java-based-config/) 非常详细的迁移文章，而且使用了component scanning。
3. [Spring 4 MVC HelloWorld Tutorial – Annotation/JavaConfig Example](http://websystique.com/springmvc/spring-4-mvc-helloworld-tutorial-annotation-javaconfig-full-example/) Spring MVC配置示例
4. [Spring bean management using Java configuration](http://www.ibm.com/developerworks/library/ws-springjava/)
5. [Spring Dependency Injection Styles – Why I love Java based configuration](https://blog.codecentric.de/en/2012/07/spring-dependency-injection-styles-why-i-love-java-based-configuration/)