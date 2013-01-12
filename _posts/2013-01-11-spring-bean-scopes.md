---
layout: post
title: Spring的Bean Scopes
---


Spring一共支持5种bean scopes，其中3种需要web-aware ApplicationContext。[Bean Scopes](http://static.springsource.org/spring/docs/3.0.0.M3/reference/html/ch04s04.html)

> out of the box, the Spring Framework supports exactly five scopes (of which three are available only if you are using a web-aware ApplicationContext). 
> 
> 1. singleton
Scopes a single bean definition to a single object instance per Spring IoC container.
>
2. prototype
Scopes a single bean definition to any number of object instances.
>
3. request
Scopes a single bean definition to the lifecycle of a single HTTP request; that is each and every HTTP request will have its own instance of a bean created off the back of a single bean definition. Only valid in the context of a web-aware Spring ApplicationContext.
>
4. session
Scopes a single bean definition to the lifecycle of a HTTP Session. Only valid in the context of a web-aware Spring ApplicationContext.
>
5. global session
Scopes a single bean definition to the lifecycle of a global HTTP Session. Typically only valid when used in a portlet context. Only valid in the context of a web-aware Spring ApplicationContext.


singleton是sping的default bean scope，即对于一个特定的bean定义，每个Spring Container只会返回同一个bean实例。这意味着所有的bean是共享的，特别适合于无状态的bean定义。这也反证了为什么rich domain model模式在Java中不流行。

> As a rule of thumb, you should use the prototype scope for all beans that are stateful, while the singleton scope should be used for stateless beans.

prototype跟singleton刚好相反，每次都返回新的实例。从某种意义上类似于使用java的new操作符。原则上用于有状态的bean定义。Struts2的Action就是使用了这个模式避免线程安全问题。但是每次都创建一个，内存成本是比较高的，所以要慎用。

需要注意的是，Spring对于prototype scope的bean，不会对其做资源释放操作。需要用户自己处理。从这个意义上来看，确实很像java的new操作。

>There is one quite important thing to be aware of when deploying a bean in the prototype scope, in that the lifecycle of the bean changes slightly. Spring does not manage the complete lifecycle of a prototype bean: the container instantiates, configures, decorates and otherwise assembles a prototype object, hands it to the client and then has no further knowledge of that prototype instance. This means that while initialization lifecycle callback methods will be called on all objects regardless of scope, in the case of prototypes, any configured destruction lifecycle callbacks will not be called. It is the responsibility of the client code to clean up prototype scoped objects and release any expensive resources that the prototype bean(s) are holding onto. (One possible way to get the Spring container to release resources used by prototype-scoped beans is through the use of a custom bean post-processor which would hold a reference to the beans that need to be cleaned up.)

这两个模式是我们都比较熟悉的模式，在任何情况下都可以使用。关于两者的区别可以参考一下这篇文章：[Spring Bean Scopes Example](http://www.mkyong.com/spring/spring-bean-scopes-examples/)

## request、session和global session scope bean(web-scoped beans)

在web-based applications环境下，spring还支持request、session和global session三种scope。

>The scopes that are described in the following paragraphs are only available if you are using a web-aware Spring ApplicationContext implementation (such as XmlWebApplicationContext). If you try using these next scopes with regular Spring IoC containers such as the XmlBeanFactory or ClassPathXmlApplicationContext, you will get an IllegalStateException complaining about an unknown bean scope.

...

>#### org.springframework.web.context.WebApplicationContext
>
Interface to provide configuration for a web application. This is read-only while the application is running, but may be reloaded if the implementation supports this. 
>
This interface adds a getServletContext() method to the generic ApplicationContext interface, and defines a well-known application attribute name that the root context must be bound to in the bootstrap process. 
>
Like generic application contexts, web application contexts are hierarchical. There is a single root context per application, while each servlet in the application (including a dispatcher servlet in the MVC framework) has its own child context. 
>
In addition to standard application context lifecycle capabilities, WebApplicationContext implementations need to detect ServletContextAware beans and invoke the setServletContext method accordingly.

#### 1. Initial web configuration

在定义web-scoped beans（即request、session和global session scope bean）之前，需要先配置一下web.xml。有如下三种方式 [3.5.4 Request, session, and global session scopes](http://static.springsource.org/spring/docs/3.0.x/spring-framework-reference/html/beans.html#beans-factory-scopes-other)

1. `DispatcherServlet` for Spring MVC
2. `RequestContextListener` for Servlet 2.4+ web container
3.  `RequestContextFilter` for older web container (Servlet 2.3)

具体例子可以参考 [Injecting request scope beans into singletons with RESTeasy and Spring](http://www.tikalk.com/java/forums/injecting-request-scope-beans-singletons-with-resteasy-and-spring)

#### 2. request scope

最明显的例子就是stucts2中的Action，每次请求都会重新创建一个。每个请求之间互补影响。

这里举个例子。比如我们希望在我们Controller中能够自动注入用户的信息。

    @Controller
    public class FooController {
    
        @Autowired
        private FooService                 fooService;
    
        @Autowired
        private FooUser                      user;
           
        ...
    }
    
FooUser的信息来自于Cookies，而且每次请求都可能不同，怎么做到呢？

以往的做法，我们会写个Intercepter，由于是Request级别，所以每次new一个FooUser，然后将用户信息从cookies中抽取出来，用来初始化FooUser对象。最后将FooUser对象存放在ThreadLocal中或者request的attribute中（即`request.setAttribute("user", fooUser);`；如果是session scope的，则一般放在session的attribute中）。然后在Controller中，通过request.getAttribute("fooUser")拿出来。

然而Spring帮你简化了这一过程。而且FooUser可以像其他bean一样互相注入（注意：这个很重要，因为request scope的bean和singleton scope的bean的生命周期是不一样的！a bean with shorter life-cycle can be injected into a bean with a longer life-cycle. ）。不需要知道它是从request中来还是ThreadLocal中来的。只需要做如下的配置：

    <bean class="me.arganzheng.study.scope.FooUser" scope="request">
		<aop:scoped-proxy />
	</bean>

`scope="request"`告诉Spring每次web请求都实例化一个新的FooUser，并且FooUser只能在该web request的上下文中获取到。

`<aop:scoped-proxy />`配置项告诉Spring不要直接注入FooUser到依赖它的bean，而是注入FooUser的一个AOP代理。这样singleton周期的bean才能注入request scope的bean。这里默认是使用GBLIB生成代理对象，如果你的request scope bean有接口，那么也可以通过`proxy-target-class="false"`配置项告诉Spring使用JDK动态代理。

设置bean的scope为`request`，并且配置`<aop:scoped-proxy />`，FooUser就可以像普通的bean一样被其他bean依赖了。

    public class FooService{

        @Autowired
        FooUser fooUser;
        
        ...
        
    }

### web scope beans的初始化

与core scope beans一样，你可以在配置的时候告诉Spring初始化参数或者属性或者相关依赖，Spring会帮你自动注入。但是Spring还没有智能到自动将request、session或者cookies中的值拿来装配你的bean。（就像Spring MVC的`@RequestParam`一样），虽然这样也挺nice。所以只能自己处理了。

比如上面的fooUser，它的值是来自于cookies，那么可以写个拦截器对它进行初始化，然后放回request的attrbute中：

    package me.arganzheng.study.beanscope.interceptor;
	
	import javax.servlet.http.HttpServletRequest;
	import javax.servlet.http.HttpServletResponse;
	
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.web.servlet.handler.HandlerInterceptorAdapter;
	
	import me.arganzheng.study.beanscope.domain.TSpInfo;
	import me.arganzheng.study.beanscope.dao.TSpInfoDAO;
	import me.arganzheng.study.util.cookie.CookieHelper;
	
	/**
	 * 
	 * 用户身份验证的interceptor，注意这里不管权限控制。
	 * 
	 * @author arganzheng
	 */
	public class AuthenticationInterceptor extends HandlerInterceptorAdapter {
		@Autowired
		private FooUser user;
		@Autowired
		private UserClient client;
		@Autowired
		private TSpInfoDAO spInfoDao;
	
		@Override
		public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
			user.setUinString(CookieHelper.getCookieValue(request, "uin"));
			user.setSkey(CookieHelper.getCookieValue(request, "skey"));
	
			// 客户端IP
			user.setClientIp(request.getHeader("X-Real-IP"));
			if(user.getClientIp() == null){
				user.setClientIp(request.getRemoteAddr());
			}
			if(user.getClientIp() == null){
				user.setClientIp("127.0.0.1");
			}
	
			// 初始化登录状态
			user.setLoggedIn(false);
			user.setLevel(-1);
			user.setSpInfo(null);
	
			if(!user.isAnonymous()){
				int level = client.checkUserLevel(user.getUin(), user.getSkey());
				user.setLevel(level);
	
				if(level >= 0){
					user.setLoggedIn(true);
	
					request.setAttribute("user", user);
	
					TSpInfo spInfo = spInfoDao.selectByPrimaryKey(user.getUin());
					user.setSpInfo(spInfo);
				}
			}
	
			return true;
		}
	}

当然，也可以使用factory-bean做初始化：具体可以参考这篇文章：[Spring: Request Scope and Injecting Current User](http://tedyoung.me/2011/04/21/spring-request-scope/)。事实上，上面的interceptor不就相当于FooUser的factory-bean吗:)


#### 3. session scope

一般用于`user profile`或者`shopping cart`。最好的例子来自于《Spring in Action》作者这篇文章: [Session-scoped beans in Spring](http://springinpractice.com/2008/05/08/session-scoped-beans-in-spring/)。

#### 4. global session scope

比较少见，这里不举例了。

#### 5. 自定义bean scope

具体可以参考 [3.9 Scope](http://springindepth.com/book/in-depth-ioc-scope.html)
和 [3.5.4.5 Scoped beans as dependencies](http://static.springsource.org/spring/docs/3.0.x/reference/beans.html#beans-factory-scopes-other-injection)


## 参考文档

1. 官方文档其实比任何书籍网站写的都详细和准确，一定要好好阅读。[3. The IoC container](http://static.springsource.org/spring/docs/3.0.x/reference/beans.html)
2. [Spring Bean Scopes](http://javapapers.com/spring/spring-bean-scopes/)
3. [Scoped Bean Dependencies](https://github.com/JamesEarlDouglas/barebones-spring-mvc/tree/master/reference/scoped-beans) 有详细的说明和工程代码，推荐阅读。里面有很多文章都不错。
4. [Spring: Request Scope and Injecting Current User](http://tedyoung.me/2011/04/21/spring-request-scope/)，不错的博客。


