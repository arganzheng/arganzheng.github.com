---
layout: post
title: Spring的Bean Scopes实现机制源码剖析
---


关于Spring的bean scopes，在我的上一篇文章 [Spring的Bean Scopes](http://blog.arganzheng.me/posts/spring-bean-scopes.html) 已经做了详细的介绍，这里我们探讨一下Spring的bean scopes底层是怎么实现的。

通过下面这行简单的配置：

    <bean class="me.arganzheng.study.scope.FooUser" scope="request">
		<aop:scoped-proxy />
	</bean>

为什么我们就可以让FooUser每次请求都生成新实例，而且可以注入到生命周期比它长的singleton bean中呢？

    public class FooService{

        @Autowired
        FooUser fooUser;
        
        ...
    }

我们以top-down的方式来探讨一下。

### 1. What happen when config `scope="request"`

这个选项会影响到spring的创建bean的方式。Spring的bean创建是在`AbstractBeanFactory`的`doGetBean`方法：

	// ..ignore imports and comments..
	public abstract class AbstractBeanFactory extends FactoryBeanRegistrySupport implements ConfigurableBeanFactory {
       
       ...

		//---------------------------------------------------------------------
		// Implementation of BeanFactory interface
		//---------------------------------------------------------------------

		...

		/**
		 * Return an instance, which may be shared or independent, of the specified bean.
		 * @param name the name of the bean to retrieve
		 * @param requiredType the required type of the bean to retrieve
		 * @param args arguments to use if creating a prototype using explicit arguments to a
		 * static factory method. It is invalid to use a non-null args value in any other case.
		 * @param typeCheckOnly whether the instance is obtained for a type check,
		 * not for actual use
		 * @return an instance of the bean
		 * @throws BeansException if the bean could not be created
		 */
		@SuppressWarnings("unchecked")
		protected <T> T doGetBean(
				final String name, final Class<T> requiredType, final Object[] args, boolean typeCheckOnly)
				throws BeansException {

			final String beanName = transformedBeanName(name);
			Object bean;

			// Eagerly check singleton cache for manually registered singletons.
			Object sharedInstance = getSingleton(beanName);
			if (sharedInstance != null && args == null) {
				if (logger.isDebugEnabled()) {
					if (isSingletonCurrentlyInCreation(beanName)) {
						logger.debug("Returning eagerly cached instance of singleton bean '" + beanName +
								"' that is not fully initialized yet - a consequence of a circular reference");
					}
					else {
						logger.debug("Returning cached instance of singleton bean '" + beanName + "'");
					}
				}
				bean = getObjectForBeanInstance(sharedInstance, name, beanName, null);
			}

			else {
				// Fail if we're already creating this bean instance:
				// We're assumably within a circular reference.
				if (isPrototypeCurrentlyInCreation(beanName)) {
					throw new BeanCurrentlyInCreationException(beanName);
				}

				// Check if bean definition exists in this factory.
				BeanFactory parentBeanFactory = getParentBeanFactory();
				if (parentBeanFactory != null && !containsBeanDefinition(beanName)) {
					// Not found -> check parent.
					String nameToLookup = originalBeanName(name);
					if (args != null) {
						// Delegation to parent with explicit args.
						return (T) parentBeanFactory.getBean(nameToLookup, args);
					}
					else {
						// No args -> delegate to standard getBean method.
						return parentBeanFactory.getBean(nameToLookup, requiredType);
					}
				}

				if (!typeCheckOnly) {
					markBeanAsCreated(beanName);
				}

				final RootBeanDefinition mbd = getMergedLocalBeanDefinition(beanName);
				checkMergedBeanDefinition(mbd, beanName, args);

				// Guarantee initialization of beans that the current bean depends on.
				String[] dependsOn = mbd.getDependsOn();
				if (dependsOn != null) {
					for (String dependsOnBean : dependsOn) {
						getBean(dependsOnBean);
						registerDependentBean(dependsOnBean, beanName);
					}
				}

				// Create bean instance.
				if (mbd.isSingleton()) {
					sharedInstance = getSingleton(beanName, new ObjectFactory() {
						public Object getObject() throws BeansException {
							try {
								return createBean(beanName, mbd, args);
							}
							catch (BeansException ex) {
								// Explicitly remove instance from singleton cache: It might have been put there
								// eagerly by the creation process, to allow for circular reference resolution.
								// Also remove any beans that received a temporary reference to the bean.
								destroySingleton(beanName);
								throw ex;
							}
						}
					});
					bean = getObjectForBeanInstance(sharedInstance, name, beanName, mbd);
				}

				else if (mbd.isPrototype()) {
					// It's a prototype -> create a new instance.
					Object prototypeInstance = null;
					try {
						beforePrototypeCreation(beanName);
						prototypeInstance = createBean(beanName, mbd, args);
					}
					finally {
						afterPrototypeCreation(beanName);
					}
					bean = getObjectForBeanInstance(prototypeInstance, name, beanName, mbd);
				}

								else {
					String scopeName = mbd.getScope();
					final Scope scope = this.scopes.get(scopeName);
					if (scope == null) {
						throw new IllegalStateException("No Scope registered for scope '" + scopeName + "'");
					}
					try {
						Object scopedInstance = scope.get(beanName, new ObjectFactory() {
							public Object getObject() throws BeansException {
								beforePrototypeCreation(beanName);
								try {
									return createBean(beanName, mbd, args);
								}
								finally {
									afterPrototypeCreation(beanName);
								}
							}
						});
						bean = getObjectForBeanInstance(scopedInstance, name, beanName, mbd);
					}
					catch (IllegalStateException ex) {
						throw new BeanCreationException(beanName,
								"Scope '" + scopeName + "' is not active for the current thread; " +
								"consider defining a scoped proxy for this bean if you intend to refer to it from a singleton",
								ex);
					}
				}
			}

			// Check if required type matches the type of the actual bean instance.
			if (requiredType != null && bean != null && !requiredType.isAssignableFrom(bean.getClass())) {
				try {
					return getTypeConverter().convertIfNecessary(bean, requiredType);
				}
				catch (TypeMismatchException ex) {
					if (logger.isDebugEnabled()) {
						logger.debug("Failed to convert bean '" + name + "' to required type [" +
								ClassUtils.getQualifiedName(requiredType) + "]", ex);
					}
					throw new BeanNotOfRequiredTypeException(name, requiredType, bean.getClass());
				}
			}
			return (T) bean;
		}

	    ...

    }

关于scope beans的创建在这段代码：

    else {
	    String scopeName = mbd.getScope();
		final Scope scope = this.scopes.get(scopeName);
		if (scope == null) {
			throw new IllegalStateException("No Scope registered for scope '" + scopeName + "'");
		}
		try {
			Object scopedInstance = scope.get(beanName, new ObjectFactory() {
		    	public Object getObject() throws BeansException {
					beforePrototypeCreation(beanName);
			    		try {
							return createBean(beanName, mbd, args);
						}
						finally {
							afterPrototypeCreation(beanName);
						}
				    }
				});
			bean = getObjectForBeanInstance(scopedInstance, name, beanName, mbd);
		}
		catch (IllegalStateException ex) {
			throw new BeanCreationException(beanName, "Scope '" + scopeName + "'" is not active for the current thread; " + 
			"consider defining a scoped proxy for this bean if you intend to refer to it from a singleton",	ex);
		}
	}

首先拿到该scope的注册类：

    String scopeName = mbd.getScope();
	final Scope scope = this.scopes.get(scopeName);

比如对于`scope="request"`的bean定义，这里会拿到`RequestScope`类。那么Scope是什么呢，它又是怎样创建bean呢？

#### scope接口定义：

	package org.springframework.beans.factory.config;

	import org.springframework.beans.factory.ObjectFactory;

	public interface Scope {

		Object get(String name, ObjectFactory<?> objectFactory);

		Object remove(String name);

		void registerDestructionCallback(String name, Runnable callback);

		Object resolveContextualObject(String key);

		String getConversationId();
	}

#### AbstractRequestAttributesScope抽象类

	package org.springframework.web.context.request;

	import org.springframework.beans.factory.ObjectFactory;
	import org.springframework.beans.factory.config.Scope;

	public abstract class AbstractRequestAttributesScope implements Scope {

		public Object get(String name, ObjectFactory objectFactory) {
			RequestAttributes attributes = RequestContextHolder.currentRequestAttributes();
			Object scopedObject = attributes.getAttribute(name, getScope());
			if (scopedObject == null) {
				scopedObject = objectFactory.getObject();
				attributes.setAttribute(name, scopedObject, getScope());
			}
			return scopedObject;
		}

		public Object remove(String name) {
			RequestAttributes attributes = RequestContextHolder.currentRequestAttributes();
			Object scopedObject = attributes.getAttribute(name, getScope());
			if (scopedObject != null) {
				attributes.removeAttribute(name, getScope());
				return scopedObject;
			}
			else {
				return null;
			}
		}

		public void registerDestructionCallback(String name, Runnable callback) {
			RequestAttributes attributes = RequestContextHolder.currentRequestAttributes();
			attributes.registerDestructionCallback(name, callback, getScope());
		}

		public Object resolveContextualObject(String key) {
			RequestAttributes attributes = RequestContextHolder.currentRequestAttributes();
			return attributes.resolveReference(key);
		}


		/**
		 * Template method that determines the actual target scope.
		 * @return the target scope, in the form of an appropriate
		 * {@link RequestAttributes} constant
		 * @see RequestAttributes#SCOPE_REQUEST
		 * @see RequestAttributes#SCOPE_SESSION
		 * @see RequestAttributes#SCOPE_GLOBAL_SESSION
		 */
		protected abstract int getScope();

	}

#### RequestScope实现类

	package org.springframework.web.context.request;

	public class RequestScope extends AbstractRequestAttributesScope {

		@Override
		protected int getScope() {
			return RequestAttributes.SCOPE_REQUEST;
		}

		public String getConversationId() {
			return null;
		}

	}

#### SessionScope实现类（包括global session scope）

	package org.springframework.web.context.request;

	import org.springframework.beans.factory.ObjectFactory;

	public class SessionScope extends AbstractRequestAttributesScope {

		private final int scope;


		/**
		 * Create a new SessionScope, storing attributes in a locally
		 * isolated session (or default session, if there is no distinction
		 * between a global session and a component-specific session).
		 */
		public SessionScope() {
			this.scope = RequestAttributes.SCOPE_SESSION;
		}

		/**
		 * Create a new SessionScope, specifying whether to store attributes
		 * in the global session, provided that such a distinction is available.
		 * <p>This distinction is important for Portlet environments, where there
		 * are two notions of a session: "portlet scope" and "application scope".
		 * If this flag is on, objects will be put into the "application scope" session;
		 * else they will end up in the "portlet scope" session (the typical default).
		 * <p>In a Servlet environment, this flag is effectively ignored.
		 * @param globalSession <code>true</code> in case of the global session as target;
		 * <code>false</code> in case of a component-specific session as target
		 * @see org.springframework.web.portlet.context.PortletRequestAttributes
		 * @see org.springframework.web.context.request.ServletRequestAttributes
		 */
		public SessionScope(boolean globalSession) {
			this.scope = (globalSession ? RequestAttributes.SCOPE_GLOBAL_SESSION : RequestAttributes.SCOPE_SESSION);
		}


		@Override
		protected int getScope() {
			return this.scope;
		}

		public String getConversationId() {
			return RequestContextHolder.currentRequestAttributes().getSessionId();
		}

		@Override
		public Object get(String name, ObjectFactory objectFactory) {
			Object mutex = RequestContextHolder.currentRequestAttributes().getSessionMutex();
			synchronized (mutex) {
				return super.get(name, objectFactory);
			}
		}

		@Override
		public Object remove(String name) {
			Object mutex = RequestContextHolder.currentRequestAttributes().getSessionMutex();
			synchronized (mutex) {
				return super.remove(name);
			}
		}
	}
		
**注意**：由于session scope存放在线程操作，所以这里重载了父类的get和set方法，使用了互斥变量作为同步手段。

可以看到对于`scope="request"`的bean定义:`Object scopedInstance = scope.get(beanName, new ObjectFactory(){ … };`其实就是调用`RequestScope`的get方法。而这个方法其实就是简单的从`ReqestContextHolder`get一下有没有这个请求参数，如果没有就从传递进去的objectFactory创建，再放入`RequestContextHolder`中，这样就可以被线程上下文共享了。

    public Object get(String name, ObjectFactory objectFactory) {
		RequestAttributes attributes = RequestContextHolder.currentRequestAttributes();
		Object scopedObject = attributes.getAttribute(name, getScope());
		if (scopedObject == null) {
			scopedObject = objectFactory.getObject();
			attributes.setAttribute(name, scopedObject, getScope());
		}
		return scopedObject;
    }
		
这样scope bean的创建就到底了，但是还有些疑问：就是`RequestContextHolder`是什么呢？
	
#### 几个重要的辅助类

##### 1 用RequestAttributes接口封装各种scope的请求参数

Spring将各种scope的请求参数统一封装成RequestAttribute接口，支持request scope和session scope和global session scope三种scope。这么做的目的是将request scope和session scope与web request结合和统一起来。

	package org.springframework.web.context.request;

	/**
	 * Abstraction for accessing attribute objects associated with a request.
	 * Supports access to request-scoped attributes as well as to session-scoped
	 * attributes, with the optional notion of a "global session".
	 *
	 * <p>Can be implemented for any kind of request/session mechanism,
	 * in particular for servlet requests and portlet requests.
	 *
	 * @see ServletRequestAttributes
	 * @see org.springframework.web.portlet.context.PortletRequestAttributes
	 */
	public interface RequestAttributes {

		int SCOPE_REQUEST = 0;

		int SCOPE_SESSION = 1;

		int SCOPE_GLOBAL_SESSION = 2;

		String REFERENCE_REQUEST = "request";

		String REFERENCE_SESSION = "session";


		Object getAttribute(String name, int scope);

		void setAttribute(String name, Object value, int scope);

		void removeAttribute(String name, int scope);

		String[] getAttributeNames(int scope);

		void registerDestructionCallback(String name, Runnable callback, int scope);

		Object resolveReference(String key);

		String getSessionId();
		
		Object getSessionMutex();
	}

其中针对servlet的发出的web请求参数（不外乎就是通过`HttpServletRequest`和`HttpSession`对象传递），用ServletRequestAttributes实现类封装，它的实现其实就是简单判断一下scope，如果是request scope，那么就委托给`HttpServletRequest`，如果是session scope，则委托给`HttpSession`。
    
	package org.springframework.web.context.request;

	import java.util.LinkedHashMap;
	import java.util.Map;

	import org.springframework.util.Assert;

	/**
	 * Abstract support class for RequestAttributes implementations,
	 * offering a request completion mechanism for request-specific destruction
	 * callbacks and for updating accessed session attributes.
	 */
	public abstract class AbstractRequestAttributes implements RequestAttributes {
        ...
	}
>

	package org.springframework.web.context.request;

	import java.util.HashMap;
	import java.util.Map;
	import javax.servlet.http.HttpServletRequest;
	import javax.servlet.http.HttpSession;

	import org.springframework.util.Assert;
	import org.springframework.util.StringUtils;
	import org.springframework.web.util.WebUtils;

	/**
	 * Servlet-based implementation of the {@link RequestAttributes} interface.
	 *
	 * <p>Accesses objects from servlet request and HTTP session scope,
	 * with no distinction between "session" and "global session".
	 *
	 * @see javax.servlet.ServletRequest#getAttribute
	 * @see javax.servlet.http.HttpSession#getAttribute
	 */
	public class ServletRequestAttributes extends AbstractRequestAttributes {

		/**
		 * Constant identifying the {@link String} prefixed to the name of a
		 * destruction callback when it is stored in a {@link HttpSession}.
		 */
		public static final String DESTRUCTION_CALLBACK_NAME_PREFIX =
				ServletRequestAttributes.class.getName() + ".DESTRUCTION_CALLBACK.";


		private final HttpServletRequest request;

		private volatile HttpSession session;

		private final Map<String, Object> sessionAttributesToUpdate = new HashMap<String, Object>();


		/**
		 * Create a new ServletRequestAttributes instance for the given request.
		 * @param request current HTTP request
		 */
		public ServletRequestAttributes(HttpServletRequest request) {
			Assert.notNull(request, "Request must not be null");
			this.request = request;
		}


		/**
		 * Exposes the native {@link HttpServletRequest} that we're wrapping.
		 */
		public final HttpServletRequest getRequest() {
			return this.request;
		}

		/**
		 * Exposes the {@link HttpSession} that we're wrapping.
		 * @param allowCreate whether to allow creation of a new session if none exists yet
		 */
		protected final HttpSession getSession(boolean allowCreate) {
			if (isRequestActive()) {
				return this.request.getSession(allowCreate);
			}
			else {
				// Access through stored session reference, if any...
				if (this.session == null && allowCreate) {
					throw new IllegalStateException(
							"No session found and request already completed - cannot create new session!");
				}
				return this.session;
			}
		}


		public Object getAttribute(String name, int scope) {
			if (scope == SCOPE_REQUEST) {
				if (!isRequestActive()) {
					throw new IllegalStateException(
							"Cannot ask for request attribute - request is not active anymore!");
				}
				return this.request.getAttribute(name);
			}
			else {
				HttpSession session = getSession(false);
				if (session != null) {
					try {
						Object value = session.getAttribute(name);
						if (value != null) {
							synchronized (this.sessionAttributesToUpdate) {
								this.sessionAttributesToUpdate.put(name, value);
							}
						}
						return value;
					}
					catch (IllegalStateException ex) {
						// Session invalidated - shouldn't usually happen.
					}
				}
				return null;
			}
		}

		public void setAttribute(String name, Object value, int scope) {
			if (scope == SCOPE_REQUEST) {
				if (!isRequestActive()) {
					throw new IllegalStateException(
							"Cannot set request attribute - request is not active anymore!");
				}
				this.request.setAttribute(name, value);
			}
			else {
				HttpSession session = getSession(true);
				synchronized (this.sessionAttributesToUpdate) {
					this.sessionAttributesToUpdate.remove(name);
				}
				session.setAttribute(name, value);
			}
		}

		public void removeAttribute(String name, int scope) {
			if (scope == SCOPE_REQUEST) {
				if (isRequestActive()) {
					this.request.removeAttribute(name);
					removeRequestDestructionCallback(name);
				}
			}
			else {
				HttpSession session = getSession(false);
				if (session != null) {
					synchronized (this.sessionAttributesToUpdate) {
						this.sessionAttributesToUpdate.remove(name);
					}
					try {
						session.removeAttribute(name);
						// Remove any registered destruction callback as well.
						session.removeAttribute(DESTRUCTION_CALLBACK_NAME_PREFIX + name);
					}
					catch (IllegalStateException ex) {
						// Session invalidated - shouldn't usually happen.
					}
				}
			}
		}

		@SuppressWarnings("unchecked")
		public String[] getAttributeNames(int scope) {
			if (scope == SCOPE_REQUEST) {
				if (!isRequestActive()) {
					throw new IllegalStateException(
							"Cannot ask for request attributes - request is not active anymore!");
				}
				return StringUtils.toStringArray(this.request.getAttributeNames());
			}
			else {
				HttpSession session = getSession(false);
				if (session != null) {
					try {
						return StringUtils.toStringArray(session.getAttributeNames());
					}
					catch (IllegalStateException ex) {
						// Session invalidated - shouldn't usually happen.
					}
				}
				return new String[0];
			}
		}

		public String getSessionId() {
			return getSession(true).getId();
		}

		public Object getSessionMutex() {
			return WebUtils.getSessionMutex(getSession(true));
		}

        ...
	}

##### 5.2 用RequestContextHolder将请求（已经被抽象为`RequestAttributes`）保存在ThreadLocal中，从而绑定到当前线程的上下文中。这样，请求就可以对整个线程可见了。并且可以选择是否子对子线程可见。（你不需要将HttpServletRequest对象往下传递，以便让biz层的对象使用了:)）

	package org.springframework.web.context.request;

	import javax.faces.context.FacesContext;

	import org.springframework.core.NamedInheritableThreadLocal;
	import org.springframework.core.NamedThreadLocal;
	import org.springframework.util.ClassUtils;

	/**
	 * Holder class to expose the web request in the form of a thread-bound
	 * {@link RequestAttributes} object. The request will be inherited
	 * by any child threads spawned by the current thread if the
	 * <code>inheritable<code> flag is set to <code>true</code>.
	 *
	 * <p>Use {@link RequestContextListener} or
	 * {@link org.springframework.web.filter.RequestContextFilter} to expose
	 * the current web request. Note that
	 * {@link org.springframework.web.servlet.DispatcherServlet} and
	 * {@link org.springframework.web.portlet.DispatcherPortlet} already
	 * expose the current request by default.
	 *
	 * @see RequestContextListener
	 * @see org.springframework.web.filter.RequestContextFilter
	 * @see org.springframework.web.servlet.DispatcherServlet
	 * @see org.springframework.web.portlet.DispatcherPortlet
	 */
	public abstract class RequestContextHolder  {
		
		private static final boolean jsfPresent =
				ClassUtils.isPresent("javax.faces.context.FacesContext", RequestContextHolder.class.getClassLoader());

		private static final ThreadLocal<RequestAttributes> requestAttributesHolder =
				new NamedThreadLocal<RequestAttributes>("Request attributes");

		private static final ThreadLocal<RequestAttributes> inheritableRequestAttributesHolder =
				new NamedInheritableThreadLocal<RequestAttributes>("Request context");


		public static void resetRequestAttributes() {
			requestAttributesHolder.remove();
			inheritableRequestAttributesHolder.remove();
		}

		public static void setRequestAttributes(RequestAttributes attributes) {
			setRequestAttributes(attributes, false);
		}

		public static void setRequestAttributes(RequestAttributes attributes, boolean inheritable) {
			if (attributes == null) {
				resetRequestAttributes();
			}
			else {
				if (inheritable) {
					inheritableRequestAttributesHolder.set(attributes);
					requestAttributesHolder.remove();
				}
				else {
					requestAttributesHolder.set(attributes);
					inheritableRequestAttributesHolder.remove();
				}
			}
		}

		public static RequestAttributes getRequestAttributes() {
			RequestAttributes attributes = requestAttributesHolder.get();
			if (attributes == null) {
				attributes = inheritableRequestAttributesHolder.get();
			}
			return attributes;
		}

		public static RequestAttributes currentRequestAttributes() throws IllegalStateException {
			RequestAttributes attributes = getRequestAttributes();
			if (attributes == null) {
				if (jsfPresent) {
					attributes = FacesRequestAttributesFactory.getFacesRequestAttributes();
				}
				if (attributes == null) {
					throw new IllegalStateException("No thread-bound request found: " +
							"Are you referring to request attributes outside of an actual web request, " +
							"or processing a request outside of the originally receiving thread? " +
							"If you are actually operating within a web request and still receive this message, " +
							"your code is probably running outside of DispatcherServlet/DispatcherPortlet: " +
							"In this case, use RequestContextListener or RequestContextFilter to expose the current request.");
				}
			}
			return attributes;
		}

		private static class FacesRequestAttributesFactory {

			public static RequestAttributes getFacesRequestAttributes() {
				FacesContext facesContext = FacesContext.getCurrentInstance();
				return (facesContext != null ? new FacesRequestAttributes(facesContext) : null);
			}
		}
	}

**NOTES && TOPS**

RequestContextHolder其实是一个ThreadLocal的封装，所以它可以放任何东西。不一定是web request。它引用了HttpServletRequest和HttpSession对象（当然必须正确初始化，见下面），所以天然封装了web request。但是本质上就是一个线程上下文容器，所以所有需要线程共享的东东都可以放在这里面，比直接操作ThreadLocal要方便的多。事实上，Spring的request、session、global session scope bean就是放在里面，就是通过它来实现的。

#### RequestContextHolder的初始化

上面听起来不错哦，将所有web请求参数封装成`RequestAttributes`，放置在`RequestContextHolder`线程上下文中，这样既达到数据共享，又是简单的线程安全。关键是谁来做这件事情？即谁来初始化`RequestContextHolder`，将`HttpServletRequest`和`HttpSession`注入给它？这就是我们一开始提到三个类：

1. `DispatcherServlet` for Spring MVC
2. `RequestContextListener` for Servlet 2.4+ web container
3.  `RequestContextFilter` for older web container (Servlet 2.3)

这里我们看一下`RequestContextListener`的实现：

	package org.springframework.web.context.request;

	import javax.servlet.ServletRequestEvent;
	import javax.servlet.ServletRequestListener;
	import javax.servlet.http.HttpServletRequest;

	import org.springframework.context.i18n.LocaleContextHolder;

	public class RequestContextListener implements ServletRequestListener {

		private static final String REQUEST_ATTRIBUTES_ATTRIBUTE =
				RequestContextListener.class.getName() + ".REQUEST_ATTRIBUTES";


		public void requestInitialized(ServletRequestEvent requestEvent) {
			if (!(requestEvent.getServletRequest() instanceof HttpServletRequest)) {
				throw new IllegalArgumentException(
						"Request is not an HttpServletRequest: " + requestEvent.getServletRequest());
			}
			HttpServletRequest request = (HttpServletRequest) requestEvent.getServletRequest();
			ServletRequestAttributes attributes = new ServletRequestAttributes(request);
			request.setAttribute(REQUEST_ATTRIBUTES_ATTRIBUTE, attributes);
			LocaleContextHolder.setLocale(request.getLocale());
			RequestContextHolder.setRequestAttributes(attributes);
		}

		public void requestDestroyed(ServletRequestEvent requestEvent) {
			ServletRequestAttributes attributes =
					(ServletRequestAttributes) requestEvent.getServletRequest().getAttribute(REQUEST_ATTRIBUTES_ATTRIBUTE);
			ServletRequestAttributes threadAttributes =
					(ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
			if (threadAttributes != null) {
				// We're assumably within the original request thread...
				if (attributes == null) {
					attributes = threadAttributes;
				}
				LocaleContextHolder.resetLocaleContext();
				RequestContextHolder.resetRequestAttributes();
			}
			if (attributes != null) {
				attributes.requestCompleted();
			}
		}

	}

非常简单，监听一下容器的事件，将容器传递给的`HttpServletRequest`和`HttpSession`对象取出来，放置在`RequestContextHolder`中而已。

#### 将request、session scope bean也作为`RequestAttrbutes`放入`RequestContextHolder`中

当Spring实例化每个类时，如果发现它要实例的类是一个request、session或者global session scope的bean，它会先从`RequestContextHolder`中获取。如果获取不到，则创建再放入`RequestContextHolder`中，这样下次就可以取到了。这是在`RequestScope`的`public Object get(String name, ObjectFactory objectFactory)`方法：

    public Object get(String name, ObjectFactory objectFactory) {
		RequestAttributes attributes = RequestContextHolder.currentRequestAttributes();
		Object scopedObject = attributes.getAttribute(name, getScope());
		if (scopedObject == null) {
			scopedObject = objectFactory.getObject();
			attributes.setAttribute(name, scopedObject, getScope());
		}
		return scopedObject;
	}

### 2. What happen when config `<aop:scoped-proxy />`

这个配置项是由`ScopedProxyBeanDefinitionDecorator`来处理的，主要是为了解决不同声明周期的bean之间的互相注入。

	package org.springframework.aop.config;

	import org.w3c.dom.Element;
	import org.w3c.dom.Node;

	import org.springframework.aop.scope.ScopedProxyUtils;
	import org.springframework.beans.factory.config.BeanDefinitionHolder;
	import org.springframework.beans.factory.parsing.BeanComponentDefinition;
	import org.springframework.beans.factory.xml.BeanDefinitionDecorator;
	import org.springframework.beans.factory.xml.ParserContext;

	/**
	 * {@link BeanDefinitionDecorator} responsible for parsing the
	 * <code>&lt;aop:scoped-proxy/&gt;</code> tag.
	 * @since 2.0
	 */
	class ScopedProxyBeanDefinitionDecorator implements BeanDefinitionDecorator {

		private static final String PROXY_TARGET_CLASS = "proxy-target-class";


		public BeanDefinitionHolder decorate(Node node, BeanDefinitionHolder definition, ParserContext parserContext) {
			boolean proxyTargetClass = true;
			if (node instanceof Element) {
				Element ele = (Element) node;
				if (ele.hasAttribute(PROXY_TARGET_CLASS)) {
					proxyTargetClass = Boolean.valueOf(ele.getAttribute(PROXY_TARGET_CLASS));
				}
			}
			
			// Register the original bean definition as it will be referenced by the scoped proxy
			// and is relevant for tooling (validation, navigation).
			BeanDefinitionHolder holder =
					ScopedProxyUtils.createScopedProxy(definition, parserContext.getRegistry(), proxyTargetClass);
			String targetBeanName = ScopedProxyUtils.getTargetBeanName(definition.getBeanName());
			parserContext.getReaderContext().fireComponentRegistered(
					new BeanComponentDefinition(definition.getBeanDefinition(), targetBeanName));
			return holder;
		}

	}
	
为什么需要这个配置项呢？也就是说为什么web scope的beans需要代理才能被其他bean互相依赖和注入？
在Spring的官方文档上有很详细的说明：[3.5.4.5 Scoped beans as dependencies](http://static.springsource.org/spring/docs/3.0.x/reference/beans.html#beans-factory-scopes-other-injection)	。
>...	
Why do definitions of beans scoped at the request, session, globalSession and custom-scope levels require the <aop:scoped-proxy/> element ? Let's examine the following singleton bean definition and contrast it with what you need to define for the aforementioned scopes. (The following userPreferences bean definition as it stands is incomplete.)
>
    <bean id="userPreferences" class="com.foo.UserPreferences" scope="session"/>
>
    <bean id="userManager" class="com.foo.UserManager">
      <property name="userPreferences" ref="userPreferences"/>
    </bean>
>    
In the preceding example, the singleton bean userManager is injected with a reference to the HTTP Session-scoped bean userPreferences. The salient point here is that the userManager bean is a singleton: it will be instantiated exactly once per container, and its dependencies (in this case only one, the userPreferences bean) are also injected only once. This means that the userManager bean will only operate on the exact same userPreferences object, that is, the one that it was originally injected with.
>
This is not the behavior you want when injecting a shorter-lived scoped bean into a longer-lived scoped bean, for example injecting an HTTP Session-scoped collaborating bean as a dependency into singleton bean. Rather, you need a single userManager object, and for the lifetime of an HTTP Session, you need a userPreferences object that is specific to said HTTP Session. Thus the container creates an object that exposes the exact same public interface as the UserPreferences class (ideally an object that is a UserPreferences instance) which can fetch the real UserPreferences object from the scoping mechanism (HTTP request, Session, etc.). The container injects this proxy object into the userManager bean, which is unaware that this UserPreferences reference is a proxy. In this example, when a UserManager instance invokes a method on the dependency-injected UserPreferences object, it actually is invoking a method on the proxy. The proxy then fetches the real UserPreferences object from (in this case) the HTTP Session, and delegates the method invocation onto the retrieved real UserPreferences object.
>
Thus you need the following, correct and complete, configuration when injecting request-, session-, and globalSession-scoped beans into collaborating objects:
>
    <bean id="userPreferences" class="com.foo.UserPreferences" scope="session">
      <aop:scoped-proxy/>
    </bean>
>
    <bean id="userManager" class="com.foo.UserManager">
      <property name="userPreferences" ref="userPreferences"/>
    </bean>
	
简单来说，就是每个Spring容器对于singleton scope的bean，它只会实例化一次，在实例化的时候它会自动注入依赖到的其他bean，这样会触发这些bean的初始化。但是仅仅这一次。下次获取singleton bean，Spring会直接从容器中返回已经初始化的对象，它依赖到的web scope bean并不会被再次实例化，也就是说这些bean具有了跟singleton bean一样的生命周期了。这明显不是我们想要的。Spring的解决方案就是使用AOP代理，让这个AOP proxy来管理web scope bean的生命周期。
	

