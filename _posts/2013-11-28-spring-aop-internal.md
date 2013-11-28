---
layout: post
title: spring AOP internal
date: 2013-11-28
tags: [spring, aop, proxy, cglib]
---


Spring AOP mechanisms
---------------------

in a word, Spring AOP is proxy-based. That's, AOP by proxying.
and Spring use one of the following two ways to create the proxy for a given target bean.

1. JDK dynamic proxies, preferred wheneven the proxied target implements at least one interface.
2. CGLIB, if the target object does not implement any interfaces, can be force to use by:

    1. set the value of the proxy-target-class attribute of the <aop:config> element to true: 

        <aop:config proxy-target-class="true">
            <!-- other beans defined here... -->
        </aop:config>

    2.  set the 'proxy-target-class' attribute of the <aop:aspectj-autoproxy> element to true when using the @AspectJ autoproxy support:

        <aop:aspectj-autoproxy proxy-target-class="true"/>


Spring AOP Implement
--------------------

difine a AopProxy Interface, see [AopProxy](http://docs.spring.io/spring/docs/3.2.0.RC1_to_3.2.0.RC2/Spring%20Framework%203.2.0.RC2/org/springframework/aop/framework/AopProxy.html)

    package org.springframework.aop.framework;

    /**
     * Delegate interface for a configured AOP proxy, allowing for the creation
     * of actual proxy objects.
     *
     * <p>Out-of-the-box implementations are available for JDK dynamic proxies
     * and for CGLIB proxies, as applied by {@link DefaultAopProxyFactory}.
     *
     * @author Rod Johnson
     * @author Juergen Hoeller
     * @see DefaultAopProxyFactory
     */
    public interface AopProxy {
        Object getProxy();

        Object getProxy(ClassLoader classLoader);
    }

There are two implements for the AopProxy:

1. [JdkDynamicAopProxy](http://docs.spring.io/spring/docs/3.2.0.RC1_to_3.2.0.RC2/Spring%20Framework%203.2.0.RC2/org/springframework/aop/framework/JdkDynamicAopProxy.html)
2. [CglibAopProxy](http://docs.spring.io/spring/docs/3.2.0.RC1_to_3.2.0.RC2/Spring%20Framework%203.2.0.RC2/org/springframework/aop/framework/CglibAopProxy.html)

The `JdkDynamicAopProxy` is implemented based on JDK [java.lang.reflect.Proxy](http://tutorials.jenkov.com/java-reflection/dynamic-proxies.html). It's quite easy to create a proxy base on the target object's interface:
 
    final class JdkDynamicAopProxy implements AopProxy, InvocationHandler, Serializable {

        public Object getProxy(ClassLoader classLoader) {
            if (logger.isDebugEnabled()) {
                logger.debug("Creating JDK dynamic proxy: target source is " + this.advised.getTargetSource());
            }
            // Determine the complete set of interfaces to proxy for the given AOP configuration. 
            Class[] proxiedInterfaces = AopProxyUtils.completeProxiedInterfaces(this.advised);
            findDefinedEqualsAndHashCodeMethods(proxiedInterfaces);
            // Returns an instance of a proxy class for the specified interfaces that dispatches method invocations to the specified invocation handler. 
            return Proxy.newProxyInstance(classLoader, proxiedInterfaces, this);
        }

    }

with Proxy class's newProxyInstance method: 

    public static Object newProxyInstance(ClassLoader loader,
                  Class<?>[] interfaces,
                  InvocationHandler h)

note that the InvocationHandler argument above is the JdkDynamicAopProxy instance itselft. This is ture cause the JdkDynamicAopProxy implements the InvocationHandler interface too.

    public interface InvocationHandler {

        public Object invoke(Object proxy, Method method, Object[] args)
        throws Throwable;
    }

The implement code for invoke method is too long, and I am not going to paste the code here. See the source code directly if you are interested. Just remember that it's nothing but the proxy pattern. The proxy implements the interfaces of the target object, and has a reference to the target.

![proxy-pattern](/media/images/proxy-pattern.png)


The `CglibAopProxy` use the [CGLIB](http://cglib.sourceforge.net/) library to generate the proxy object.


    final class CglibAopProxy implements AopProxy, Serializable {

        public Object getProxy() {
            return getProxy(null);
        }

        public Object getProxy(ClassLoader classLoader) {
            try {
                Class<?> rootClass = this.advised.getTargetClass();
                Assert.state(rootClass != null, "Target class must be available for creating a CGLIB proxy");

                Class<?> proxySuperClass = rootClass;
                if (ClassUtils.isCglibProxyClass(rootClass)) {
                    proxySuperClass = rootClass.getSuperclass();
                    Class<?>[] additionalInterfaces = rootClass.getInterfaces();
                    for (Class<?> additionalInterface : additionalInterfaces) {
                        this.advised.addInterface(additionalInterface);
                    }
                }

                // Validate the class, writing log messages as necessary.
                validateClassIfNecessary(proxySuperClass);

                // Configure CGLIB Enhancer...
                Enhancer enhancer = createEnhancer();
                if (classLoader != null) {
                    enhancer.setClassLoader(classLoader);
                    if (classLoader instanceof SmartClassLoader &&
                            ((SmartClassLoader) classLoader).isClassReloadable(proxySuperClass)) {
                        enhancer.setUseCache(false);
                    }
                }
                enhancer.setSuperclass(proxySuperClass);
                enhancer.setStrategy(new UndeclaredThrowableStrategy(UndeclaredThrowableException.class));
                enhancer.setInterfaces(AopProxyUtils.completeProxiedInterfaces(this.advised));
                enhancer.setInterceptDuringConstruction(false);

                Callback[] callbacks = getCallbacks(rootClass);
                enhancer.setCallbacks(callbacks);
                enhancer.setCallbackFilter(new ProxyCallbackFilter(
                        this.advised.getConfigurationOnlyCopy(), this.fixedInterceptorMap, this.fixedInterceptorOffset));

                Class<?>[] types = new Class[callbacks.length];
                for (int x = 0; x < types.length; x++) {
                    types[x] = callbacks[x].getClass();
                }
                enhancer.setCallbackTypes(types);

                // Generate the proxy class and create a proxy instance.
                Object proxy;
                if (this.constructorArgs != null) {
                    proxy = enhancer.create(this.constructorArgTypes, this.constructorArgs);
                }
                else {
                    proxy = enhancer.create();
                }

                return proxy;
            }
            catch () {
                // ..Exception handle..
            }
        }
    }    

Note that how CglibAopProxy different from the JdkDynamicAopProxy. The CglibAopProxy will construct a class(Enhancer) that extends the target class, instead of implements the target class's interfaces.

    enhancer.setSuperclass(proxySuperClass);

That's why it can proxy target object that has no interface.

Also note that both the JdkDynamicAopProxy and CglibAopProxy construct the proxy object on the fly, that is on runtime, not compile time. And because it has to construct the proxy class first before it can instance the proxy object, this may be a performance bottleneck. But in spring, this offen happen once when startup spring application context.


Spring use the DefaultAopProxyFactory to create the AOP proxy object. DefaultAopProxyFactory will automatically create CGLIB-based proxies if one the following is true for a given AdvisedSupport instance: 

 * the "optimize" flag is set 
 * the "proxyTargetClass" flag is set 
 * no proxy interfaces have been specified 

**DefaultAopProxyFactory**

    public class DefaultAopProxyFactory implements AopProxyFactory, Serializable {
        public AopProxy createAopProxy(AdvisedSupport config) throws AopConfigException {
            if (config.isOptimize() || config.isProxyTargetClass() || hasNoUserSuppliedProxyInterfaces(config)) {
                Class targetClass = config.getTargetClass();
                if (targetClass == null) {
                    throw new AopConfigException("TargetSource cannot determine target class: " +
                            "Either an interface or a target is required for proxy creation.");
                }
                if (targetClass.isInterface()) {
                    return new JdkDynamicAopProxy(config);
                }
                return CglibProxyFactory.createCglibProxy(config);
            }
            else {
                return new JdkDynamicAopProxy(config);
            }
        }

        /**
         * Inner factory class used to just introduce a CGLIB dependency
         * when actually creating a CGLIB proxy.
         */
        private static class CglibProxyFactory {

            public static AopProxy createCglibProxy(AdvisedSupport advisedSupport) {
                return new CglibAopProxy(advisedSupport);
            }
        }
    }


Resources
---------

1. [9. Aspect Oriented Programming with Spring](http://docs.spring.io/spring/docs/3.2.4.RELEASE/spring-framework-reference/html/aop.html#aop-understanding-aop-proxies)
2. [8.6 Proxying mechanisms](http://docs.spring.io/spring/docs/3.0.0.M3/reference/html/ch08s06.html)
3. [Java Reflection: Dynamic Proxies](http://tutorials.jenkov.com/java-reflection/dynamic-proxies.html)