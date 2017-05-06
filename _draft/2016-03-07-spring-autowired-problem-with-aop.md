advised:

org.springframework.aop.framework.ProxyFactory: 1 interfaces [com.baidu.mobojoy.mobopay.modules.user.service.interfaces.UserService]; 2 advisors [org.springframework.aop.interceptor.ExposeInvocationInterceptor.ADVISOR, InstantiationModelAwarePointcutAdvisor: expression [annotationPointCut()]; advice method [public java.lang.Object com.baidu.mobojoy.mobopay.common.cache.RedisCacheAspect.aroundQueryDB(org.aspectj.lang.ProceedingJoinPoint) throws java.lang.Throwable]; perClauseKind=SINGLETON]; targetSource [SingletonTargetSource for target object [com.baidu.mobojoy.mobopay.modules.user.service.impl.UserServiceImpl@5560b64d]]; proxyTargetClass=false; optimize=false; opaque=false; exposeProxy=false; frozen=false
catalog: true


org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'com.baidu.mobojoy.mobopay.remote.user.UserServiceApiImplTest': Injection of autowired dependencies failed; nested exception is org.springframework.beans.factory.BeanCreationException: Could not autowire field: private com.baidu.mobojoy.mobopay.modules.user.service.impl.UserServiceImpl com.baidu.mobojoy.mobopay.remote.user.UserServiceApiImplTest.userServiceImpl; nested exception is org.springframework.beans.factory.NoSuchBeanDefinitionException: No qualifying bean of type [com.baidu.mobojoy.mobopay.modules.user.service.impl.UserServiceImpl] found for dependency: expected at least 1 bean which qualifies as autowire candidate for this dependency. Dependency annotations: {@org.springframework.beans.factory.annotation.Autowired(required=true)}
	at org.springframework.beans.factory.annotation.AutowiredAnnotationBeanPostProcessor.postProcessPropertyValues(AutowiredAnnotationBeanPostProcessor.java:334)
	at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.populateBean(AbstractAutowireCapableBeanFactory.java:1214)
	at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.autowireBeanProperties(AbstractAutowireCapableBeanFactory.java:385)
	at org.springframework.test.context.support.DependencyInjectionTestExecutionListener.injectDependencies(DependencyInjectionTestExecutionListener.java:118)
	at org.springframework.test.context.support.DependencyInjectionTestExecutionListener.prepareTestInstance(DependencyInjectionTestExecutionListener.java:83)
	at org.springframework.test.context.TestContextManager.prepareTestInstance(TestContextManager.java:228)
	at org.springframework.test.context.junit4.SpringJUnit4ClassRunner.createTest(SpringJUnit4ClassRunner.java:230)
	at org.springframework.test.context.junit4.SpringJUnit4ClassRunner$1.runReflectiveCall(SpringJUnit4ClassRunner.java:289)
	at org.junit.internal.runners.model.ReflectiveCallable.run(ReflectiveCallable.java:12)
	at org.springframework.test.context.junit4.SpringJUnit4ClassRunner.methodBlock(SpringJUnit4ClassRunner.java:291)
	at org.springframework.test.context.junit4.SpringJUnit4ClassRunner.runChild(SpringJUnit4ClassRunner.java:249)
	at org.springframework.test.context.junit4.SpringJUnit4ClassRunner.runChild(SpringJUnit4ClassRunner.java:89)
	at org.junit.runners.ParentRunner$3.run(ParentRunner.java:290)
	at org.junit.runners.ParentRunner$1.schedule(ParentRunner.java:71)
	at org.junit.runners.ParentRunner.runChildren(ParentRunner.java:288)
	at org.junit.runners.ParentRunner.access$000(ParentRunner.java:58)
	at org.junit.runners.ParentRunner$2.evaluate(ParentRunner.java:268)
	at org.springframework.test.context.junit4.statements.RunBeforeTestClassCallbacks.evaluate(RunBeforeTestClassCallbacks.java:61)
	at org.springframework.test.context.junit4.statements.RunAfterTestClassCallbacks.evaluate(RunAfterTestClassCallbacks.java:70)
	at org.junit.runners.ParentRunner.run(ParentRunner.java:363)
	at org.springframework.test.context.junit4.SpringJUnit4ClassRunner.run(SpringJUnit4ClassRunner.java:193)
	at org.eclipse.jdt.internal.junit4.runner.JUnit4TestReference.run(JUnit4TestReference.java:50)
	at org.eclipse.jdt.internal.junit.runner.TestExecution.run(TestExecution.java:38)
	at org.eclipse.jdt.internal.junit.runner.RemoteTestRunner.runTests(RemoteTestRunner.java:459)
	at org.eclipse.jdt.internal.junit.runner.RemoteTestRunner.runTests(RemoteTestRunner.java:675)
	at org.eclipse.jdt.internal.junit.runner.RemoteTestRunner.run(RemoteTestRunner.java:382)
	at org.eclipse.jdt.internal.junit.runner.RemoteTestRunner.main(RemoteTestRunner.java:192)
Caused by: org.springframework.beans.factory.BeanCreationException: Could not autowire field: private com.baidu.mobojoy.mobopay.modules.user.service.impl.UserServiceImpl com.baidu.mobojoy.mobopay.remote.user.UserServiceApiImplTest.userServiceImpl; nested exception is org.springframework.beans.factory.NoSuchBeanDefinitionException: No qualifying bean of type [com.baidu.mobojoy.mobopay.modules.user.service.impl.UserServiceImpl] found for dependency: expected at least 1 bean which qualifies as autowire candidate for this dependency. Dependency annotations: {@org.springframework.beans.factory.annotation.Autowired(required=true)}
	at org.springframework.beans.factory.annotation.AutowiredAnnotationBeanPostProcessor$AutowiredFieldElement.inject(AutowiredAnnotationBeanPostProcessor.java:573)
	at org.springframework.beans.factory.annotation.InjectionMetadata.inject(InjectionMetadata.java:88)
	at org.springframework.beans.factory.annotation.AutowiredAnnotationBeanPostProcessor.postProcessPropertyValues(AutowiredAnnotationBeanPostProcessor.java:331)
	... 26 more
Caused by: org.springframework.beans.factory.NoSuchBeanDefinitionException: No qualifying bean of type [com.baidu.mobojoy.mobopay.modules.user.service.impl.UserServiceImpl] found for dependency: expected at least 1 bean which qualifies as autowire candidate for this dependency. Dependency annotations: {@org.springframework.beans.factory.annotation.Autowired(required=true)}
	at org.springframework.beans.factory.support.DefaultListableBeanFactory.raiseNoSuchBeanDefinitionException(DefaultListableBeanFactory.java:1373)
	at org.springframework.beans.factory.support.DefaultListableBeanFactory.doResolveDependency(DefaultListableBeanFactory.java:1119)
	at org.springframework.beans.factory.support.DefaultListableBeanFactory.resolveDependency(DefaultListableBeanFactory.java:1014)
	at org.springframework.beans.factory.annotation.AutowiredAnnotationBeanPostProcessor$AutowiredFieldElement.inject(AutowiredAnnotationBeanPostProcessor.java:545)
	... 28 more


1. 检查Spring容器是不是已经有这个bean：applicationContext.getBean(""); 可以通过 DefaultListableBeanFactory.beanDefinitionNames查看所有托管的bean名称。

	需要注意的时候名称存在，bean存在未必意味着注入成功，还需要判断类型是否匹配。否则会出错。
2. 如果不存在，说明包扫描有问题
3. 如果存在，那么说明 autowired查找有问题。有可能是类型不匹配
