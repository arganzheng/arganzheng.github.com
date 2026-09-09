---
title: mybatis-spring-boot把service接口当成Mapper扫描问题原因和解决方案
layout: post
catalog: true
---


package com.baidu.mobojoy.mobopay.remote.user;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.util.Assert;

import com.baidu.mobojoy.mobopay.modules.user.domain.User;
import com.baidu.mobojoy.mobopay.modules.user.service.impl.UserServiceImpl;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = UserDubboProviderApplication.class)
@TestPropertySource(locations = "classpath:application-development.properties")
public class UserServiceApiImplTest {

    @Autowired
    private UserServiceImpl userServiceImpl;

    @Test
    @Ignore
    public void contextLoads() {
    }

    @Test
    public void testGetVistorUserInfo() {
        String deviceId = "arganzheng";
        String countryCode = "USA";
        String androidId = "iphone6s";
        User user = userServiceImpl.getVistorUserInfo(deviceId, countryCode, androidId);
        Assert.notNull(user);
    }
}



	org.apache.ibatis.binding.BindingException: Invalid bound statement (not found): com.baidu.mobojoy.mobopay.modules.user.service.interfaces.UserService.getVistorUserInfo
		at org.apache.ibatis.binding.MapperMethod$SqlCommand.<init>(MapperMethod.java:196)
		at org.apache.ibatis.binding.MapperMethod.<init>(MapperMethod.java:44)
		at org.apache.ibatis.binding.MapperProxy.cachedMapperMethod(MapperProxy.java:59)
		at org.apache.ibatis.binding.MapperProxy.invoke(MapperProxy.java:52)
		at com.sun.proxy.$Proxy46.getVistorUserInfo(Unknown Source)
		at com.baidu.mobojoy.mobopay.modules.user.service.impl.UserServiceImplTest.testSelectAccountByName(UserServiceImplTest.java:22)
		at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
		at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
		at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
		at java.lang.reflect.Method.invoke(Method.java:497)
		at org.junit.runners.model.FrameworkMethod$1.runReflectiveCall(FrameworkMethod.java:50)
		at org.junit.internal.runners.model.ReflectiveCallable.run(ReflectiveCallable.java:12)
		at org.junit.runners.model.FrameworkMethod.invokeExplosively(FrameworkMethod.java:47)
		at org.junit.internal.runners.statements.InvokeMethod.evaluate(InvokeMethod.java:17)
		at org.springframework.test.context.junit4.statements.RunBeforeTestMethodCallbacks.evaluate(RunBeforeTestMethodCallbacks.java:75)
		at org.springframework.test.context.junit4.statements.RunAfterTestMethodCallbacks.evaluate(RunAfterTestMethodCallbacks.java:86)
		at org.springframework.test.context.junit4.statements.SpringRepeat.evaluate(SpringRepeat.java:84)
		at org.junit.runners.ParentRunner.runLeaf(ParentRunner.java:325)
		at org.springframework.test.context.junit4.SpringJUnit4ClassRunner.runChild(SpringJUnit4ClassRunner.java:254)
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


解决方案

1. 不要自动装配mybatis-spring-boot，使用原来的手动配置方式。

方法，不依赖
	@EnableAutoConfiguration(exclude={MybatisAutoConfiguration.class})




