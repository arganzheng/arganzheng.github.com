---
layout: post
title: Spring事务配置
---


## 原理

Spring是基于proxy方式的事务实现机制。如果你的target实现了某个接口，那么Spring可以根据这个接口动态生成一个proxy，在你的service方法前后进行如下事务管理：

    tx.begin(); 
    try{
        target.service();
    }catch(Exception e){
        tx.rollback();
    }
    tx.commit();

![](/media/images/tx.png)

而动态代理的生成方式，Spring支持两种：

1. 如果target有实现接口，那么使用JDK动态代理
2. 否则使用CGLIB Proxy

>Creates a JDK proxy when proxy interfaces are given, and a CGLIB proxy for the actual target class if not. Note that the latter will only work if the target class does not have final methods, as a dynamic subclass will be created at runtime.


## 配置

### 使用[`TransactionProxyFactoryBean`](http://static.springsource.org/spring/docs/2.5.x/api/org/springframework/transaction/interceptor/TransactionProxyFactoryBean.html)配置方式(老的方式)

>
There are three main properties that need to be specified:
>
+ "transactionManager": the PlatformTransactionManager implementation to use (for example, a JtaTransactionManager instance)
+ "target": the target object that a transactional proxy should be created for
+ "transactionAttributes": the transaction attributes (for example, propagation behavior and "readOnly" flag) per target method name (or method name pattern)

示例配置：

    <?xml version= "1.0" encoding ="UTF-8"?>
    <beans xmlns="http://www.springframework.org/schema/beans"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="
          http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-3.0.xsd">

         <!-- PlatformTransactionManager，为了让Spring @SpringJUnit4ClassRunner @Transactional生效，id必须是transactionManager -->
         <bean id="transactionManager"
           class= "org.springframework.jdbc.datasource.DataSourceTransactionManager" >
               <property name= "dataSource" ref ="dataSource" />
         </bean>

         <bean id="fooTransactionDefinition" abstract="true"
               class= "org.springframework.transaction.interceptor.TransactionProxyFactoryBean">
               <property name="transactionManager" ref="transactionManager" />
               <property name="transactionAttributes" >
                    <props>
                        <prop key="get*">PROPAGATION_SUPPORTS,readOnly</prop>
                        <prop key="find*">PROPAGATION_SUPPORTS,readOnly</prop>
                        <prop key="is*">PROPAGATION_SUPPORTS,readOnly</prop>
                        <prop key="list*">PROPAGATION_SUPPORTS,readOnly</prop>
                        <prop key="query*">PROPAGATION_SUPPORTS,readOnly</prop>
                        <prop key="*">PROPAGATION_REQUIRED </prop>
                   </props>
               </property>
         </bean>

         <bean id="fooService" parent="fooTransactionDefinition">
               <property name= "target">
                   <bean class="me.arganzheng.study.transaction.service.FooService">
                       ...
                   </bean>
               </property>
         </bean>

         <bean class= "org.springframework.beans.factory.config.PropertyPlaceholderConfigurer" >
               <property name="location">
                   <value>classpath:db.properties</value>
               </property>
         </bean>

         <bean id="dataSource" destroy-method="close"
               class= "org.apache.commons.dbcp.BasicDataSource" >
               <property name= "driverClassName" value="${jdbc.driverClassName}" />
               <property name= "url" value ="${jdbc.url}" />
               <property name= "username" value="${jdbc.username}" />
               <property name= "password" value="${jdbc.password}" />
         </bean>
    </beans>

### 使用AOP配置方式

    <?xml version= "1.0" encoding ="UTF-8"?>
    <beans xmlns="http://www.springframework.org/schema/beans"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xmlns:tx= "http://www.springframework.org/schema/tx"
          xmlns:aop="http://www.springframework.org/schema/aop"
          xsi:schemaLocation="
              http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-3.0.xsd
              http://www.springframework.org/schema/tx http://www.springframework.org/schema/tx/spring-tx-3.0.xsd
              http://www.springframework.org/schema/aop http://www.springframework.org/schema/aop/spring-aop-3.0.xsd">

         <!-- 事务配置 -->
         <tx:advice id="txAdvice" transaction-manager="transactionManager" >
             <tx:attributes>
                 <tx:method name="get*" read-only="true"/>
                 <tx:method name="find*" read-only="true"/>
                 <tx:method name="is*" read-only="true"/>
                 <tx:method name="list*" read-only="true"/>
                 <tx:method name="query*" read-only="true"/>
                 <tx:method name="*" />
             </tx:attributes>
         </tx:advice>

         <aop:config>
             <aop:pointcut id="serviceOperation" expression="execution(* me.arganzheng.study.transaction.service.*.*(..))" />
             <aop:advisor advice-ref="txAdvice" pointcut-ref="serviceOperation" />
         </aop:config>

         <!-- PlatformTransactionManager，为了让Spring @SpringJUnit4ClassRunner @Transactional生效，id必须是transactionManager -->
         <bean id="transactionManager"
               class= "org.springframework.jdbc.datasource.DataSourceTransactionManager" >
             <property name= "dataSource" ref ="dataSource" />
         </bean>

         <bean id="fooService" parent="fooTransactionDefinition">
               <property name= "target">
                   <bean class="me.arganzheng.study.transaction.service.FooService">
                       ...
                   </bean>
               </property>
         </bean>

         <bean class= "org.springframework.beans.factory.config.PropertyPlaceholderConfigurer" >
               <property name="location">
                   <value>classpath:db.properties</value>
               </property>
         </bean>

         <bean id="dataSource" destroy-method="close"
               class= "org.apache.commons.dbcp.BasicDataSource" >
               <property name= "driverClassName" value="${jdbc.driverClassName}" />
               <property name= "url" value ="${jdbc.url}" />
               <property name= "username" value="${jdbc.username}" />
               <property name= "password" value="${jdbc.password}" />
         </bean>
    </beans>

## 参考文档

1. 官方文档永远都是值得一看的：[10. Transaction Management](http://static.springsource.org/spring/docs/3.0.x/reference/transaction.html)
2. 关于AOP编程方式实现事务：[7. Aspect Oriented Programming with Spring](http://static.springsource.org/spring/docs/3.0.x/reference/aop.html)
3. 这里有个非常详细的例子：[Declaring Transaction Attributes on Spring's TransactionProxyFactoryBean](http://www.nofluffjuststuff.com/blog/scott_leberknight/2006/01/declaring_transaction_attributes_on_spring_s_transactionproxyfactorybean)
4. 结合这篇文章就可以搞定了：[Basic transactions in Spring using TransactionProxyFactoryBean](http://nerdnotes.wordpress.com/2007/03/30/basic-transactions-in-spring-using-transactionproxyfactorybean/)。基本是step-by-step，还有说明。

