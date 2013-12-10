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

当然，你可以通过配置强制Spring使用CGLIB Proxy方式。具体参考笔者的另一篇文章：[Spring AOP internal](http://blog.arganzheng.me/posts/spring-aop-internal.html)


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
                 <tx:method name="get*" propagation="SUPPORTS" read-only="true"/>
                 <tx:method name="find*" propagation="SUPPORTS" read-only="true"/>
                 <tx:method name="is*" propagation="SUPPORTS" read-only="true"/>
                 <tx:method name="list*" propagation="SUPPORTS" read-only="true"/>
                 <tx:method name="query*" propagation="SUPPORTS" read-only="true"/>
                 <tx:method name="*" />
             </tx:attributes>
         </tx:advice>

         <aop:config proxy-target-class="true">
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

**注意** Spring默认会将propagation设置为`REQUIRED`，将readOnly设置为false，并且rollback only for `unchecked exceptions`。另外，isolation level设置为底层数据库的default值。如果是使用JPA，isolation level 是底层的persistence provider。如果是Hibernate，所有事务的isolation level应该设置为REPEATABLE_READ。如果你没有配置`propagation="SUPPORTS"`而只是配置了`read-only="true"`，你会看到一个Spring对一个readOnly的事务也是起一个事务，并且成功后会commit改事务的哦！


    [12-10 14:04:40.202][DEBUG][Thread-14] C2CTcpIoHandler:41 : (0x00497E4B: nio socket, server, /172.xxx.xxx.xxx:35216 => /10.191.xxx.xxx:xxxx) sessionOpened。
    [12-10 14:04:40.206][DEBUG][Thread-14] DefaultListableBeanFactory:244 : Returning cached instance of singleton bean 'lotteryAO'
    [12-10 14:04:40.206][DEBUG][Thread-14] DataSourceTransactionManager:365 : Creating new transaction with name [com.ecc.victor.buyer.activity.lottery.service.LotteryRecordService.getRecordList4o2o]: PROPAGATION_REQUIRED,ISOLATION_DEFAULT,readOnly
    [12-10 14:04:40.269][DEBUG][Thread-14] DataSourceTransactionManager:204 : Acquired Connection [jdbc:mysql://xxx.xxx.xxx.xxx:xxxxx/xxxxx?useUnicode=true&characterEncoding=utf8&zeroDateTimeBehavior=convertToNull, UserName=xxxx@xxx.xxx.xxx.xxx, MySQL-AB JDBC Driver] for JDBC transaction
    [12-10 14:04:40.299][DEBUG][Thread-14] DataSourceUtils:153 : Setting JDBC Connection [jdbc:mysql://xxx.xxx.xxx.xxx:xxxx/xxx?useUnicode=true&characterEncoding=utf8&zeroDateTimeBehavior=convertToNull, UserName=xxx@xxx.xxx.xxx.xxx, MySQL-AB JDBC Driver] read-only
    [12-10 14:04:40.330][DEBUG][Thread-14] DataSourceTransactionManager:221 : Switching JDBC Connection [jdbc:mysql://xxx.xxx.xxx.xxx:xxxx/xxxx?useUnicode=true&characterEncoding=utf8&zeroDateTimeBehavior=convertToNull, UserName=xxxx@xxx.xxx.xxx.xxx, MySQL-AB JDBC Driver] to manual commit
    [12-10 14:04:40.360][DEBUG][Thread-14] SqlMapClientTemplate:168 : Opened SqlMapSession [com.ibatis.sqlmap.engine.impl.SqlMapSessionImpl@547aaf5a] for iBATIS operation
    [12-10 14:04:40.360][DEBUG][Thread-14] Connection:27 : {conn-100627} Connection
    [12-10 14:04:40.360][DEBUG][Thread-14] SqlMapClientTemplate:185 : Obtained JDBC Connection [Transaction-aware proxy for target Connection  from DataSource [org.apache.commons.dbcp.BasicDataSource@277889e9]] for iBATIS operation
    [12-10 14:04:40.361][DEBUG][Thread-14] Connection:27 : {conn-100627} Preparing Statement:    select    *   from    xxxxxx    where     user_id=?     and     (status='xxx' or status='xxxx' or status='xxxx')    and     activity_id in (select id from xxxx where seller_uin=? and type='xxx')  
    [12-10 14:04:40.361][DEBUG][Thread-14] PreparedStatement:27 : {pstm-100628} Executing Statement:    select    *   from    xxxxx    where     user_id=?     and     (status='Hit' or status='xxxx' or status='xxx')    and     activity_id in (select id from t_victor_activity where seller_uin=? and type='xxxx')  
    [12-10 14:04:40.361][DEBUG][Thread-14] PreparedStatement:27 : {pstm-100628} Parameters: [xxx, xxxx]
    [12-10 14:04:40.361][DEBUG][Thread-14] PreparedStatement:27 : {pstm-100628} Types: [java.lang.String, java.lang.Long]
    [12-10 14:04:40.392][DEBUG][Thread-14] ResultSet:27 : {rset-100629} ResultSet
    ...
    [12-10 14:04:40.395][DEBUG][Thread-14] DataSourceTransactionManager:752 : Initiating transaction commit
    [12-10 14:04:40.426][DEBUG][Thread-14] DataSourceTransactionManager:264 : Committing JDBC transaction on Connection [jdbc:mysql://xxx.xxx.xxx.xxx:xxx/xxxx?useUnicode=true&characterEncoding=utf8&zeroDateTimeBehavior=convertToNull, UserName=xxx@xxx.xxx.xxx.xxx, MySQL-AB JDBC Driver]
    [12-10 14:04:40.517][DEBUG][Thread-14] DataSourceUtils:222 : Resetting read-only flag of JDBC Connection [jdbc:mysql://xxx.xxx.xxx.xxx:xxx/xxx?useUnicode=true&characterEncoding=utf8&zeroDateTimeBehavior=convertToNull, UserName=xxx@xxx.xxx.xxx.xxx, MySQL-AB JDBC Driver]
    [12-10 14:04:40.547][DEBUG][Thread-14] DataSourceTransactionManager:322 : Releasing JDBC Connection [jdbc:mysql://xxx.xxx.xxx.xxx:xxx/xxxx?useUnicode=true&characterEncoding=utf8&zeroDateTimeBehavior=convertToNull, UserName=xxxx@xxx.xxx.xxx.xxx, MySQL-AB JDBC Driver] after transaction
    [12-10 14:04:40.547][DEBUG][Thread-14] DataSourceUtils:332 : Returning JDBC Connection to DataSource
    [12-10 14:04:40.547][DEBUG][Thread-14] LotteryActivityService:128 : LotteryAO getLotteryRecord getRecordList4o2o cost[341]ms
    [12-10 14:04:40.548][DEBUG][Thread-14] LotteryActivityService:142 : LotteryAO getLotteryRecord cost[342]ms


### 使用`@Transactional`注解配置方式

其实从Spring2.x就开始支持Anotation方式进行事务配置了。不过相对于XML配置，粒度比较细。


    package org.springframework.transaction.annotation;

    import java.lang.annotation.Documented;
    import java.lang.annotation.ElementType;
    import java.lang.annotation.Inherited;
    import java.lang.annotation.Retention;
    import java.lang.annotation.RetentionPolicy;
    import java.lang.annotation.Target;

    import org.springframework.transaction.TransactionDefinition;

    @Target({ElementType.METHOD, ElementType.TYPE})
    @Retention(RetentionPolicy.RUNTIME)
    @Inherited
    @Documented
    public @interface Transactional {

      /**
       * A qualifier value for the specified transaction.
       * <p>May be used to determine the target transaction manager,
       * matching the qualifier value (or the bean name) of a specific
       * {@link org.springframework.transaction.PlatformTransactionManager}
       * bean definition.
       */
      String value() default "";

      /**
       * The transaction propagation type.
       * Defaults to {@link Propagation#REQUIRED}.
       * @see org.springframework.transaction.interceptor.TransactionAttribute#getPropagationBehavior()
       */
      Propagation propagation() default Propagation.REQUIRED;

      /**
       * The transaction isolation level.
       * Defaults to {@link Isolation#DEFAULT}.
       * @see org.springframework.transaction.interceptor.TransactionAttribute#getIsolationLevel()
       */
      Isolation isolation() default Isolation.DEFAULT;

      /**
       * The timeout for this transaction.
       * Defaults to the default timeout of the underlying transaction system.
       * @see org.springframework.transaction.interceptor.TransactionAttribute#getTimeout()
       */
      int timeout() default TransactionDefinition.TIMEOUT_DEFAULT;

      /**
       * {@code true} if the transaction is read-only.
       * Defaults to {@code false}.
       * <p>This just serves as a hint for the actual transaction subsystem;
       * it will <i>not necessarily</i> cause failure of write access attempts.
       * A transaction manager which cannot interpret the read-only hint will
       * <i>not</i> throw an exception when asked for a read-only transaction.
       * @see org.springframework.transaction.interceptor.TransactionAttribute#isReadOnly()
       */
      boolean readOnly() default false;

      /**
       * Defines zero (0) or more exception {@link Class classes}, which must be a
       * subclass of {@link Throwable}, indicating which exception types must cause
       * a transaction rollback.
       * <p>This is the preferred way to construct a rollback rule, matching the
       * exception class and subclasses.
       * <p>Similar to {@link org.springframework.transaction.interceptor.RollbackRuleAttribute#RollbackRuleAttribute(Class clazz)}
       */
      Class<? extends Throwable>[] rollbackFor() default {};

      /**
       * Defines zero (0) or more exception names (for exceptions which must be a
       * subclass of {@link Throwable}), indicating which exception types must cause
       * a transaction rollback.
       * <p>This can be a substring, with no wildcard support at present.
       * A value of "ServletException" would match
       * {@link javax.servlet.ServletException} and subclasses, for example.
       * <p><b>NB: </b>Consider carefully how specific the pattern is, and whether
       * to include package information (which isn't mandatory). For example,
       * "Exception" will match nearly anything, and will probably hide other rules.
       * "java.lang.Exception" would be correct if "Exception" was meant to define
       * a rule for all checked exceptions. With more unusual {@link Exception}
       * names such as "BaseBusinessException" there is no need to use a FQN.
       * <p>Similar to {@link org.springframework.transaction.interceptor.RollbackRuleAttribute#RollbackRuleAttribute(String exceptionName)}
       */
      String[] rollbackForClassName() default {};

      /**
       * Defines zero (0) or more exception {@link Class Classes}, which must be a
       * subclass of {@link Throwable}, indicating which exception types must <b>not</b>
       * cause a transaction rollback.
       * <p>This is the preferred way to construct a rollback rule, matching the
       * exception class and subclasses.
       * <p>Similar to {@link org.springframework.transaction.interceptor.NoRollbackRuleAttribute#NoRollbackRuleAttribute(Class clazz)}
       */
      Class<? extends Throwable>[] noRollbackFor() default {};

      /**
       * Defines zero (0) or more exception names (for exceptions which must be a
       * subclass of {@link Throwable}) indicating which exception types must <b>not</b>
       * cause a transaction rollback.
       * <p>See the description of {@link #rollbackForClassName()} for more info on how
       * the specified names are treated.
       * <p>Similar to {@link org.springframework.transaction.interceptor.NoRollbackRuleAttribute#NoRollbackRuleAttribute(String exceptionName)}
       */
      String[] noRollbackForClassName() default {};

    }

可以注在方法或者类上，基本是跟XML配置的`<tx:method/>`标签对应。


## 参考文档

1. 官方文档永远都是值得一看的：[10. Transaction Management](http://static.springsource.org/spring/docs/3.0.x/reference/transaction.html)
2. 关于AOP编程方式实现事务：[7. Aspect Oriented Programming with Spring](http://static.springsource.org/spring/docs/3.0.x/reference/aop.html)
3. 这里有个非常详细的例子：[Declaring Transaction Attributes on Spring's TransactionProxyFactoryBean](http://www.nofluffjuststuff.com/blog/scott_leberknight/2006/01/declaring_transaction_attributes_on_spring_s_transactionproxyfactorybean)
4. 结合这篇文章就可以搞定了：[Basic transactions in Spring using TransactionProxyFactoryBean](http://nerdnotes.wordpress.com/2007/03/30/basic-transactions-in-spring-using-transactionproxyfactorybean/)。基本是step-by-step，还有说明。

