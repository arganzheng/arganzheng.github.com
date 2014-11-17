---
title: JPA的事务管理器配置
layout: post
---

Guanxing工程在后来为了避免写DAO，引入了JPA。为了避免影响老的工程，JPA使用了自己的事务管理器。

spring-service.xml

	<!-- 事务配置 -->
	<tx:advice id="txAdvice" transaction-manager="transactionManager">
		<tx:attributes>
			<tx:method name="get*" propagation="SUPPORTS" read-only="true" />
			<tx:method name="find*" propagation="SUPPORTS" read-only="true" />
			<tx:method name="is*" propagation="SUPPORTS" read-only="true" />
			<tx:method name="list*" propagation="SUPPORTS" read-only="true" />
			<tx:method name="query*" propagation="SUPPORTS" read-only="true" />
			<tx:method name="*" />
		</tx:attributes>
	</tx:advice>

	<aop:config proxy-target-class="true">
		<aop:pointcut id="serviceOperation"
			expression="execution(* com.baidu.global.mobile.server.guanxing.service..*(..))" />
		<aop:advisor advice-ref="txAdvice" pointcut-ref="serviceOperation" />
	</aop:config>

	<!-- PlatformTransactionManager，为了让Spring @SpringJUnit4ClassRunner @Transactional生效，id必须是transactionManager -->
	<bean id="transactionManager"
		class="org.springframework.jdbc.datasource.DataSourceTransactionManager">
		<property name="dataSource" ref="dataSource" />
	</bean>

spring-jpa.xml

	<?xml version="1.0" encoding="UTF-8"?>
	<beans xmlns="http://www.springframework.org/schema/beans"
		xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:context="http://www.springframework.org/schema/context"
		xmlns:tx="http://www.springframework.org/schema/tx" xmlns:jpa="http://www.springframework.org/schema/data/jpa"
		xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd
			http://www.springframework.org/schema/context http://www.springframework.org/schema/context/spring-context.xsd
			http://www.springframework.org/schema/tx 
	        http://www.springframework.org/schema/tx/spring-tx.xsd
	        http://www.springframework.org/schema/data/jpa
	        http://www.springframework.org/schema/data/jpa/spring-jpa.xsd">

		<!-- Configures Spring Data JPA and sets the base package of my DAOs. -->
		<jpa:repositories
			base-package="com.baidu.global.mobile.server.guanxing.monitor.repository" />

		<bean id="emf"
			class="org.springframework.orm.jpa.LocalContainerEntityManagerFactoryBean">
			<property name="dataSource" ref="dataSource" />
			<property name="packagesToScan"
				value="com.baidu.global.mobile.server.guanxing.monitor.model" />
			<property name="jpaVendorAdapter">
				<bean class="org.springframework.orm.jpa.vendor.HibernateJpaVendorAdapter" />
				<!-- <bean class="org.springframework.orm.jpa.vendor.HibernateJpaVendorAdapter" 
					/> <property name="generateDdl" value="${jpa.generateDdl}" /> <property name="databasePlatform" 
					value="${persistence.dialect}" /> </bean> -->
			</property>
			<property name="jpaProperties">
				<props>
					<prop key="hibernate.hbm2ddl.auto">${hibernate.hbm2ddl.auto}</prop>
					<prop key="hibernate.dialect">${hibernate.dialect}</prop>
					<prop key="hibernate.format_sql">${hibernate.format_sql}</prop>
					<prop key="hibernate.show_sql">${hibernate.show_sql}</prop>
				</props>
			</property>
		</bean>

		<bean id="transactionManager" class="org.springframework.orm.jpa.JpaTransactionManager">
			<property name="entityManagerFactory" ref="emf" />
		</bean>

		<tx:annotation-driven />

		<!-- <bean id="persistenceExceptionTranslationPostProcessor" class="org.springframework.dao.annotation.PersistenceExceptionTranslationPostProcessor" 
			/> -->

	</beans>

然后发现线上经常报JPA事务配置转换错误（具体报错信息但是没有记录下来，忘记了。。）。定位了一下，发现是因为id冲突了，

	<!-- PlatformTransactionManager，为了让Spring @SpringJUnit4ClassRunner @Transactional生效，id必须是transactionManager -->
	<bean id="transactionManager"
		class="org.springframework.jdbc.datasource.DataSourceTransactionManager">
		<property name="dataSource" ref="dataSource" />
	</bean>

	<bean id="transactionManager" class="org.springframework.orm.jpa.JpaTransactionManager">
		<property name="entityManagerFactory" ref="emf" />
	</bean>


DataSourceTransactionManager和JpaTransactionManager的id都是系统默认的transactionManager。这样的结果就会导致一个被无视了。于是将JPA的改了一个id，变成：

	<bean id="txManager" class="org.springframework.orm.jpa.JpaTransactionManager">
		<property name="entityManagerFactory" ref="emf" />
	</bean>

但是发现JPA的所有更新操作都失效了，只有查询是可以的。该回去就可以了，所以可以确定是JPA的事务配置问题。Google了一下，发现Springside提供了Spring多事务管理器配置的模板工程：[spring对多个transactionManager的支持已成熟，取消原来的显式定义](https://github.com/springside/springside4/blob/8ec78d2d5a09671c6eb8ec1001cb91a515f22072/examples/showcase/src/main/resources/applicationContext.xml)。照着配置一下就可以了：

		<!-- Configures Spring Data JPA and sets the base package of my DAOs. -->
		<jpa:repositories
			base-package="com.baidu.global.mobile.server.guanxing.monitor.repository"
			transaction-manager-ref="txManager" />

		<bean id="txManager" class="org.springframework.orm.jpa.JpaTransactionManager">
			<property name="entityManagerFactory" ref="emf" />
		</bean>

		<tx:annotation-driven transaction-manager="txManager"
			proxy-target-class="true" />


**注意** 一定要同时配置`<tx:annotation-driven  transaction-manager="txManager">`，否则会报如下错误：

	ErrorMessage: Could not open JPA EntityManager for transaction; nested exception is java.lang.IllegalStateException: Already value [org.springframework.jdbc.datasource.ConnectionHolder@11623292] for key [{ CreateTime:"2014-11-17 11:48:08", ActiveCount:2, PoolingCount:0, CreateCount:1, DestroyCount:0, CloseCount:4, ConnectCount:6, Connections:[ ] }] bound to thread [http-bio-80-exec-1]
