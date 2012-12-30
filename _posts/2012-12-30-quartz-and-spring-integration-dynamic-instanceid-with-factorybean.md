---
layout: post
title: Quartz与Spring的整合-使用Spring的FactoryBean实现动态Properties
---


## 问题

每个Quartz调度器实例都需要有一个标识，这是在配置`org.springframework.scheduling.quartz.SchedulerFactoryBean`的时候配置进去的。例如：

	<?xml version="1.0" encoding="GBK"?>
	<beans xmlns="http://www.springframework.org/schema/beans"
		xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
		xsi:schemaLocation="
	       http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-2.0.xsd">
	 
		<bean class="org.springframework.scheduling.quartz.SchedulerFactoryBean">
			<property name="quartzProperties">
			    <props>
			        <prop key="org.quartz.scheduler.instanceId">a static instance Id</prop>
					<prop key="org.quartz.jobStore.dataSource">FuwuQuartzDataSource</prop>
					<prop key="org.quartz.dataSource.FuwuQuartzDataSource.driver">${jdbc.driverClassName}</prop>
					<prop key="org.quartz.dataSource.FuwuQuartzDataSource.URL">${jdbc.url}</prop>
					<prop key="org.quartz.dataSource.FuwuQuartzDataSource.user">${jdbc.username}</prop>
					<prop key="org.quartz.dataSource.FuwuQuartzDataSource.password">${jdbc.password}</prop>
					<prop key="org.quartz.jobStore.tablePrefix">T_APP_QRTZ_</prop>
					<prop key="org.quartz.jobStore.isClustered">true</prop>
					<prop key="org.quartz.jobStore.driverDelegateClass">org.quartz.impl.jdbcjobstore.MSSQLDelegate</prop>
					<prop key="org.quartz.jobStore.class">org.quartz.impl.jdbcjobstore.JobStoreTX</prop>
				</props>
			</property>
		</bean>
	</beans>


[Quartz Enterprise Job Scheduler 1.x Configuration Reference](http://quartz-scheduler.org/documentation/quartz-1.x/configuration/ConfigMain)
 

#### org.quartz.scheduler.instanceName 

每个 Scheduler 必须给定一个名称来标识。当在同一个程序中有多个实例时，这个名称作为客户代码识别是哪个 Scheduler 而用。假如你用到了集群特性，你就必须为集群中的每一个实例使用相同的名称，以使它们成为“逻辑上” 是同一个 Scheduler 。

#### org.quartz.scheduler.instanceId

每个 Quartz Scheduler 必须指定一个唯一的 ID。这个值可以是任何字符串值，只要对于所有的 Scheduler 是唯一的。如果你想要自动生成的 ID，那你可以使用 AUTO 作为 instanceId 。从版本 1.5.1 开始，你能够定制如何自动生成实例 ID。见 instanceIDGenerator.class 属性，会在接下来讲到。

#### org.quartz.scheduler.instanceIdGenerator.class 

从版本 1.5.1 开始，这个属性允许你定制instanceId 的生成，这个属性仅被用于属性 org.quartz.scheduler.instanceId 设置为 AUTO 的情况下。默认是 org.quartz.simpl.SimpleInstanceIdGenerator ，它会基于主机名和时间戳来产生实例 ID 的。

关键在于这段配置：

     <prop key="org.quartz.scheduler.instanceId">a static instance Id</prop>

这是指定的静态ID，如果设置为AUTO，那么默认使用主机名和时间戳来产生实例ID，在大部分情况下是满足要求的。但是一般来说对于运行机，主机名一般没有什么用处，IP更重要，如何让IP作为实例ID呢？

## 解决方案

1. 使用`org.quartz.scheduler.instanceIdGenerator.class`
自定义一个instanceIdGenerator，然后配置`org.quartz.scheduler.instanceIdGenerator.class`为该自定义idGenerator，即可。

2. 使用`FactoryBean`机制

`FactoryBean`可以让你`狸猫换太子`。虽然指定的class为factoryBean的类路径，但是Spring实例化该factoryBean之后，还会调用factoryBean的工厂方法，返回真正的bean。

关于Spring的几种DI方式，这篇博文写的非常好，强烈推荐仔细阅读 [Are You Using The Full Power Of Spring When Injecting Your Dependencies?](http://www.skorks.com/2008/10/are-you-using-the-full-power-of-spring-when-injecting-your-dependencies/)

简单来说，Spring提供了几种方式让我们使用FactoryBean机制：

1. 使用`factory-method`配置
2. 实现`FactoryBean`接口
3. 使用`factory-bean`和`factory-method`配置

这里我们采用第1种方式：

	package me.arganzheng.study.quartz.task;
	
	import java.net.Inet4Address;
	import java.net.InetAddress;
	import java.net.NetworkInterface;
	import java.util.Enumeration;
	import java.util.List;
	import java.util.Properties;
	
	import org.apache.commons.collections.CollectionUtils;
	import org.apache.commons.io.IOUtils;
	import org.apache.commons.lang.StringUtils;
	import org.apache.log4j.Logger;
	
	/**
	 * 根据IP和机器名生成Quartz服务器的ID。
	 * 
	 * @author arganzheng
	 */
	public class QuartzSchedulerInstanceIdGenerator {
		private static final Logger log = Logger.getLogger(QuartzSchedulerInstanceIdGenerator.class);
	
		public static Properties generateInstanceIfForCurrentMachine(Properties properties, String instanceIdKey){
			String id = null;
	
			if(isWindows()){ // 有点恶，不过我们现在开发机都是windows.囧zr
				id = getHostName();
			}else{
				id = getIp();
			}
	
			if(id == null){
				id = "QRTZ-Sched-" + System.nanoTime();
			}
	
			log.info("Quartz Scheduler " + id + " is starting...");
	
			properties.setProperty(instanceIdKey, id);
			return properties;
		}
	
		@SuppressWarnings("unchecked")
		public static String getHostName(){
			String hostName = null;
			try{
				List<String> lines = IOUtils.readLines(
						Runtime.getRuntime().exec(
								isWindows() ? "cmd.exe /c \"set computername\"" : "hostname"
							).getInputStream()
						);
				if(CollectionUtils.isNotEmpty(lines) && StringUtils.isNotBlank(lines.get(0))){
					hostName = lines.get(0);
	
					if(hostName.indexOf('=') > 0){
						hostName = hostName.substring(hostName.indexOf('=') + 1);
					}
				}
			}catch(Throwable ex){
				log.error("获取机器名失败！", ex);
			}
	
			return hostName;
		}
	
		private static boolean isWindows(){
			return StringUtils.indexOfIgnoreCase(System.getProperty("os.name"), "Windows") >= 0;
		}
	
		public static String getIp(){
			try {
				Enumeration<NetworkInterface> nis = NetworkInterface.getNetworkInterfaces();
				while(nis.hasMoreElements()){
					NetworkInterface ni = nis.nextElement();
					if(!ni.isUp()) continue;
					if(ni.isLoopback()) continue;
					if(ni.isPointToPoint()) continue;
					if(ni.isVirtual()) continue;
	
					Enumeration<InetAddress> ias = ni.getInetAddresses();
					while(ias.hasMoreElements()){
						InetAddress ia = ias.nextElement();
						if(ia instanceof Inet4Address){
							return ia.getHostAddress();
						}
					}
				}
			} catch (Exception ex) {
				log.error("获取IP失败！", ex);
			}
	
			return null;
		}
	}

这里我们判断如果是windows环境就使用hostname，否则使用ip作为instanceId。需要注意的是该工厂方法必须返回Properties类对象，否者会运行时候出错。

然后在Spring中配置Quartz的入口：

	<?xml version="1.0" encoding="GBK"?>
	<beans xmlns="http://www.springframework.org/schema/beans"
		xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
		xsi:schemaLocation="
	       http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-2.0.xsd">
	 
		<bean class="org.springframework.scheduling.quartz.SchedulerFactoryBean">
			<property name="quartzProperties">
				<bean class="me.arganzheng.study.quartz.task.QuartzSchedulerInstanceIdGenerator" factory-method="generateInstanceIfForCurrentMachine">
					<constructor-arg>
						<props>
							<prop key="org.quartz.jobStore.dataSource">FuwuQuartzDataSource</prop>
							<prop key="org.quartz.dataSource.FuwuQuartzDataSource.driver">${jdbc.driverClassName}</prop>
							<prop key="org.quartz.dataSource.FuwuQuartzDataSource.URL">${jdbc.url}</prop>
							<prop key="org.quartz.dataSource.FuwuQuartzDataSource.user">${jdbc.username}</prop>
							<prop key="org.quartz.dataSource.FuwuQuartzDataSource.password">${jdbc.password}</prop>
							<prop key="org.quartz.jobStore.tablePrefix">T_APP_QRTZ_</prop>
							<prop key="org.quartz.jobStore.isClustered">true</prop>
							<prop key="org.quartz.jobStore.driverDelegateClass">org.quartz.impl.jdbcjobstore.MSSQLDelegate</prop>
							<prop key="org.quartz.jobStore.class">org.quartz.impl.jdbcjobstore.JobStoreTX</prop>
						</props>
					</constructor-arg>
					<constructor-arg>
						<value>org.quartz.scheduler.instanceId</value>
					</constructor-arg>
				</bean>
			</property>
		</bean>
	</beans>

可以看到我们配置了两个`	<constructor-arg>`给了`me.arganzheng.study.quartz.task.QuartzSchedulerInstanceIdGenerator`，这两个构造参数都会传递给指定的factory-method。


## 其他用途

比如上面的dataSource配置，其实也可以使用这种方式动态返回不同的数据源配置。如dev环境，test环境和production环境。

--EOF--
