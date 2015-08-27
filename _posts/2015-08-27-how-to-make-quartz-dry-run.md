---
title: 如何让一个Quartz实例不执行任务
layout: post
---


Quartz默认的集群方式每个节点（实例）都是对等，都可以创建任务，都会去抢任务。但是有时候我们希望能够有不对等的结构，比如运营后台提供友好的界面管理任务，然后有一些肉机，专门执行任务。这样的好处，任务不会影响到管理后台，管理后台的发布也不会中断执行中的任务。

谷歌和翻看官方文档都没有发现这个配置，只能查看源码了。最后发现有一个类看起来就是为这个实现的：

	package org.quartz.simpl;

	import org.slf4j.Logger;
	import org.slf4j.LoggerFactory;
	import org.quartz.SchedulerConfigException;
	import org.quartz.spi.ThreadPool;

	/**
	 * <p>
	 * This is class is a simple implementation of a zero size thread pool, based on the
	 * <code>{@link org.quartz.spi.ThreadPool}</code> interface.
	 * </p>
	 * 
	 * <p>
	 * The pool has zero <code>Thread</code>s and does not grow or shrink based on demand.
	 * Which means it is obviously not useful for most scenarios.  When it may be useful
	 * is to prevent creating any worker threads at all - which may be desirable for
	 * the sole purpose of preserving system resources in the case where the scheduler
	 * instance only exists in order to schedule jobs, but which will never execute
	 * jobs (e.g. will never have start() called on it).
	 * </p>
	 * 
	 * <p>
	 * </p>
	 * 
	 * @author Wayne Fay
	 */
	public class ZeroSizeThreadPool implements ThreadPool {

	    private final Logger log = LoggerFactory.getLogger(getClass());

	    public ZeroSizeThreadPool() {
	    }

	 
	    public Logger getLog() {
	        return log;
	    }

	    public int getPoolSize() {
	        return 0;
	    }

	    public void initialize() throws SchedulerConfigException {
	    }

	    public void shutdown() {
	        shutdown(true);
	    }

	    public void shutdown(boolean waitForJobsToComplete) {
	        getLog().debug("shutdown complete");
	    }

	    public boolean runInThread(Runnable runnable) {
	        throw new UnsupportedOperationException("This ThreadPool should not be used on Scheduler instances that are start()ed.");
	    }

	    public int blockForAvailableThreads() {
	        throw new UnsupportedOperationException("This ThreadPool should not be used on Scheduler instances that are start()ed.");
	    }

	    public void setInstanceId(String schedInstId) {
	    }

	    public void setInstanceName(String schedName) {
	    }

	}

ZeroSizeThreadPool，正如注释上所说的: `When it may be useful is to prevent creating any worker threads at all - which may be desirable for the sole purpose of preserving system resources in the case where the scheduler instance only exists in order to schedule jobs, but which will never execute jobs (e.g. will never have start() called on it).` 只schedule jobs, never execute jobs.

Perfect，看起来What exactly we need. 


相应的配置项是：[org.quartz.threadPool.class](http://quartz-scheduler.org/documentation/quartz-1.x/configuration/ConfigThreadPool)。于是spring-quartz.xml可以这么配置：


	<?xml version="1.0" encoding="GBK"?>
	<beans xmlns="http://www.springframework.org/schema/beans"
		xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
		xsi:schemaLocation="
	       http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-2.0.xsd">

		<bean class="org.springframework.scheduling.quartz.SchedulerFactoryBean">
			<property name="jobFactory">
				<bean
					class="com.baidu.global.mobile.server.guanxing.task.SpringBeanJobFactory" />
			</property>
			<property name="dataSource" ref="dataSource" />
			<property name="quartzProperties">
				<props>
					<prop key="org.quartz.jobStore.dataSource">GuanxingQuartzDataSource</prop>
					<prop key="org.quartz.jobStore.tablePrefix">QRTZ_</prop>
					<prop key="org.quartz.jobStore.isClustered">true</prop>
					<!-- 通过配置org.quartz.threadPool.class决定是否执行任务，还是只是创建任务。因为我们的Service 没有封装成二方库，所以没有办法拆分。 -->
					<prop key="org.quartz.threadPool.class">${org.quartz.threadPool.class}</prop>
					<prop key="org.quartz.threadPool.threadCount">${org.quartz.threadPool.threadCount}</prop>
					<prop key="org.quartz.jobStore.class">org.quartz.impl.jdbcjobstore.JobStoreTX</prop>
					<prop key="org.quartz.jobStore.driverDelegateClass">org.quartz.impl.jdbcjobstore.StdJDBCDelegate</prop>
					<prop key="org.quartz.scheduler.jobFactory.class">com.baidu.global.mobile.server.guanxing.task.SpringBeanJobFactory
					</prop>
					<prop key="org.quartz.scheduler.instanceId">AUTO</prop>
					<prop key="org.quartz.scheduler.instanceIdGenerator.class">com.baidu.global.mobile.server.guanxing.task.QuartzSchedulerInstanceIdGenerator
					</prop>
				</props>
			</property>
			<property name="globalJobListeners">
				<list>
					<bean
						class="com.baidu.global.mobile.server.guanxing.task.QuartzEventListener" />
				</list>
			</property>
		</bean>
	</beans>

其实最好是把job实现代码给拆分成一个单独的工程，但是由于我们的job实现很多引用了service方法，然后这些service方法并没有封装成二方库。所以简单的实现就是同一套代码，不同的配置项。配置文件还是通过env环境变量区分:

	org.quartz.threadPool.class=org.quartz.simpl.ZeroSizeThreadPool
	org.quartz.threadPool.threadCount=0

不过启动还是报错了：

	[ERROR]	2015-08-26 19:27:55,282	[main] org.springframework.web.context.ContextLoader (ContextLoader.java:331) - Context initialization failed
	org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'messageTaskFacade': Injection of autowired dependencies failed; nested exception is org.springframework.beans.factory.BeanCreationException: Could not autowire field: private org.quartz.Scheduler com.baidu.global.mobile.server.guanxing.facade.MessageTaskFacade.scheduler; nested exception is org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'org.springframework.scheduling.quartz.SchedulerFactoryBean#0' defined in class path resource [conf-spring/spring-quartz.xml]: Invocation of init method failed; nested exception is org.quartz.SchedulerException: ThreadPool class 'org.quartz.simpl.ZeroSizeThreadPool' props could not be configured. [See nested exception: java.lang.NoSuchMethodException: No setter for property 'threadCount']
		at org.springframework.beans.factory.annotation.AutowiredAnnotationBeanPostProcessor.postProcessPropertyValues(AutowiredAnnotationBeanPostProcessor.java:289)
		at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.populateBean(AbstractAutowireCapableBeanFactory.java:1146)
		at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.doCreateBean(AbstractAutowireCapableBeanFactory.java:519)
		at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.createBean(AbstractAutowireCapableBeanFactory.java:458)
		at org.springframework.beans.factory.support.AbstractBeanFactory$1.getObject(AbstractBeanFactory.java:296)
		at org.springframework.beans.factory.support.DefaultSingletonBeanRegistry.getSingleton(DefaultSingletonBeanRegistry.java:223)
		at org.springframework.beans.factory.support.AbstractBeanFactory.doGetBean(AbstractBeanFactory.java:293)
		at org.springframework.beans.factory.support.AbstractBeanFactory.getBean(AbstractBeanFactory.java:194)
		at org.springframework.beans.factory.support.DefaultListableBeanFactory.preInstantiateSingletons(DefaultListableBeanFactory.java:633)
		at org.springframework.context.support.AbstractApplicationContext.finishBeanFactoryInitialization(AbstractApplicationContext.java:932)
		at org.springframework.context.support.AbstractApplicationContext.refresh(AbstractApplicationContext.java:479)
		at org.springframework.web.context.ContextLoader.configureAndRefreshWebApplicationContext(ContextLoader.java:410)
		at org.springframework.web.context.ContextLoader.initWebApplicationContext(ContextLoader.java:306)
		at org.springframework.web.context.ContextLoaderListener.contextInitialized(ContextLoaderListener.java:112)
		at org.apache.catalina.core.StandardContext.listenerStart(StandardContext.java:4210)
		at org.apache.catalina.core.StandardContext.start(StandardContext.java:4709)
		at org.apache.catalina.core.ContainerBase.addChildInternal(ContainerBase.java:799)
		at org.apache.catalina.core.ContainerBase.addChild(ContainerBase.java:779)
		at org.apache.catalina.core.StandardHost.addChild(StandardHost.java:583)
		at org.apache.catalina.startup.HostConfig.deployWAR(HostConfig.java:943)
		at org.apache.catalina.startup.HostConfig.deployWARs(HostConfig.java:778)
		at org.apache.catalina.startup.HostConfig.deployApps(HostConfig.java:504)
		at org.apache.catalina.startup.HostConfig.start(HostConfig.java:1317)
		at org.apache.catalina.startup.HostConfig.lifecycleEvent(HostConfig.java:324)
		at org.apache.catalina.util.LifecycleSupport.fireLifecycleEvent(LifecycleSupport.java:142)
		at org.apache.catalina.core.ContainerBase.start(ContainerBase.java:1065)
		at org.apache.catalina.core.StandardHost.start(StandardHost.java:822)
		at org.apache.catalina.core.ContainerBase.start(ContainerBase.java:1057)
		at org.apache.catalina.core.StandardEngine.start(StandardEngine.java:463)
		at org.apache.catalina.core.StandardService.start(StandardService.java:525)
		at org.apache.catalina.core.StandardServer.start(StandardServer.java:754)
		at org.apache.catalina.startup.Catalina.start(Catalina.java:595)
		at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
		at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
		at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
		at java.lang.reflect.Method.invoke(Method.java:483)
		at org.apache.catalina.startup.Bootstrap.start(Bootstrap.java:289)
		at org.apache.catalina.startup.Bootstrap.main(Bootstrap.java:414)
	Caused by: org.springframework.beans.factory.BeanCreationException: Could not autowire field: private org.quartz.Scheduler com.baidu.global.mobile.server.guanxing.facade.MessageTaskFacade.scheduler; nested exception is org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'org.springframework.scheduling.quartz.SchedulerFactoryBean#0' defined in class path resource [conf-spring/spring-quartz.xml]: Invocation of init method failed; nested exception is org.quartz.SchedulerException: ThreadPool class 'org.quartz.simpl.ZeroSizeThreadPool' props could not be configured. [See nested exception: java.lang.NoSuchMethodException: No setter for property 'threadCount']
		at org.springframework.beans.factory.annotation.AutowiredAnnotationBeanPostProcessor$AutowiredFieldElement.inject(AutowiredAnnotationBeanPostProcessor.java:517)
		at org.springframework.beans.factory.annotation.InjectionMetadata.inject(InjectionMetadata.java:87)
		at org.springframework.beans.factory.annotation.AutowiredAnnotationBeanPostProcessor.postProcessPropertyValues(AutowiredAnnotationBeanPostProcessor.java:286)
		... 37 more
	Caused by: org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'org.springframework.scheduling.quartz.SchedulerFactoryBean#0' defined in class path resource [conf-spring/spring-quartz.xml]: Invocation of init method failed; nested exception is org.quartz.SchedulerException: ThreadPool class 'org.quartz.simpl.ZeroSizeThreadPool' props could not be configured. [See nested exception: java.lang.NoSuchMethodException: No setter for property 'threadCount']
		at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.initializeBean(AbstractAutowireCapableBeanFactory.java:1512)
		at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.doCreateBean(AbstractAutowireCapableBeanFactory.java:521)
		at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.createBean(AbstractAutowireCapableBeanFactory.java:458)
		at org.springframework.beans.factory.support.AbstractBeanFactory$1.getObject(AbstractBeanFactory.java:296)
		at org.springframework.beans.factory.support.DefaultSingletonBeanRegistry.getSingleton(DefaultSingletonBeanRegistry.java:223)
		at org.springframework.beans.factory.support.AbstractBeanFactory.doGetBean(AbstractBeanFactory.java:293)
		at org.springframework.beans.factory.support.AbstractBeanFactory.getBean(AbstractBeanFactory.java:194)
		at org.springframework.beans.factory.support.DefaultListableBeanFactory.findAutowireCandidates(DefaultListableBeanFactory.java:917)
		at org.springframework.beans.factory.support.DefaultListableBeanFactory.doResolveDependency(DefaultListableBeanFactory.java:860)
		at org.springframework.beans.factory.support.DefaultListableBeanFactory.resolveDependency(DefaultListableBeanFactory.java:775)
		at org.springframework.beans.factory.annotation.AutowiredAnnotationBeanPostProcessor$AutowiredFieldElement.inject(AutowiredAnnotationBeanPostProcessor.java:489)
		... 39 more
	Caused by: org.quartz.SchedulerException: ThreadPool class 'org.quartz.simpl.ZeroSizeThreadPool' props could not be configured. [See nested exception: java.lang.NoSuchMethodException: No setter for property 'threadCount']
		at org.quartz.impl.StdSchedulerFactory.instantiate(StdSchedulerFactory.java:851)
		at org.quartz.impl.StdSchedulerFactory.getScheduler(StdSchedulerFactory.java:1519)
		at org.springframework.scheduling.quartz.SchedulerFactoryBean.createScheduler(SchedulerFactoryBean.java:593)
		at org.springframework.scheduling.quartz.SchedulerFactoryBean.afterPropertiesSet(SchedulerFactoryBean.java:476)
		at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.invokeInitMethods(AbstractAutowireCapableBeanFactory.java:1571)
		at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.initializeBean(AbstractAutowireCapableBeanFactory.java:1509)
		... 49 more
	Caused by: java.lang.NoSuchMethodException: No setter for property 'threadCount'
		at org.quartz.impl.StdSchedulerFactory.setBeanProps(StdSchedulerFactory.java:1407)
		at org.quartz.impl.StdSchedulerFactory.instantiate(StdSchedulerFactory.java:849)
		... 54 more


根据堆栈，原来Quartz的配置项注入方式比较简单粗暴：是根据配置项反射注入Bean属性来的。因为我配置了`org.quartz.threadPool.threadCount`，但是ZeroSizeThreadPool并没有这个属性，所以就报错了。但是这个配置项又不好去掉，去掉意味着线程池是默认的10个。在有些情况下我们还是希望能够对这个配置项进行调优的。

不过最神奇的是即使把`org.quartz.threadPool.threadCount`配置项去掉，还是报同样的错误。。断点调试发现，如果你没有配置某个配置项目，那么Quartz会把默认配置项作为Key去bean中查找对应的属性的。而`org.quartz.threadPool.threadCount`默认就是10。所以不管你有没有配置这个配置项都是会出错的。。也就是说这个`ZeroSizeThreadPool`其实就是一个摆设品，根本没法正常使用。

那么最简单就是extends ZeroSizeThreadPool，增加threadCount属性。现在能够正常启动了，但是一启动就大量报错：

	[ERROR]	2015-08-27 09:34:54,700	[org.springframework.scheduling.quartz.SchedulerFactoryBean#0_QuartzSchedulerThread] org.quartz.core.QuartzSchedulerThread (QuartzSchedulerThread.java:418) - Runtime error occurred in main trigger firing loop.
	java.lang.UnsupportedOperationException: This ThreadPool should not be used on Scheduler instances that are start()ed.
		at com.baidu.global.mobile.server.guanxing.task.MyZeroSizeThreadPool.blockForAvailableThreads(MyZeroSizeThreadPool.java:42)
		at org.quartz.core.QuartzSchedulerThread.run(QuartzSchedulerThread.java:263)

这是因为QuartzSchedulerThread被启动了，然后这个线程会不断的去DB获取即将要执行的triggers，然后扔给线程池去执行：

	package org.quartz.core;

	/**
	 * <p>
	 * The thread responsible for performing the work of firing <code>{@link Trigger}</code>
	 * s that are registered with the <code>{@link QuartzScheduler}</code>.
	 * </p>
	 */
	public class QuartzSchedulerThread extends Thread {

	    private QuartzScheduler qs;

	    private QuartzSchedulerResources qsRsrcs;

	    private final Object sigLock = new Object();

	    private boolean signaled;
	    private long signaledNextFireTime;

	    private boolean paused;

	    private AtomicBoolean halted;

	    private Random random = new Random(System.currentTimeMillis());

	    // When the scheduler finds there is no current trigger to fire, how long
	    // it should wait until checking again...
	    private static long DEFAULT_IDLE_WAIT_TIME = 30L * 1000L;

	    private long idleWaitTime = DEFAULT_IDLE_WAIT_TIME;

	    private int idleWaitVariablness = 7 * 1000;


	    /**
	     * <p>
	     * The main processing loop of the <code>QuartzSchedulerThread</code>.
	     * </p>
	     */
	    @Override
	    public void run() {
	        boolean lastAcquireFailed = false;

	        while (!halted.get()) {
	            try {
	                // check if we're supposed to pause...
	                /// ....

	                int availThreadCount = qsRsrcs.getThreadPool().blockForAvailableThreads();
	                if(availThreadCount > 0) { // will always be true, due to semantics of blockForAvailableThreads...

	                    List<OperableTrigger> triggers = null;

	                    long now = System.currentTimeMillis();

	                    clearSignaledSchedulingChange();
	                    try {
	                        triggers = qsRsrcs.getJobStore().acquireNextTriggers(
	                                now + idleWaitTime, Math.min(availThreadCount, qsRsrcs.getMaxBatchSize()), qsRsrcs.getBatchTimeWindow());
	                        lastAcquireFailed = false;
	                    } catch (...){
	                        /// ...
	                    }

	                    if (triggers != null && !triggers.isEmpty()) {
	                        now = System.currentTimeMillis();
	                        long triggerTime = triggers.get(0).getNextFireTime().getTime();
	                        long timeUntilTrigger = triggerTime - now;
	                        while(timeUntilTrigger > 2) {
	                            synchronized (sigLock) {
	                                if (halted.get()) {
	                                    break;
	                                }
	                                if (!isCandidateNewTimeEarlierWithinReason(triggerTime, false)) {
	                                    try {
	                                        // we could have blocked a long while
	                                        // on 'synchronize', so we must recompute
	                                        now = System.currentTimeMillis();
	                                        timeUntilTrigger = triggerTime - now;
	                                        if(timeUntilTrigger >= 1)
	                                            sigLock.wait(timeUntilTrigger);
	                                    } catch (InterruptedException ignore) {
	                                    }
	                                }
	                            }
	                            if(releaseIfScheduleChangedSignificantly(triggers, triggerTime)) {
	                                break;
	                            }
	                            now = System.currentTimeMillis();
	                            timeUntilTrigger = triggerTime - now;
	                        }

	                        // this happens if releaseIfScheduleChangedSignificantly decided to release triggers
	                        if(triggers.isEmpty())
	                            continue;

	                        // set triggers to 'executing'
	                        List<TriggerFiredResult> bndles = new ArrayList<TriggerFiredResult>();

	                        boolean goAhead = true;
	                        synchronized(sigLock) {
	                            goAhead = !halted.get();
	                        }
	                        if(goAhead) {
	                            try {
	                                List<TriggerFiredResult> res = qsRsrcs.getJobStore().triggersFired(triggers);
	                                if(res != null)
	                                    bndles = res;
	                            } catch (SchedulerException se) {
	                                qs.notifySchedulerListenersError(
	                                        "An error occurred while firing triggers '"
	                                                + triggers + "'", se);
	                                //QTZ-179 : a problem occurred interacting with the triggers from the db
	                                //we release them and loop again
	                                for (int i = 0; i < triggers.size(); i++) {
	                                    qsRsrcs.getJobStore().releaseAcquiredTrigger(triggers.get(i));
	                                }
	                                continue;
	                            }

	                        }

	                        for (int i = 0; i < bndles.size(); i++) {
	                            TriggerFiredResult result =  bndles.get(i);
	                            TriggerFiredBundle bndle =  result.getTriggerFiredBundle();
	                            Exception exception = result.getException();

	                            if (exception instanceof RuntimeException) {
	                                getLog().error("RuntimeException while firing trigger " + triggers.get(i), exception);
	                                qsRsrcs.getJobStore().releaseAcquiredTrigger(triggers.get(i));
	                                continue;
	                            }

	                            // it's possible to get 'null' if the triggers was paused,
	                            // blocked, or other similar occurrences that prevent it being
	                            // fired at this time...  or if the scheduler was shutdown (halted)
	                            if (bndle == null) {
	                                qsRsrcs.getJobStore().releaseAcquiredTrigger(triggers.get(i));
	                                continue;
	                            }

	                            JobRunShell shell = null;
	                            try {
	                                shell = qsRsrcs.getJobRunShellFactory().createJobRunShell(bndle);
	                                shell.initialize(qs);
	                            } catch (SchedulerException se) {
	                                qsRsrcs.getJobStore().triggeredJobComplete(triggers.get(i), bndle.getJobDetail(), CompletedExecutionInstruction.SET_ALL_JOB_TRIGGERS_ERROR);
	                                continue;
	                            }

	                            if (qsRsrcs.getThreadPool().runInThread(shell) == false) {
	                                // this case should never happen, as it is indicative of the
	                                // scheduler being shutdown or a bug in the thread pool or
	                                // a thread pool being used concurrently - which the docs
	                                // say not to do...
	                                getLog().error("ThreadPool.runInThread() return false!");
	                                qsRsrcs.getJobStore().triggeredJobComplete(triggers.get(i), bndle.getJobDetail(), CompletedExecutionInstruction.SET_ALL_JOB_TRIGGERS_ERROR);
	                            }

	                        }

	                        continue; // while (!halted)
	                    }
	                } else { // if(availThreadCount > 0)
	                    // should never happen, if threadPool.blockForAvailableThreads() follows contract
	                    continue; // while (!halted)
	                }

	                long now = System.currentTimeMillis();
	                long waitTime = now + getRandomizedIdleWaitTime();
	                long timeUntilContinue = waitTime - now;
	                synchronized(sigLock) {
	                    try {
	                      if(!halted.get()) {
	                        // QTZ-336 A job might have been completed in the mean time and we might have
	                        // missed the scheduled changed signal by not waiting for the notify() yet
	                        // Check that before waiting for too long in case this very job needs to be
	                        // scheduled very soon
	                        if (!isScheduleChanged()) {
	                          sigLock.wait(timeUntilContinue);
	                        }
	                      }
	                    } catch (InterruptedException ignore) {
	                    }
	                }

	            } catch(RuntimeException re) {
	                getLog().error("Runtime error occurred in main trigger firing loop.", re);
	            }
	        } // while (!halted)

	        // drop references to scheduler stuff to aid garbage collection...
	        qs = null;
	        qsRsrcs = null;
	    }
	}

关键代码在这里：

	while (!halted.get()) {
        try {
            // check if we're supposed to pause...
            /// ....

            int availThreadCount = qsRsrcs.getThreadPool().blockForAvailableThreads();
            if(availThreadCount > 0) { // will always be true, due to semantics of blockForAvailableThreads...

    	 	} else { // if(availThreadCount > 0)
                // should never happen, if threadPool.blockForAvailableThreads() follows contract
                continue; // while (!halted)
            }
		｝
		...
	｝

虽然，我们可以让ZeroSizeThreadPool的blockForAvailableThreads()方法返回0，而不是抛异常。但是仍然会导致QuartzSchedulerThread不断的去扫描数据库获取即将要执行的triggers，对CPU和DB都是一种浪费。有没有更好的方式呢？可以直接就让Quartz不要启动QuartzSchedulerThread线程了？

查看一下源码：其实QuartzSchedulerThread是在这里启动的：

	package org.quartz.core;

	/**
	 * <p>
	 * This is the heart of Quartz, an indirect implementation of the <code>{@link org.quartz.Scheduler}</code>
	 * interface, containing methods to schedule <code>{@link org.quartz.Job}</code>s,
	 * register <code>{@link org.quartz.JobListener}</code> instances, etc.
	 * </p>
	 * 
	 */
	public class QuartzScheduler implements RemotableQuartzScheduler {

	    public QuartzScheduler(QuartzSchedulerResources resources, long idleWaitTime, @Deprecated long dbRetryInterval)
	        throws SchedulerException {
	        this.resources = resources;
	        if (resources.getJobStore() instanceof JobListener) {
	            addInternalJobListener((JobListener)resources.getJobStore());
	        }

	        this.schedThread = new QuartzSchedulerThread(this, resources);
	        ThreadExecutor schedThreadExecutor = resources.getThreadExecutor();
	        schedThreadExecutor.execute(this.schedThread);
	        if (idleWaitTime > 0) {
	            this.schedThread.setIdleWaitTime(idleWaitTime);
	        }

	        jobMgr = new ExecutingJobsManager();
	        addInternalJobListener(jobMgr);
	        errLogger = new ErrorLogger();
	        addInternalSchedulerListener(errLogger);

	        signaler = new SchedulerSignalerImpl(this, this.schedThread);

	        if(shouldRunUpdateCheck()) 
	            updateTimer = scheduleUpdateCheck();
	        else
	            updateTimer = null;

	        getLog().info("Quartz Scheduler v." + getVersion() + " created.");
	    }
	}

这里先创建了一个QuartzSchedulerThread，然后扔到ThreadExecutor中执行：

 	this.schedThread = new QuartzSchedulerThread(this, resources);
    ThreadExecutor schedThreadExecutor = resources.getThreadExecutor();
    schedThreadExecutor.execute(this.schedThread);

而默认的ThreadExecutor实现是：


	package org.quartz.impl;

	import org.quartz.spi.ThreadExecutor;

	/**
	 * Schedules work on a newly spawned thread. This is the default Quartz
	 * behavior.
	 *
	 * @author matt.accola
	 * @version $Revision$ $Date$
	 */
	public class DefaultThreadExecutor implements ThreadExecutor {

	    public void initialize() {
	    }

	    public void execute(Thread thread) {
	        thread.start();
	    }

	}

可以看到execute方法就是将thread启动。那么一个解决方案就来了:

	package com.baidu.global.mobile.server.guanxing.task;

	import org.quartz.spi.ThreadExecutor;

	/**
	 * Schedules work on a newly spawned thread. This is the default Quartz behavior.
	 *
	 * @author matt.accola
	 * @version $Revision$ $Date$
	 */
	public class DryRunThreadExecutor implements ThreadExecutor {

	    public void initialize() {
	    }

	    public void execute(Thread thread) {
	        // do nothing;
	    }

	}

不过查看了一个官方文档并没有提供executor的配置项，但是翻看源码确实有的：

	public class StdSchedulerFactory implements SchedulerFactory {

	    public static final String PROP_THREAD_EXECUTOR = "org.quartz.threadExecutor";

    	public static final String PROP_THREAD_EXECUTOR_CLASS = "org.quartz.threadExecutor.class";

		private Scheduler instantiate() throws SchedulerException {
			...
		 	// Get ThreadExecutor Properties
	        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	        String threadExecutorClass = cfg.getStringProperty(PROP_THREAD_EXECUTOR_CLASS);
	        if (threadExecutorClass != null) {
	            tProps = cfg.getPropertyGroup(PROP_THREAD_EXECUTOR, true);
	            try {
	                threadExecutor = (ThreadExecutor) loadHelper.loadClass(threadExecutorClass).newInstance();
	                log.info("Using custom implementation for ThreadExecutor: " + threadExecutorClass);

	                setBeanProps(threadExecutor, tProps);
	            } catch (Exception e) {
	                initException = new SchedulerException(
	                        "ThreadExecutor class '" + threadExecutorClass + "' could not be instantiated.", e);
	                throw initException;
	            }
	        } else {
	            log.info("Using default implementation for ThreadExecutor");
	            threadExecutor = new DefaultThreadExecutor();
	        }
	        ...
    	}
	}


所以于是配置一下：

	<prop key="org.quartz.threadExecutor.class">com.baidu.global.mobile.server.guanxing.task.DryRunThreadExecutor</prop>

果然整个世界清静很多：）

最后整个配置如下：

spring-quartz.xml:

	<?xml version="1.0" encoding="GBK"?>
	<beans xmlns="http://www.springframework.org/schema/beans"
		xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
		xsi:schemaLocation="
	       http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-2.0.xsd">

		<bean class="org.springframework.scheduling.quartz.SchedulerFactoryBean">
			<property name="jobFactory">
				<bean
					class="com.baidu.global.mobile.server.guanxing.task.SpringBeanJobFactory" />
			</property>
			<property name="dataSource" ref="dataSource" />
			<property name="quartzProperties">
				<props>
					<prop key="org.quartz.jobStore.dataSource">GuanxingQuartzDataSource</prop>
					<prop key="org.quartz.jobStore.tablePrefix">QRTZ_</prop>
					<prop key="org.quartz.jobStore.isClustered">${org.quartz.jobStore.isClustered}</prop>
					<!-- 通过配置 threadPool和threadExecutor 决定是否执行任务，还是只是创建任务。因为我们的Service 没有封装成二方库，所以没有办法拆分。 -->
					<prop key="org.quartz.threadExecutor.class">${org.quartz.threadExecutor.class}</prop>
					<prop key="org.quartz.threadPool.class">${org.quartz.threadPool.class}</prop>
					<prop key="org.quartz.threadPool.threadCount">${org.quartz.threadPool.threadCount}</prop>
					<prop key="org.quartz.jobStore.class">org.quartz.impl.jdbcjobstore.JobStoreTX</prop>
					<prop key="org.quartz.jobStore.driverDelegateClass">org.quartz.impl.jdbcjobstore.StdJDBCDelegate</prop>
					<prop key="org.quartz.scheduler.jobFactory.class">com.baidu.global.mobile.server.guanxing.task.SpringBeanJobFactory</prop>
					<prop key="org.quartz.scheduler.instanceId">AUTO</prop>
					<prop key="org.quartz.scheduler.instanceIdGenerator.class">com.baidu.global.mobile.server.guanxing.task.QuartzSchedulerInstanceIdGenerator
					</prop>
				</props>
			</property>
			<property name="globalJobListeners">
				<list>
					<bean
						class="com.baidu.global.mobile.server.guanxing.task.QuartzEventListener" />
				</list>
			</property>
		</bean>
	</beans>

对于运营后台，配置如下：

	org.quartz.jobStore.isClustered=true
	org.quartz.threadExecutor.class=com.baidu.global.mobile.server.guanxing.quartz.DryRunThreadExecutor
	org.quartz.threadPool.class= com.baidu.global.mobile.server.guanxing.quartz.MyZeroSizeThreadPool
	org.quartz.threadPool.threadCount=0

对于执行机，配置如下：

	org.quartz.jobStore.isClustered=true
	org.quartz.threadExecutor.class=org.quartz.impl.DefaultThreadExecutor
	org.quartz.threadPool.class=org.quartz.simpl.SimpleThreadPool
	org.quartz.threadPool.threadCount=12

需要注意的是，org.quartz.jobStore.isClustered一定要配置成true，否则Quartz在启动时候尝试job recovery：	

	[ERROR]	2015-08-27 10:29:19,909	[main] org.springframework.web.context.ContextLoader (ContextLoader.java:331) - Context initialization failed
	org.springframework.context.ApplicationContextException: Failed to start bean 'org.springframework.scheduling.quartz.SchedulerFactoryBean#0'; nested exception is org.springframework.scheduling.SchedulingException: Could not start Quartz Scheduler; nested exception is org.quartz.SchedulerConfigException: Failure occured during job recovery. [See nested exception: org.quartz.JobPersistenceException: Couldn't store trigger 'monitor-api.ApiTestCaseJob-81' for 'monitor-api.ApiTestCaseJob-81' job:com.baidu.global.mobile.server.guanxing.monitor.job.ApiTestCaseJob [See nested exception: java.lang.ClassNotFoundException: com.baidu.global.mobile.server.guanxing.monitor.job.ApiTestCaseJob]]
		at org.springframework.context.support.DefaultLifecycleProcessor.doStart(DefaultLifecycleProcessor.java:170)
		at org.springframework.context.support.DefaultLifecycleProcessor.access$200(DefaultLifecycleProcessor.java:51)
		at org.springframework.context.support.DefaultLifecycleProcessor$LifecycleGroup.start(DefaultLifecycleProcessor.java:339)
		at org.springframework.context.support.DefaultLifecycleProcessor.startBeans(DefaultLifecycleProcessor.java:143)
		at org.springframework.context.support.DefaultLifecycleProcessor.onRefresh(DefaultLifecycleProcessor.java:108)
		at org.springframework.context.support.AbstractApplicationContext.finishRefresh(AbstractApplicationContext.java:945)
		at org.springframework.context.support.AbstractApplicationContext.refresh(AbstractApplicationContext.java:482)
		at org.springframework.web.context.ContextLoader.configureAndRefreshWebApplicationContext(ContextLoader.java:410)
		at org.springframework.web.context.ContextLoader.initWebApplicationContext(ContextLoader.java:306)
		at org.springframework.web.context.ContextLoaderListener.contextInitialized(ContextLoaderListener.java:112)
		at org.apache.catalina.core.StandardContext.listenerStart(StandardContext.java:4210)
		at org.apache.catalina.core.StandardContext.start(StandardContext.java:4709)
		at org.apache.catalina.core.ContainerBase.addChildInternal(ContainerBase.java:799)
		at org.apache.catalina.core.ContainerBase.addChild(ContainerBase.java:779)
		at org.apache.catalina.core.StandardHost.addChild(StandardHost.java:583)
		at org.apache.catalina.startup.HostConfig.deployWAR(HostConfig.java:943)
		at org.apache.catalina.startup.HostConfig.deployWARs(HostConfig.java:778)
		at org.apache.catalina.startup.HostConfig.deployApps(HostConfig.java:504)
		at org.apache.catalina.startup.HostConfig.start(HostConfig.java:1317)
		at org.apache.catalina.startup.HostConfig.lifecycleEvent(HostConfig.java:324)
		at org.apache.catalina.util.LifecycleSupport.fireLifecycleEvent(LifecycleSupport.java:142)
		at org.apache.catalina.core.ContainerBase.start(ContainerBase.java:1065)
		at org.apache.catalina.core.StandardHost.start(StandardHost.java:822)
		at org.apache.catalina.core.ContainerBase.start(ContainerBase.java:1057)
		at org.apache.catalina.core.StandardEngine.start(StandardEngine.java:463)
		at org.apache.catalina.core.StandardService.start(StandardService.java:525)
		at org.apache.catalina.core.StandardServer.start(StandardServer.java:754)
		at org.apache.catalina.startup.Catalina.start(Catalina.java:595)
		at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
		at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
		at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
		at java.lang.reflect.Method.invoke(Method.java:483)
		at org.apache.catalina.startup.Bootstrap.start(Bootstrap.java:289)
		at org.apache.catalina.startup.Bootstrap.main(Bootstrap.java:414)
	Caused by: org.springframework.scheduling.SchedulingException: Could not start Quartz Scheduler; nested exception is org.quartz.SchedulerConfigException: Failure occured during job recovery. [See nested exception: org.quartz.JobPersistenceException: Couldn't store trigger 'monitor-api.ApiTestCaseJob-81' for 'monitor-api.ApiTestCaseJob-81' job:com.baidu.global.mobile.server.guanxing.monitor.job.ApiTestCaseJob [See nested exception: java.lang.ClassNotFoundException: com.baidu.global.mobile.server.guanxing.monitor.job.ApiTestCaseJob]]
		at org.springframework.scheduling.quartz.SchedulerFactoryBean.start(SchedulerFactoryBean.java:710)
		at org.springframework.context.support.DefaultLifecycleProcessor.doStart(DefaultLifecycleProcessor.java:167)
		... 33 more
	Caused by: org.quartz.SchedulerConfigException: Failure occured during job recovery. [See nested exception: org.quartz.JobPersistenceException: Couldn't store trigger 'monitor-api.ApiTestCaseJob-81' for 'monitor-api.ApiTestCaseJob-81' job:com.baidu.global.mobile.server.guanxing.monitor.job.ApiTestCaseJob [See nested exception: java.lang.ClassNotFoundException: com.baidu.global.mobile.server.guanxing.monitor.job.ApiTestCaseJob]]
		at org.quartz.impl.jdbcjobstore.JobStoreSupport.schedulerStarted(JobStoreSupport.java:692)
		at org.quartz.core.QuartzScheduler.start(QuartzScheduler.java:567)
		at org.quartz.impl.StdScheduler.start(StdScheduler.java:142)
		at org.springframework.scheduling.quartz.SchedulerFactoryBean.startScheduler(SchedulerFactoryBean.java:644)
		at org.springframework.scheduling.quartz.SchedulerFactoryBean.start(SchedulerFactoryBean.java:707)
		... 34 more
	Caused by: org.quartz.JobPersistenceException: Couldn't store trigger 'monitor-api.ApiTestCaseJob-81' for 'monitor-api.ApiTestCaseJob-81' job:com.baidu.global.mobile.server.guanxing.monitor.job.ApiTestCaseJob [See nested exception: java.lang.ClassNotFoundException: com.baidu.global.mobile.server.guanxing.monitor.job.ApiTestCaseJob]
		at org.quartz.impl.jdbcjobstore.JobStoreSupport.storeTrigger(JobStoreSupport.java:1223)
		at org.quartz.impl.jdbcjobstore.JobStoreSupport.doUpdateOfMisfiredTrigger(JobStoreSupport.java:1037)
		at org.quartz.impl.jdbcjobstore.JobStoreSupport.recoverMisfiredJobs(JobStoreSupport.java:986)
		at org.quartz.impl.jdbcjobstore.JobStoreSupport.recoverJobs(JobStoreSupport.java:866)
		at org.quartz.impl.jdbcjobstore.JobStoreSupport$1.executeVoid(JobStoreSupport.java:838)
		at org.quartz.impl.jdbcjobstore.JobStoreSupport$VoidTransactionCallback.execute(JobStoreSupport.java:3703)
		at org.quartz.impl.jdbcjobstore.JobStoreSupport$VoidTransactionCallback.execute(JobStoreSupport.java:3701)
		at org.quartz.impl.jdbcjobstore.JobStoreSupport.executeInNonManagedTXLock(JobStoreSupport.java:3787)
		at org.quartz.impl.jdbcjobstore.JobStoreSupport.recoverJobs(JobStoreSupport.java:834)
		at org.quartz.impl.jdbcjobstore.JobStoreSupport.schedulerStarted(JobStoreSupport.java:690)
		... 38 more
	Caused by: java.lang.ClassNotFoundException: com.baidu.global.mobile.server.guanxing.monitor.job.ApiTestCaseJob
		at org.apache.catalina.loader.WebappClassLoader.loadClass(WebappClassLoader.java:1680)
		at org.apache.catalina.loader.WebappClassLoader.loadClass(WebappClassLoader.java:1526)
		at org.springframework.scheduling.quartz.ResourceLoaderClassLoadHelper.loadClass(ResourceLoaderClassLoadHelper.java:75)
		at org.springframework.scheduling.quartz.ResourceLoaderClassLoadHelper.loadClass(ResourceLoaderClassLoadHelper.java:80)
		at org.quartz.impl.jdbcjobstore.StdJDBCDelegate.selectJobDetail(StdJDBCDelegate.java:852)
		at org.quartz.impl.jdbcjobstore.JobStoreSupport.storeTrigger(JobStoreSupport.java:1205)
		... 47 more





