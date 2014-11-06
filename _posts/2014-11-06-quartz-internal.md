---
title: Quartz工作机制
layout: post
---

使用了几年Quartz，其实还是停留在使用的层面，最近想做一个自动化监控报警的，想了解一下Quartz是怎样定时触发job的。于是深入了解了一下代码。


1. 启动初始化
-------------

Quartz启动时会根据配置信息进行相应的初始化。例如下面的这个spring-quartz.xml配置文件：

	<?xml version="1.0" encoding="GBK"?>
	<beans xmlns="http://www.springframework.org/schema/beans"
		xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
		xsi:schemaLocation="
	       http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-2.0.xsd">

		<bean class="org.springframework.scheduling.quartz.SchedulerFactoryBean">
			<property name="jobFactory">
				<bean
					class="me.arganzheng.study.quartz.task.SpringBeanJobFactory" />
			</property>
			<property name="quartzProperties">
				<props>
					<prop key="org.quartz.jobStore.dataSource">QuartzDataSource</prop>
					<prop key="org.quartz.dataSource.QuartzDataSource.driver">${jdbc.driverClassName}</prop>
					<prop key="org.quartz.dataSource.QuartzDataSource.URL">${jdbc.url}</prop>
					<prop key="org.quartz.dataSource.QuartzDataSource.user">${jdbc.username}</prop>
					<prop key="org.quartz.dataSource.QuartzDataSource.password">${jdbc.password}</prop>
					<prop key="org.quartz.jobStore.tablePrefix">QRTZ_</prop>
					<prop key="org.quartz.jobStore.isClustered">true</prop>
					<prop key="org.quartz.threadPool.threadCount">3</prop>
					<prop key="org.quartz.jobStore.class">org.quartz.impl.jdbcjobstore.JobStoreTX</prop>
					<prop key="org.quartz.jobStore.driverDelegateClass">org.quartz.impl.jdbcjobstore.StdJDBCDelegate</prop>
					<prop key="org.quartz.scheduler.jobFactory.class">me.arganzheng.study.quartz.task.SpringBeanJobFactory
					</prop>
					<prop key="org.quartz.scheduler.instanceId">AUTO</prop>
					<prop key="org.quartz.scheduler.instanceIdGenerator.class">me.arganzheng.study.quartz.task.QuartzSchedulerInstanceIdGenerator
					</prop>
				</props>
			</property>
			<property name="globalJobListeners">
				<list>
					<bean
						class="me.arganzheng.study.quartz.task.QuartzEventListener" />
				</list>
			</property>
		</bean>
	</beans>

可以看出主要的配置信息包括：

### 1. 实例ID

* org.quartz.scheduler.instanceId
* org.quartz.scheduler.instanceIdGenerator.class

### 2. jobStore的数据源

主要用于存储JobDetails和Triggers。Quartz包含了三种内建的[JobStore](http://quartz-scheduler.org/documentation/quartz-2.x/tutorials/tutorial-lesson-09):

1. RAMJobStore
2. JDBCJobStore
3. TerracottaJobStore

我们这里使用的是JDBCJobStore。需要先创建相应的表格。SQL语句可以在jar包中找到，比如quartz_2.2.1_mysql_innodb.sql，执行之后会创建如下表格以及相应的索引：

* QRTZ_FIRED_TRIGGERS: 存储与已触发的 Trigger 相关的状态信息，以及相联 Job 的执行信息。
* QRTZ_PAUSED_TRIGGER_GRPS: 存储已暂停的 Trigger 组的信息。
* QRTZ_SCHEDULER_STATE: 存储少量的有关 Scheduler 的状态信息，假如是用于集群中，可以看到其他的 Scheduler 实例。
* QRTZ_LOCKS: 存储程序的悲观锁的信息(假如使用了悲观锁)  
* QRTZ_SIMPLE_TRIGGERS: 存储简单的 Trigger，包括重复次数，间隔，以及已触发的次数。
* QRTZ_SIMPROP_TRIGGERS, 
* QRTZ_CRON_TRIGGERS: 存储 Cron Trigger，包括 Cron 表达式和时区信息。
* QRTZ_BLOB_TRIGGERS: Trigger 作为 Blob 类型存储(用于 Quartz 用户用 JDBC 创建他们自己定制的 Trigger 类型，JobStore 并不知道如何存储实例的时候) 。
* QRTZ_TRIGGERS: 存储已配置的 Trigger 的信息 
* QRTZ_JOB_DETAILS: 存储每一个已配置的 Job 的详细信息。 
* QRTZ_CALENDARS:以 Blob 类型存储 Quartz 的 Calendar 信息 

**注意** 如果你的`org.quartz.jobStore.tablePrefix`不是配置成QRTZ_，那么需要修改相应的SQL语句。

**TIP** [Auto Create Quartz Tables at Startup](http://caffeineinduced.wordpress.com/2010/04/05/auto-create-quartz-tables-at-startup/)

可以在应用启动的时候自动进行表结构创建。

### 3. 执行job的线程池

Quartz的job是通过线程池来执行的。

* org.quartz.threadPool.threadCount: 配置线程池的大小。默认是10个线程。
* org.quartz.threadPool.class: 线程池的类型

### 4. JobFactory

这里为了达到自动注入Spring托管的service，自定义了一个JobFactory，具体可以参考笔者前面的文章：[Quartz与Spring的整合-Quartz中的job如何自动注入spring容器托管的对象](http://blog.arganzheng.me/posts/quartz-and-spring-integration-ioc-autowire.html)


这些配置其实对应了`org.springframework.scheduling.quart.SchedulerFactoryBean`中的属性：

	public class SchedulerFactoryBean extends SchedulerAccessor implements FactoryBean<Scheduler>, BeanNameAware,
			ApplicationContextAware, InitializingBean, DisposableBean, SmartLifecycle {

		public static final String PROP_THREAD_COUNT = "org.quartz.threadPool.threadCount";

		public static final int DEFAULT_THREAD_COUNT = 10;


		private static final ThreadLocal<ResourceLoader> configTimeResourceLoaderHolder =
				new ThreadLocal<ResourceLoader>();

		private static final ThreadLocal<Executor> configTimeTaskExecutorHolder =
				new ThreadLocal<Executor>();

		private static final ThreadLocal<DataSource> configTimeDataSourceHolder =
				new ThreadLocal<DataSource>();

		private static final ThreadLocal<DataSource> configTimeNonTransactionalDataSourceHolder =
				new ThreadLocal<DataSource>();

		private Class<? extends SchedulerFactory> schedulerFactoryClass = StdSchedulerFactory.class;

		private String schedulerName;

		private Resource configLocation;

		private Properties quartzProperties;


		private Executor taskExecutor;

		private DataSource dataSource;

		private DataSource nonTransactionalDataSource;


	    private Map<String, ?> schedulerContextMap;

		private ApplicationContext applicationContext;

		private String applicationContextSchedulerContextKey;

		private JobFactory jobFactory;

		private boolean jobFactorySet = false;


		private boolean autoStartup = true;

		private int startupDelay = 0;

		private int phase = Integer.MAX_VALUE;

		private boolean exposeSchedulerInRepository = false;

		private boolean waitForJobsToCompleteOnShutdown = false;


		private Scheduler scheduler;

	}

然后SchedulerFactoryBean会根据配置的信息创建Scheduler实例：

然后启动Scheduler:

	public class StdSchedulerFactory implements SchedulerFactory {

		private Scheduler instantiate() throws SchedulerException {
	        if (cfg == null) {
	            initialize();
	        }

	        if (initException != null) {
	            throw initException;
	        }

	        JobStore js = null;
	        ThreadPool tp = null;
	        QuartzScheduler qs = null;
	        DBConnectionManager dbMgr = null;
	        String instanceIdGeneratorClass = null;
	        Properties tProps = null;
	        String userTXLocation = null;
	        boolean wrapJobInTx = false;
	        boolean autoId = false;
	        long idleWaitTime = -1;
	        long dbFailureRetry = 15000L; // 15 secs
	        String classLoadHelperClass;
	        String jobFactoryClass;
	        ThreadExecutor threadExecutor;


	        SchedulerRepository schedRep = SchedulerRepository.getInstance();

	        // Get Scheduler Properties
	        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	        /// ....

	        // If Proxying to remote scheduler, short-circuit here...
	        // ~~~~~~~~~~~~~~~~~~
	        /// ...


	        // Create class load helper
	        /// ... loadHelper.initialize();

	        // If Proxying to remote JMX scheduler, short-circuit here...
	        // ~~~~~~~~~~~~~~~~~~
	        /// ....

	        // Get ThreadPool Properties
	        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	        /// ....

	        // Get JobStore Properties
	        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	        /// ....

	        // Set up any DataSources
	        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	        ///....

	        // Set up any SchedulerPlugins
	        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	        /// ....

	        // Set up any JobListeners
	        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	        /// ....
	        

	        // Set up any TriggerListeners
	        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	        /// ...

	        // Get ThreadExecutor Properties
	        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	        /// ...


	        // Fire everything up
	        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	        try {
	                
	            JobRunShellFactory jrsf = null; // Create correct run-shell factory...
	    
	            if (userTXLocation != null) {
	                UserTransactionHelper.setUserTxLocation(userTXLocation);
	            }
	    
	            if (wrapJobInTx) {
	                jrsf = new JTAJobRunShellFactory();
	            } else {
	                jrsf = new JTAAnnotationAwareJobRunShellFactory();
	            }
	    
	            if (autoId) {
	                try {
	                  schedInstId = DEFAULT_INSTANCE_ID;
	                  if (js.isClustered()) {
	                      schedInstId = instanceIdGenerator.generateInstanceId();
	                  }
	                } catch (Exception e) {
	                    getLog().error("Couldn't generate instance Id!", e);
	                    throw new IllegalStateException("Cannot run without an instance id.");
	                }
	            }

	            if (js.getClass().getName().startsWith("org.terracotta.quartz")) {
	                try {
	                    String uuid = (String) js.getClass().getMethod("getUUID").invoke(js);
	                    if(schedInstId.equals(DEFAULT_INSTANCE_ID)) {
	                        schedInstId = "TERRACOTTA_CLUSTERED,node=" + uuid;
	                        if (jmxObjectName == null) {
	                            jmxObjectName = QuartzSchedulerResources.generateJMXObjectName(schedName, schedInstId);
	                        }
	                    } else if(jmxObjectName == null) {
	                        jmxObjectName = QuartzSchedulerResources.generateJMXObjectName(schedName, schedInstId + ",node=" + uuid);
	                    }
	                } catch(Exception e) {
	                    throw new RuntimeException("Problem obtaining node id from TerracottaJobStore.", e);
	                }

	                if(null == cfg.getStringProperty(PROP_SCHED_JMX_EXPORT)) {
	                    jmxExport = true;
	                }
	            }
	            
	            if (js instanceof JobStoreSupport) {
	                JobStoreSupport jjs = (JobStoreSupport)js;
	                jjs.setDbRetryInterval(dbFailureRetry);
	                if(threadsInheritInitalizersClassLoader)
	                    jjs.setThreadsInheritInitializersClassLoadContext(threadsInheritInitalizersClassLoader);
	                
	                jjs.setThreadExecutor(threadExecutor);
	            }
	    
	            QuartzSchedulerResources rsrcs = new QuartzSchedulerResources();
	            rsrcs.setName(schedName);
	            rsrcs.setThreadName(threadName);
	            rsrcs.setInstanceId(schedInstId);
	            rsrcs.setJobRunShellFactory(jrsf);
	            rsrcs.setMakeSchedulerThreadDaemon(makeSchedulerThreadDaemon);
	            rsrcs.setThreadsInheritInitializersClassLoadContext(threadsInheritInitalizersClassLoader);
	            rsrcs.setRunUpdateCheck(!skipUpdateCheck);
	            rsrcs.setBatchTimeWindow(batchTimeWindow);
	            rsrcs.setMaxBatchSize(maxBatchSize);
	            rsrcs.setInterruptJobsOnShutdown(interruptJobsOnShutdown);
	            rsrcs.setInterruptJobsOnShutdownWithWait(interruptJobsOnShutdownWithWait);
	            rsrcs.setJMXExport(jmxExport);
	            rsrcs.setJMXObjectName(jmxObjectName);

	            if (managementRESTServiceEnabled) {
	                ManagementRESTServiceConfiguration managementRESTServiceConfiguration = new ManagementRESTServiceConfiguration();
	                managementRESTServiceConfiguration.setBind(managementRESTServiceHostAndPort);
	                managementRESTServiceConfiguration.setEnabled(managementRESTServiceEnabled);
	                rsrcs.setManagementRESTServiceConfiguration(managementRESTServiceConfiguration);
	            }
	    
	            if (rmiExport) {
	                rsrcs.setRMIRegistryHost(rmiHost);
	                rsrcs.setRMIRegistryPort(rmiPort);
	                rsrcs.setRMIServerPort(rmiServerPort);
	                rsrcs.setRMICreateRegistryStrategy(rmiCreateRegistry);
	                rsrcs.setRMIBindName(rmiBindName);
	            }
	    
	            SchedulerDetailsSetter.setDetails(tp, schedName, schedInstId);

	            rsrcs.setThreadExecutor(threadExecutor);
	            threadExecutor.initialize();

	            rsrcs.setThreadPool(tp);
	            if(tp instanceof SimpleThreadPool) {
	                if(threadsInheritInitalizersClassLoader)
	                    ((SimpleThreadPool)tp).setThreadsInheritContextClassLoaderOfInitializingThread(threadsInheritInitalizersClassLoader);
	            }
	            tp.initialize();
	            tpInited = true;
	    
	            rsrcs.setJobStore(js);
	    
	            // add plugins
	            for (int i = 0; i < plugins.length; i++) {
	                rsrcs.addSchedulerPlugin(plugins[i]);
	            }
	    
	            qs = new QuartzScheduler(rsrcs, idleWaitTime, dbFailureRetry);
	            qsInited = true;
	    
	            // Create Scheduler ref...
	            Scheduler scheduler = instantiate(rsrcs, qs);
	    
	            // set job factory if specified
	            if(jobFactory != null) {
	                qs.setJobFactory(jobFactory);
	            }
	    
	            // Initialize plugins now that we have a Scheduler instance.
	            for (int i = 0; i < plugins.length; i++) {
	                plugins[i].initialize(pluginNames[i], scheduler, loadHelper);
	            }
	    
	            // add listeners
	            for (int i = 0; i < jobListeners.length; i++) {
	                qs.getListenerManager().addJobListener(jobListeners[i], EverythingMatcher.allJobs());
	            }
	            for (int i = 0; i < triggerListeners.length; i++) {
	                qs.getListenerManager().addTriggerListener(triggerListeners[i], EverythingMatcher.allTriggers());
	            }
	    
	            // set scheduler context data...
	            for(Object key: schedCtxtProps.keySet()) {
	                String val = schedCtxtProps.getProperty((String) key);    
	                scheduler.getContext().put((String)key, val);
	            }
	    
	            // fire up job store, and runshell factory
	    
	            js.setInstanceId(schedInstId);
	            js.setInstanceName(schedName);
	            js.setThreadPoolSize(tp.getPoolSize());
	            js.initialize(loadHelper, qs.getSchedulerSignaler());

	            jrsf.initialize(scheduler);
	            
	            qs.initialize();
	    
	            getLog().info(
	                    "Quartz scheduler '" + scheduler.getSchedulerName()
	                            + "' initialized from " + propSrc);
	    
	            getLog().info("Quartz scheduler version: " + qs.getVersion());
	    
	            // prevents the repository from being garbage collected
	            qs.addNoGCObject(schedRep);
	            // prevents the db manager from being garbage collected
	            if (dbMgr != null) {
	                qs.addNoGCObject(dbMgr);
	            }
	    
	            schedRep.bind(scheduler);
	            return scheduler;
	        }
	        catch(SchedulerException e) {
	            shutdownFromInstantiateException(tp, qs, tpInited, qsInited);
	            throw e;
	        }
	        catch(RuntimeException re) {
	            shutdownFromInstantiateException(tp, qs, tpInited, qsInited);
	            throw re;
	        }
	        catch(Error re) {
	            shutdownFromInstantiateException(tp, qs, tpInited, qsInited);
	            throw re;
	        }
	    }
	}

其中创建QuartzScheduler是Quartz的核心类，间接实现org.quartz.Scheduler接口，包含注册和定时调度job。


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

这里最主要是创建了一个QuartzSchedulerThread：

	this.schedThread = new QuartzSchedulerThread(this, resources);
    ThreadExecutor schedThreadExecutor = resources.getThreadExecutor();
    schedThreadExecutor.execute(this.schedThread);


这个类的职责就是不断的去获取即将要执行的triggers，然后执行他们。

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


这个线程的主要工作如下：

#### 1. 获取即将执行的Triggers：Get a handle to the next trigger to be fired, and mark it as 'reserved' by the calling scheduler.

	List<OperableTrigger> triggers = qsRsrcs.getJobStore().acquireNextTriggers(now + idleWaitTime, Math.min(availThreadCount, qsRsrcs.getMaxBatchSize()), qsRsrcs.getBatchTimeWindow());

其中：

1. idleWaitTime 默认为30s。

		private long idleWaitTime = DEFAULT_IDLE_WAIT_TIME;
	  	// When the scheduler finds there is no current trigger to fire, how long
	    // it should wait until checking again...
	    private static long DEFAULT_IDLE_WAIT_TIME = 30L * 1000L;

2.  maxBatchSize默认为1。
3.  batchTimeWindow默认为0。

这个方法会查询一个给定时间区间的triggers，然后


		// FUTURE_TODO: this really ought to return something like a FiredTriggerBundle,
	    // so that the fireInstanceId doesn't have to be on the trigger...
	    protected List<OperableTrigger> acquireNextTrigger(Connection conn, long noLaterThan, int maxCount, long timeWindow)
	        throws JobPersistenceException {
	        
	        List<OperableTrigger> acquiredTriggers = new ArrayList<OperableTrigger>();
	        Set<JobKey> acquiredJobKeysForNoConcurrentExec = new HashSet<JobKey>();
	        final int MAX_DO_LOOP_RETRY = 3;
	        int currentLoopCount = 0;
	        long firstAcquiredTriggerFireTime = 0;
	        
	        do {
	            currentLoopCount ++;
	            try {
	                List<TriggerKey> keys = getDelegate().selectTriggerToAcquire(conn, noLaterThan + timeWindow, getMisfireTime(), maxCount);
	                
	                // No trigger is ready to fire yet.
	                if (keys == null || keys.size() == 0)
	                    return acquiredTriggers;
	                
	                for(TriggerKey triggerKey: keys) { // 获取每个trigger的详情
	                    // If our trigger is no longer available, try a new one.
	                    OperableTrigger nextTrigger = retrieveTrigger(conn, triggerKey);
	                    if(nextTrigger == null) {
	                        continue; // next trigger
	                    }
	                    
	                    // If trigger's job is set as @DisallowConcurrentExecution, and it has already been added to result, then
	                    // put it back into the timeTriggers set and continue to search for next trigger.
	                    JobKey jobKey = nextTrigger.getJobKey();
	                    JobDetail job = getDelegate().selectJobDetail(conn, jobKey, getClassLoadHelper());
	                    if (job.isConcurrentExectionDisallowed()) {
	                        if (acquiredJobKeysForNoConcurrentExec.contains(jobKey)) {
	                            continue; // next trigger
	                        } else {
	                            acquiredJobKeysForNoConcurrentExec.add(jobKey);
	                        }
	                    }
	                    
	                    // We now have a acquired trigger, let's add to return list.
	                    // If our trigger was no longer in the expected state, try a new one.
	                    int rowsUpdated = getDelegate().updateTriggerStateFromOtherState(conn, triggerKey, STATE_ACQUIRED, STATE_WAITING);
	                    if (rowsUpdated <= 0) {
	                        continue; // next trigger
	                    }
	                    nextTrigger.setFireInstanceId(getFiredTriggerRecordId());
	                    getDelegate().insertFiredTrigger(conn, nextTrigger, STATE_ACQUIRED, null);

	                    acquiredTriggers.add(nextTrigger);
	                    if(firstAcquiredTriggerFireTime == 0)
	                        firstAcquiredTriggerFireTime = nextTrigger.getNextFireTime().getTime();
	                }

	                // if we didn't end up with any trigger to fire from that first
	                // batch, try again for another batch. We allow with a max retry count.
	                if(acquiredTriggers.size() == 0 && currentLoopCount < MAX_DO_LOOP_RETRY) {
	                    continue;
	                }
	                
	                // We are done with the while loop.
	                break;
	            } catch (Exception e) {
	                throw new JobPersistenceException(
	                          "Couldn't acquire next trigger: " + e.getMessage(), e);
	            }
	        } while (true);
	        
	        // Return the acquired trigger list
	        return acquiredTriggers;
	    }

关键的操作是下面三个：
	                
##### 1. List<TriggerKey> keys = getDelegate().selectTriggerToAcquire(conn, noLaterThan + timeWindow, getMisfireTime(), maxCount);

public List<TriggerKey> selectTriggerToAcquire(Connection conn, long noLaterThan, long noEarlierThan, int maxCount): Select the next trigger which will fire to fire between the two given timestamps in ascending order of fire time, and then descending by priority.

##### 2. int rowsUpdated = getDelegate().updateTriggerStateFromOtherState(conn, triggerKey, STATE_ACQUIRED, STATE_WAITING);

public int updateTriggerStateFromOtherState(Connection conn, TriggerKey triggerKey, String newState, String oldState) throws SQLException: Update the given trigger to the given new state, if it is in the given old state.

这个是个乐观锁更新。

##### 3. getDelegate().insertFiredTrigger(conn, nextTrigger, STATE_ACQUIRED, null);

public int insertFiredTrigger(Connection conn, OperableTrigger trigger, String state, JobDetail job) throws SQLException:  Insert a fired trigger.


#### 2. 执行这些triggers：

	long triggerTime = triggers.get(0).getNextFireTime().getTime(); 
	long timeUntilTrigger = triggerTime - now;

然后等待：sigLock.wait(timeUntilTrigger); 这里是使用了Object的wait和notify方法进行等待和唤醒：
	
	public final native void wait(long timeout) throws InterruptedException;
	public final native void notify();

标记已经执行: ACQUIRED=>EXECUTING：

	List<TriggerFiredResult> res = qsRsrcs.getJobStore().triggersFired(triggers);

然后构造一个JobRunShell，用于执行job：

	// 创建JobRunShell以及相应的job实例（根据job的classname调用jobFactory实例化job对象）
	JobRunShell shell = qsRsrcs.getJobRunShellFactory().createJobRunShell(bndle);
	shell.initialize(qs);
	// 执行job
	qsRsrcs.getThreadPool().runInThread(shell);

最后根据执行的结果更新Trigger状态：    

	public enum TriggerState { NONE, NORMAL, PAUSED, COMPLETE, ERROR, BLOCKED }

最后等待一个随机时间，继续上面的操作。


### 关于Quartz的Lock

Quartz定义了一个Semaphore接口反之并发读写问题：

	package org.quartz.impl.jdbcjobstore

	/**
	 * An interface for providing thread/resource locking in order to protect
	 * resources from being altered by multiple threads at the same time.
	 */
	public interface Semaphore {
	 	
	    /**
	     * Grants a lock on the identified resource to the calling thread (blocking
	     * until it is available).
	     * 
	     * @param conn Database connection used to establish lock.  Can be null if
	     * <code>{@link #requiresConnection()}</code> returns false.
	     * 
	     * @return true if the lock was obtained.
	     */
	    boolean obtainLock(Connection conn, String lockName) throws LockException;

	    /**
	     * Release the lock on the identified resource if it is held by the calling
	     * thread.
	     */
	    void releaseLock(String lockName) throws LockException;

	    /**
	     * Whether this Semaphore implementation requires a database connection for
	     * its lock management operations.
	     * 
	     * @see #obtainLock(Connection, String)
	     * @see #releaseLock(String)
	     */
	    boolean requiresConnection();
	}

主要有如下实现：

* Semaphore
	* DBSemaphore: Base class for database based lock handlers for providing thread/resource locking in order to protect resources from being altered by multiple threads at the same time.
		* StdRowLockSemaphore：Internal database based lock handler for providing thread/resource locking in order to protect resources from being altered by multiple threads at the same time.
			* public static final String SELECT_FOR_LOCK = "SELECT * FROM " + TABLE_PREFIX_SUBST + TABLE_LOCKS + " WHERE " + COL_SCHEDULER_NAME + " = " + SCHED_NAME_SUBST + " AND " + COL_LOCK_NAME + " = ? FOR UPDATE";
			* public static final String INSERT_LOCK = "INSERT INTO " + TABLE_PREFIX_SUBST + TABLE_LOCKS + "(" + COL_SCHEDULER_NAME + ", " + COL_LOCK_NAME + ") VALUES (" + SCHED_NAME_SUBST + ", ?)"; 
		* UpdateLockRowSemaphore: Provide thread/resource locking in order to protect resources from being altered by multiple threads at the same time using a db row update.
			* public static final String UPDATE_FOR_LOCK = "UPDATE " + TABLE_PREFIX_SUBST + TABLE_LOCKS + " SET " + COL_LOCK_NAME + " = " + COL_LOCK_NAME + " WHERE " + COL_SCHEDULER_NAME + " = " + SCHED_NAME_SUBST + " AND " + COL_LOCK_NAME + " = ? ";
			* public static final String INSERT_LOCK = "INSERT INTO " + TABLE_PREFIX_SUBST + TABLE_LOCKS + "(" + COL_SCHEDULER_NAME + ", " + COL_LOCK_NAME + ") VALUES (" + SCHED_NAME_SUBST + ", ?)"; 
	* JTANonClusterdSemaphore: Provides in memory thread/resource locking that is JTA javax.transaction.Transaction aware. It is most appropriate for use when using org.quartz.impl.jdbcjobstore.JobStoreCMT without clustering. This Semaphore implementation is not Quartz cluster safe. 
	* SimpleSemaphore: Internal in-memory lock handler for providing thread/resource locking in order to protect resources from being altered by multiple threads at the same time.

对于使用JDBC jobStore的，一般lockHandler都是配置为DBSemaphore。




    
