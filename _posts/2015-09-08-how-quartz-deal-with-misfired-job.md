---
title: Quartz的misfire机制
layout: post
---


Quartz在如下情况下可能没有办法按时执行job：

1. 所有的worker线程都处于忙碌状态（可能在执行更高优先级的任务）
2. Quartz scheduler挂掉了
3. 这个任务的开始执行时间就是在过去（可能是创建时候配置错了）


org.quartz.jobStore.misfireThreshold (in milliseconds), 默认是60000ms(也就是1min). 这也是Quartz支持秒级别精确调度的一个体现，超过1分钟没有执行的任务就被认为是misfired。It defines how late the trigger should be to be considered misfired. 如果有任务超过了org.quartz.jobStore.misfireThreshold时间还没有执行，那么Quartz会有一个特别的misfire handler thread对其进行处理。

Quartz内建了几个misfire处理机制，并且不同的trigger有不同的msifire处理机制:

* Trigger: MISFIRE_INSTRUCTION_SMART_POLICY = 0; MISFIRE_INSTRUCTION_IGNORE_MISFIRE_POLICY = -1;
	* CalendarIntervalTrigger: MISFIRE_INSTRUCTION_FIRE_ONCE_NOW = 1; MISFIRE_INSTRUCTION_DO_NOTHING = 2;
		* CalendarIntervalTriggerImpl: 见下面
	* CoreTrigger
		* CalendarIntervalTriggerImpl: 见下面
		* CronTriggerImpl: 见下面
		* DailTimeIntervalTriggerImpl: 见下面
		* SimpleTriggerImpl: 见下面
	* CronTrigger: MISFIRE_INSTRUCTION_FIRE_ONCE_NOW = 1; MISFIRE_INSTRUCTION_DO_NOTHING = 2;
		* CronTriggerImpl
	* DailyTimeIntervalTrigger: MISFIRE_INSTRUCTION_FIRE_ONCE_NOW = 1; MISFIRE_INSTRUCTION_DO_NOTHING = 2;
		* DailTimeIntervalTriggerImpl
	* MutableTrigger
		* OperableTrigger
			* AbstractTrigger: int misfireInstruction = MISFIRE_INSTRUCTION_SMART_POLICY;
				* CalendarIntervalTriggerImpl: 继承AbstractTrigger
				* CronTriggerImpl: 继承AbstractTrigger
				* DailTimeIntervalTriggerImpl: 继承AbstractTrigger
				* SimpleTriggerImpl: 继承AbstractTrigger
	* SimpleTrigger: MISFIRE_INSTRUCTION_FIRE_NOW = 1; int MISFIRE_INSTRUCTION_RESCHEDULE_NOW_WITH_EXISTING_REPEAT_COUNT = 2; MISFIRE_INSTRUCTION_RESCHEDULE_NOW_WITH_REMAINING_REPEAT_COUNT = 3; MISFIRE_INSTRUCTION_RESCHEDULE_NEXT_WITH_REMAINING_COUNT = 4; MISFIRE_INSTRUCTION_RESCHEDULE_NEXT_WITH_EXISTING_COUNT = 5;
		* SimpleTriggerImpl 

看起来很复杂，其实真正的实现也就是四个：

* CalendarIntervalTriggerImpl
* CronTriggerImpl
* DailTimeIntervalTriggerImpl
* SimpleTriggerImpl

每一个trigger都有一个misfireInstruction，默认都是MISFIRE_INSTRUCTION_SMART_POLICY。可以通过`setMisfireInstruction(int misfireInstruction)`更改，不过这个方法会对调用`validateMisfireInstruction(int candidateMisfireInstruction)`进行验证。如果验证不通过，会抛`IllegalArgumentException("The misfire instruction code is invalid for this type of trigger.")`异常。
并且在misfired之后，会调用`updateAfterMisfire(org.quartz.Calendar cal)`进行相应的处理。


以jdbcjobstore为例，Quartz是有一个线程专门扫描和处理misfired的任务（因为代码特别长，将近4000行，我做了裁剪，只保留跟misfired相关的代码）：

	package org.quartz.impl.jdbcjobstore;

	/**
	 * Contains base functionality for JDBC-based JobStore implementations.
	 */
	public abstract class JobStoreSupport implements JobStore, Constants {

		/**
	     * The the number of milliseconds by which a trigger must have missed its
	     * next-fire-time, in order for it to be considered "misfired" and thus
	     * have its misfire instruction applied.
	     */
	    private long misfireThreshold = 60000L; // one minute

	    private MisfireHandler misfireHandler = null;

 		/** 
 		 * the maximum number of misfired triggers that the misfire handling
	     * thread will try to recover at one time (within one transaction).  The
	     * default is 20.
	     */
	    protected int maxToRecoverAtATime = 20;
	    
	    /**
	     * whether to check to see if there are Triggers that have misfired
	     * before actually acquiring the lock to recover them.  This should be 
	     * set to false if the majority of the time, there are are misfired
	     * Triggers.
	     */
	    private boolean doubleCheckLockMisfireHandler = true;
	    
	    private ThreadExecutor threadExecutor = new DefaultThreadExecutor();
	    
	    
	   
	    /**
	     * @see org.quartz.spi.JobStore#schedulerStarted()
	     */
	    public void schedulerStarted() throws SchedulerException {

	        if (isClustered()) {
	            clusterManagementThread = new ClusterManager();
	            if(initializersLoader != null)
	                clusterManagementThread.setContextClassLoader(initializersLoader);
	            clusterManagementThread.initialize();
	        } else {
	            try {
	                recoverJobs();
	            } catch (SchedulerException se) {
	                throw new SchedulerConfigException(
	                        "Failure occured during job recovery.", se);
	            }
	        }

	        misfireHandler = new MisfireHandler();
	        if(initializersLoader != null)
	            misfireHandler.setContextClassLoader(initializersLoader);
	        misfireHandler.initialize();
	        schedulerRunning = true;
	        
	        getLog().debug("JobStore background threads started (as scheduler was started).");
	    }
	    
	    
	    /**
	     * Recover any failed or misfired jobs and clean up the data store as
	     * appropriate.
	     * 
	     * @throws JobPersistenceException if jobs could not be recovered
	     */
	    protected void recoverJobs() throws JobPersistenceException {
	        executeInNonManagedTXLock(
	            LOCK_TRIGGER_ACCESS,
	            new VoidTransactionCallback() {
	                public void executeVoid(Connection conn) throws JobPersistenceException {
	                    recoverJobs(conn);
	                }
	            }, null);
	    }
	    
	    /**
	     * <p>
	     * Will recover any failed or misfired jobs and clean up the data store as
	     * appropriate.
	     * </p>
	     * 
	     * @throws JobPersistenceException
	     *           if jobs could not be recovered
	     */
	    protected void recoverJobs(Connection conn) throws JobPersistenceException {
	        try {
	            // update inconsistent job states
	            int rows = getDelegate().updateTriggerStatesFromOtherStates(conn,
	                    STATE_WAITING, STATE_ACQUIRED, STATE_BLOCKED);

	            rows += getDelegate().updateTriggerStatesFromOtherStates(conn,
	                        STATE_PAUSED, STATE_PAUSED_BLOCKED, STATE_PAUSED_BLOCKED);
	            
	            getLog().info(
	                    "Freed " + rows
	                            + " triggers from 'acquired' / 'blocked' state.");

	            // clean up misfired jobs
	            recoverMisfiredJobs(conn, true);
	            
	            // recover jobs marked for recovery that were not fully executed
	            List<OperableTrigger> recoveringJobTriggers = getDelegate()
	                    .selectTriggersForRecoveringJobs(conn);
	            getLog()
	                    .info(
	                            "Recovering "
	                                    + recoveringJobTriggers.size()
	                                    + " jobs that were in-progress at the time of the last shut-down.");

	            for (OperableTrigger recoveringJobTrigger: recoveringJobTriggers) {
	                if (jobExists(conn, recoveringJobTrigger.getJobKey())) {
	                    recoveringJobTrigger.computeFirstFireTime(null);
	                    storeTrigger(conn, recoveringJobTrigger, null, false,
	                            STATE_WAITING, false, true);
	                }
	            }
	            getLog().info("Recovery complete.");

	            // remove lingering 'complete' triggers...
	            List<TriggerKey> cts = getDelegate().selectTriggersInState(conn, STATE_COMPLETE);
	            for(TriggerKey ct: cts) {
	                removeTrigger(conn, ct);
	            }
	            getLog().info(
	                "Removed " + cts.size() + " 'complete' triggers.");
	            
	            // clean up any fired trigger entries
	            int n = getDelegate().deleteFiredTriggers(conn);
	            getLog().info("Removed " + n + " stale fired job entries.");
	        } catch (JobPersistenceException e) {
	            throw e;
	        } catch (Exception e) {
	            throw new JobPersistenceException("Couldn't recover jobs: "
	                    + e.getMessage(), e);
	        }
	    }

	    protected long getMisfireTime() {
	        long misfireTime = System.currentTimeMillis();
	        if (getMisfireThreshold() > 0) {
	            misfireTime -= getMisfireThreshold();
	        }

	        return (misfireTime > 0) ? misfireTime : 0;
	    }

	    /**
	     * Helper class for returning the composite result of trying
	     * to recover misfired jobs.
	     */
	    protected static class RecoverMisfiredJobsResult {
	        public static final RecoverMisfiredJobsResult NO_OP =
	            new RecoverMisfiredJobsResult(false, 0, Long.MAX_VALUE);
	        
	        private boolean _hasMoreMisfiredTriggers;
	        private int _processedMisfiredTriggerCount;
	        private long _earliestNewTime;
	        
	        public RecoverMisfiredJobsResult(
	            boolean hasMoreMisfiredTriggers, int processedMisfiredTriggerCount, long earliestNewTime) {
	            _hasMoreMisfiredTriggers = hasMoreMisfiredTriggers;
	            _processedMisfiredTriggerCount = processedMisfiredTriggerCount;
	            _earliestNewTime = earliestNewTime;
	        }
	        
	        public boolean hasMoreMisfiredTriggers() {
	            return _hasMoreMisfiredTriggers;
	        }
	        public int getProcessedMisfiredTriggerCount() {
	            return _processedMisfiredTriggerCount;
	        } 
	        public long getEarliestNewTime() {
	            return _earliestNewTime;
	        } 
	    }
	    
	    protected RecoverMisfiredJobsResult recoverMisfiredJobs(
	        Connection conn, boolean recovering)
	        throws JobPersistenceException, SQLException {

	        // If recovering, we want to handle all of the misfired
	        // triggers right away.
	        int maxMisfiresToHandleAtATime = 
	            (recovering) ? -1 : getMaxMisfiresToHandleAtATime();
	        
	        List<TriggerKey> misfiredTriggers = new LinkedList<TriggerKey>();
	        long earliestNewTime = Long.MAX_VALUE;
	        // We must still look for the MISFIRED state in case triggers were left 
	        // in this state when upgrading to this version that does not support it. 
	        boolean hasMoreMisfiredTriggers =
	            getDelegate().hasMisfiredTriggersInState(
	                conn, STATE_WAITING, getMisfireTime(), 
	                maxMisfiresToHandleAtATime, misfiredTriggers);

	        if (hasMoreMisfiredTriggers) {
	            getLog().info(
	                "Handling the first " + misfiredTriggers.size() +
	                " triggers that missed their scheduled fire-time.  " +
	                "More misfired triggers remain to be processed.");
	        } else if (misfiredTriggers.size() > 0) { 
	            getLog().info(
	                "Handling " + misfiredTriggers.size() + 
	                " trigger(s) that missed their scheduled fire-time.");
	        } else {
	            getLog().debug(
	                "Found 0 triggers that missed their scheduled fire-time.");
	            return RecoverMisfiredJobsResult.NO_OP; 
	        }

	        for (TriggerKey triggerKey: misfiredTriggers) {
	            
	            OperableTrigger trig = 
	                retrieveTrigger(conn, triggerKey);

	            if (trig == null) {
	                continue;
	            }

	            doUpdateOfMisfiredTrigger(conn, trig, false, STATE_WAITING, recovering);

	            if(trig.getNextFireTime() != null && trig.getNextFireTime().getTime() < earliestNewTime)
	                earliestNewTime = trig.getNextFireTime().getTime();
	        }

	        return new RecoverMisfiredJobsResult(
	                hasMoreMisfiredTriggers, misfiredTriggers.size(), earliestNewTime);
	    }

	    protected boolean updateMisfiredTrigger(Connection conn,
	            TriggerKey triggerKey, String newStateIfNotComplete, boolean forceState)
	        throws JobPersistenceException {
	        try {

	            OperableTrigger trig = retrieveTrigger(conn, triggerKey);

	            long misfireTime = System.currentTimeMillis();
	            if (getMisfireThreshold() > 0) {
	                misfireTime -= getMisfireThreshold();
	            }

	            if (trig.getNextFireTime().getTime() > misfireTime) {
	                return false;
	            }

	            doUpdateOfMisfiredTrigger(conn, trig, forceState, newStateIfNotComplete, false);

	            return true;

	        } catch (Exception e) {
	            throw new JobPersistenceException(
	                    "Couldn't update misfired trigger '" + triggerKey + "': " + e.getMessage(), e);
	        }
	    }

	    private void doUpdateOfMisfiredTrigger(Connection conn, OperableTrigger trig, boolean forceState, String newStateIfNotComplete, boolean recovering) throws JobPersistenceException {
	        Calendar cal = null;
	        if (trig.getCalendarName() != null) {
	            cal = retrieveCalendar(conn, trig.getCalendarName());
	        }

	        schedSignaler.notifyTriggerListenersMisfired(trig);

	        trig.updateAfterMisfire(cal);

	        if (trig.getNextFireTime() == null) {
	            storeTrigger(conn, trig,
	                null, true, STATE_COMPLETE, forceState, recovering);
	            schedSignaler.notifySchedulerListenersFinalized(trig);
	        } else {
	            storeTrigger(conn, trig, null, true, newStateIfNotComplete,
	                    forceState, false);
	        }
	    }
	    
	    
	    //---------------------------------------------------------------------------
	    // Management methods
	    //---------------------------------------------------------------------------

	    protected RecoverMisfiredJobsResult doRecoverMisfires() throws JobPersistenceException {
	        boolean transOwner = false;
	        Connection conn = getNonManagedTXConnection();
	        try {
	            RecoverMisfiredJobsResult result = RecoverMisfiredJobsResult.NO_OP;
	            
	            // Before we make the potentially expensive call to acquire the 
	            // trigger lock, peek ahead to see if it is likely we would find
	            // misfired triggers requiring recovery.
	            int misfireCount = (getDoubleCheckLockMisfireHandler()) ?
	                getDelegate().countMisfiredTriggersInState(
	                    conn, STATE_WAITING, getMisfireTime()) : 
	                Integer.MAX_VALUE;
	            
	            if (misfireCount == 0) {
	                getLog().debug(
	                    "Found 0 triggers that missed their scheduled fire-time.");
	            } else {
	                transOwner = getLockHandler().obtainLock(conn, LOCK_TRIGGER_ACCESS);
	                
	                result = recoverMisfiredJobs(conn, false);
	            }
	            
	            commitConnection(conn);
	            return result;
	        } catch (JobPersistenceException e) {
	            rollbackConnection(conn);
	            throw e;
	        } catch (SQLException e) {
	            rollbackConnection(conn);
	            throw new JobPersistenceException("Database error recovering from misfires.", e);
	        } catch (RuntimeException e) {
	            rollbackConnection(conn);
	            throw new JobPersistenceException("Unexpected runtime exception: "
	                    + e.getMessage(), e);
	        } finally {
	            try {
	                releaseLock(LOCK_TRIGGER_ACCESS, transOwner);
	            } finally {
	                cleanupConnection(conn);
	            }
	        }
	    }

	    
	    /////////////////////////////////////////////////////////////////////////////
	    //
	    // MisfireHandler Thread
	    //
	    /////////////////////////////////////////////////////////////////////////////

	    class MisfireHandler extends Thread {

	        private volatile boolean shutdown = false;

	        private int numFails = 0;
	        

	        MisfireHandler() {
	            this.setName("QuartzScheduler_" + instanceName + "-" + instanceId + "_MisfireHandler");
	            this.setDaemon(getMakeThreadsDaemons());
	        }

	        public void initialize() {
	            ThreadExecutor executor = getThreadExecutor();
	            executor.execute(MisfireHandler.this);
	        }

	        public void shutdown() {
	            shutdown = true;
	            this.interrupt();
	        }

	        private RecoverMisfiredJobsResult manage() {
	            try {
	                getLog().debug("MisfireHandler: scanning for misfires...");

	                RecoverMisfiredJobsResult res = doRecoverMisfires();
	                numFails = 0;
	                return res;
	            } catch (Exception e) {
	                if(numFails % 4 == 0) {
	                    getLog().error(
	                        "MisfireHandler: Error handling misfires: "
	                                + e.getMessage(), e);
	                }
	                numFails++;
	            }
	            return RecoverMisfiredJobsResult.NO_OP;
	        }

	        @Override
	        public void run() {
	            
	            while (!shutdown) {

	                long sTime = System.currentTimeMillis();

	                RecoverMisfiredJobsResult recoverMisfiredJobsResult = manage();

	                if (recoverMisfiredJobsResult.getProcessedMisfiredTriggerCount() > 0) {
	                    signalSchedulingChangeImmediately(recoverMisfiredJobsResult.getEarliestNewTime());
	                }

	                if (!shutdown) {
	                    long timeToSleep = 50l;  // At least a short pause to help balance threads
	                    if (!recoverMisfiredJobsResult.hasMoreMisfiredTriggers()) {
	                        timeToSleep = getMisfireThreshold() - (System.currentTimeMillis() - sTime);
	                        if (timeToSleep <= 0) {
	                            timeToSleep = 50l;
	                        }

	                        if(numFails > 0) {
	                            timeToSleep = Math.max(getDbRetryInterval(), timeToSleep);
	                        }
	                    }
	                    
	                    try {
	                        Thread.sleep(timeToSleep);
	                    } catch (Exception ignore) {
	                    }
	                }//while !shutdown
	            }
	        }
	    }
	}

	// EOF

可以看到其实Quartz起了一个单独的MisfireHandler线程来处理这些misfired job。它的逻辑很简单，就是找出misfired的trigger，然后尝试恢复他们:


	 protected RecoverMisfiredJobsResult recoverMisfiredJobs(
	        Connection conn, boolean recovering)
	        throws JobPersistenceException, SQLException {

	        // If recovering, we want to handle all of the misfired
	        // triggers right away.
	        int maxMisfiresToHandleAtATime = 
	            (recovering) ? -1 : getMaxMisfiresToHandleAtATime();
	        
	        List<TriggerKey> misfiredTriggers = new LinkedList<TriggerKey>();
	        long earliestNewTime = Long.MAX_VALUE;
	        // We must still look for the MISFIRED state in case triggers were left 
	        // in this state when upgrading to this version that does not support it. 
	        boolean hasMoreMisfiredTriggers =
	            getDelegate().hasMisfiredTriggersInState(
	                conn, STATE_WAITING, getMisfireTime(), 
	                maxMisfiresToHandleAtATime, misfiredTriggers);

	        if (hasMoreMisfiredTriggers) {
	            getLog().info(
	                "Handling the first " + misfiredTriggers.size() +
	                " triggers that missed their scheduled fire-time.  " +
	                "More misfired triggers remain to be processed.");
	        } else if (misfiredTriggers.size() > 0) { 
	            getLog().info(
	                "Handling " + misfiredTriggers.size() + 
	                " trigger(s) that missed their scheduled fire-time.");
	        } else {
	            getLog().debug(
	                "Found 0 triggers that missed their scheduled fire-time.");
	            return RecoverMisfiredJobsResult.NO_OP; 
	        }

	        for (TriggerKey triggerKey: misfiredTriggers) {
	            
	            OperableTrigger trig = 
	                retrieveTrigger(conn, triggerKey);

	            if (trig == null) {
	                continue;
	            }

	            doUpdateOfMisfiredTrigger(conn, trig, false, STATE_WAITING, recovering);

	            if(trig.getNextFireTime() != null && trig.getNextFireTime().getTime() < earliestNewTime)
	                earliestNewTime = trig.getNextFireTime().getTime();
	        }

	        return new RecoverMisfiredJobsResult(
	                hasMoreMisfiredTriggers, misfiredTriggers.size(), earliestNewTime);
	    }


对获取到的misfired trigger，进行如下恢复操作：

	for (TriggerKey triggerKey: misfiredTriggers) {
	            
        OperableTrigger trig = 
            retrieveTrigger(conn, triggerKey);

        if (trig == null) {
            continue;
        }

        doUpdateOfMisfiredTrigger(conn, trig, false, STATE_WAITING, recovering);

        if(trig.getNextFireTime() != null && trig.getNextFireTime().getTime() < earliestNewTime)
            earliestNewTime = trig.getNextFireTime().getTime();
    }

     private void doUpdateOfMisfiredTrigger(Connection conn, OperableTrigger trig, boolean forceState, String newStateIfNotComplete, boolean recovering) throws JobPersistenceException {
        Calendar cal = null;
        if (trig.getCalendarName() != null) {
            cal = retrieveCalendar(conn, trig.getCalendarName());
        }

        schedSignaler.notifyTriggerListenersMisfired(trig);

        trig.updateAfterMisfire(cal);

        if (trig.getNextFireTime() == null) {
            storeTrigger(conn, trig,
                null, true, STATE_COMPLETE, forceState, recovering);
            schedSignaler.notifySchedulerListenersFinalized(trig);
        } else {
            storeTrigger(conn, trig, null, true, newStateIfNotComplete,
                    forceState, false);
        }
    }

doUpdateOfMisfiredTrigger的逻辑其实蛮简单的，就是一个切面，前后通知triggerListener，主要委托给trigger本事的updateAfterMisfire方法。这是一个抽象方法，所有的trigger实现类都要实现这个方法，我们看一下我们最常用的CronTrigger是怎么实现的：

	/**
     * <p>
     * Updates the <code>CronTrigger</code>'s state based on the
     * MISFIRE_INSTRUCTION_XXX that was selected when the <code>CronTrigger</code>
     * was created.
     * </p>
     * 
     * <p>
     * If the misfire instruction is set to MISFIRE_INSTRUCTION_SMART_POLICY,
     * then the following scheme will be used: <br>
     * <ul>
     * <li>The instruction will be interpreted as <code>MISFIRE_INSTRUCTION_FIRE_ONCE_NOW</code>
     * </ul>
     * </p>
     */
    @Override
    public void updateAfterMisfire(org.quartz.Calendar cal) {
        int instr = getMisfireInstruction();

        if(instr == Trigger.MISFIRE_INSTRUCTION_IGNORE_MISFIRE_POLICY)
            return;

        if (instr == MISFIRE_INSTRUCTION_SMART_POLICY) {
            instr = MISFIRE_INSTRUCTION_FIRE_ONCE_NOW;
        }

        if (instr == MISFIRE_INSTRUCTION_DO_NOTHING) {
            Date newFireTime = getFireTimeAfter(new Date());
            while (newFireTime != null && cal != null
                    && !cal.isTimeIncluded(newFireTime.getTime())) {
                newFireTime = getFireTimeAfter(newFireTime);
            }
            setNextFireTime(newFireTime);
        } else if (instr == MISFIRE_INSTRUCTION_FIRE_ONCE_NOW) {
            setNextFireTime(new Date());
        }
    }

发现其实做的事情很简单，就是两件事情：

1. 获取和调整misfireInstruction
2. 根据misfireInstruction进行相应的处理，主要是设置nextFireTime，以便让该miss的任务立即执行或者后面执行。

SimpleTriggerImpl也是类似，但是相对复杂一下。这里就不展开了。感兴趣的同学直接看源代码就是了。


参考文档
-------

1. [http://www.nurkiewicz.com/2012/04/quartz-scheduler-misfire-instructions.html](Quartz scheduler misfire instructions explained)