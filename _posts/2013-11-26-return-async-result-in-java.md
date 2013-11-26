---
layout: post
title: return async result in java
---


在java中，异步操作往往喜欢返回一个Future对象，顾名思义，表示结果在未来的某个时刻才完成，虽然future是立马返回的，但是其实是一个未知的结果。调用方可以（需要）查询这个future对象，看是不是已经出结果了。其实是一种polling方式来的。

这些Future对象，一般都是实现java.util.concurrent.Future接口：
    
    package java.util.concurrent;
    
    /**
     * A <tt>Future</tt> represents the result of an asynchronous
     * computation.  Methods are provided to check if the computation is
     * complete, to wait for its completion, and to retrieve the result of
     * the computation.  The result can only be retrieved using method
     * <tt>get</tt> when the computation has completed, blocking if
     * necessary until it is ready.  Cancellation is performed by the
     * <tt>cancel</tt> method.  Additional methods are provided to
     * determine if the task completed normally or was cancelled. Once a
     * computation has completed, the computation cannot be cancelled.
     * If you would like to use a <tt>Future</tt> for the sake
     * of cancellability but not provide a usable result, you can
     * declare types of the form <tt>Future&lt;?&gt;</tt> and
     * return <tt>null</tt> as a result of the underlying task.
     *
     * <p>
     * <b>Sample Usage</b> (Note that the following classes are all
     * made-up.) <p>
     * <pre>
     * interface ArchiveSearcher { String search(String target); }
     * class App {
     *   ExecutorService executor = ...
     *   ArchiveSearcher searcher = ...
     *   void showSearch(final String target)
     *       throws InterruptedException {
     *     Future&lt;String&gt; future
     *       = executor.submit(new Callable&lt;String&gt;() {
     *         public String call() {
     *             return searcher.search(target);
     *         }});
     *     displayOtherThings(); // do other things while searching
     *     try {
     *       displayText(future.get()); // use future
     *     } catch (ExecutionException ex) { cleanup(); return; }
     *   }
     * }
     * </pre>
     *
     * The {@link FutureTask} class is an implementation of <tt>Future</tt> that
     * implements <tt>Runnable</tt>, and so may be executed by an <tt>Executor</tt>.
     * For example, the above construction with <tt>submit</tt> could be replaced by:
     * <pre>
     *     FutureTask&lt;String&gt; future =
     *       new FutureTask&lt;String&gt;(new Callable&lt;String&gt;() {
     *         public String call() {
     *           return searcher.search(target);
     *       }});
     *     executor.execute(future);
     * </pre>
     *
     * <p>Memory consistency effects: Actions taken by the asynchronous computation
     * <a href="package-summary.html#MemoryVisibility"> <i>happen-before</i></a>
     * actions following the corresponding {@code Future.get()} in another thread.
     *
     * @see FutureTask
     * @see Executor
     * @since 1.5
     * @author Doug Lea
     * @param <V> The result type returned by this Future's <tt>get</tt> method
     */
    public interface Future<V> {
    
        /**
         * Attempts to cancel execution of this task.  This attempt will
         * fail if the task has already completed, has already been cancelled,
         * or could not be cancelled for some other reason. If successful,
         * and this task has not started when <tt>cancel</tt> is called,
         * this task should never run.  If the task has already started,
         * then the <tt>mayInterruptIfRunning</tt> parameter determines
         * whether the thread executing this task should be interrupted in
         * an attempt to stop the task.
         *
         * <p>After this method returns, subsequent calls to {@link #isDone} will
         * always return <tt>true</tt>.  Subsequent calls to {@link #isCancelled}
         * will always return <tt>true</tt> if this method returned <tt>true</tt>.
         *
         * @param mayInterruptIfRunning <tt>true</tt> if the thread executing this
         * task should be interrupted; otherwise, in-progress tasks are allowed
         * to complete
         * @return <tt>false</tt> if the task could not be cancelled,
         * typically because it has already completed normally;
         * <tt>true</tt> otherwise
         */
        boolean cancel(boolean mayInterruptIfRunning);
    
        /**
         * Returns <tt>true</tt> if this task was cancelled before it completed
         * normally.
         *
         * @return <tt>true</tt> if this task was cancelled before it completed
         */
        boolean isCancelled();
    
        /**
         * Returns <tt>true</tt> if this task completed.
         *
         * Completion may be due to normal termination, an exception, or
         * cancellation -- in all of these cases, this method will return
         * <tt>true</tt>.
         *
         * @return <tt>true</tt> if this task completed
         */
        boolean isDone();
    
        /**
         * Waits if necessary for the computation to complete, and then
         * retrieves its result.
         *
         * @return the computed result
         * @throws CancellationException if the computation was cancelled
         * @throws ExecutionException if the computation threw an
         * exception
         * @throws InterruptedException if the current thread was interrupted
         * while waiting
         */
        V get() throws InterruptedException, ExecutionException;
    
        /**
         * Waits if necessary for at most the given time for the computation
         * to complete, and then retrieves its result, if available.
         *
         * @param timeout the maximum time to wait
         * @param unit the time unit of the timeout argument
         * @return the computed result
         * @throws CancellationException if the computation was cancelled
         * @throws ExecutionException if the computation threw an
         * exception
         * @throws InterruptedException if the current thread was interrupted
         * while waiting
         * @throws TimeoutException if the wait timed out
         */
        V get(long timeout, TimeUnit unit)
            throws InterruptedException, ExecutionException, TimeoutException;
    }

因为Future往往与异步操作有关，而在Java中，异步操作最场景的做法就是扔给一个线程处理，所以JDK又提供了一个更高层一点的类——`FutureTask`，将这两者结合起来：
    
    package java.util.concurrent;
    import java.util.concurrent.locks.*;
    
    /**
     * A cancellable asynchronous computation.  This class provides a base
     * implementation of {@link Future}, with methods to start and cancel
     * a computation, query to see if the computation is complete, and
     * retrieve the result of the computation.  The result can only be
     * retrieved when the computation has completed; the <tt>get</tt>
     * method will block if the computation has not yet completed.  Once
     * the computation has completed, the computation cannot be restarted
     * or cancelled.
     *
     * <p>A <tt>FutureTask</tt> can be used to wrap a {@link Callable} or
     * {@link java.lang.Runnable} object.  Because <tt>FutureTask</tt>
     * implements <tt>Runnable</tt>, a <tt>FutureTask</tt> can be
     * submitted to an {@link Executor} for execution.
     *
     * <p>In addition to serving as a standalone class, this class provides
     * <tt>protected</tt> functionality that may be useful when creating
     * customized task classes.
     *
     * @since 1.5
     * @author Doug Lea
     * @param <V> The result type returned by this FutureTask's <tt>get</tt> method
     */
    public class FutureTask<V> implements RunnableFuture<V> {
        /** Synchronization control for FutureTask */
        private final Sync sync;
    
        /**
         * Creates a <tt>FutureTask</tt> that will upon running, execute the
         * given <tt>Callable</tt>.
         *
         * @param  callable the callable task
         * @throws NullPointerException if callable is null
         */
        public FutureTask(Callable<V> callable) {
            if (callable == null)
                throw new NullPointerException();
            sync = new Sync(callable);
        }
    
        /**
         * Creates a <tt>FutureTask</tt> that will upon running, execute the
         * given <tt>Runnable</tt>, and arrange that <tt>get</tt> will return the
         * given result on successful completion.
         *
         * @param  runnable the runnable task
         * @param result the result to return on successful completion. If
         * you don't need a particular result, consider using
         * constructions of the form:
         * <tt>Future&lt;?&gt; f = new FutureTask&lt;Object&gt;(runnable, null)</tt>
         * @throws NullPointerException if runnable is null
         */
        public FutureTask(Runnable runnable, V result) {
            sync = new Sync(Executors.callable(runnable, result));
        }
    
        public boolean isCancelled() {
            return sync.innerIsCancelled();
        }
    
        public boolean isDone() {
            return sync.innerIsDone();
        }
    
        public boolean cancel(boolean mayInterruptIfRunning) {
            return sync.innerCancel(mayInterruptIfRunning);
        }
    
        /**
         * @throws CancellationException {@inheritDoc}
         */
        public V get() throws InterruptedException, ExecutionException {
            return sync.innerGet();
        }
    
        /**
         * @throws CancellationException {@inheritDoc}
         */
        public V get(long timeout, TimeUnit unit)
            throws InterruptedException, ExecutionException, TimeoutException {
            return sync.innerGet(unit.toNanos(timeout));
        }
    
        /**
         * Protected method invoked when this task transitions to state
         * <tt>isDone</tt> (whether normally or via cancellation). The
         * default implementation does nothing.  Subclasses may override
         * this method to invoke completion callbacks or perform
         * bookkeeping. Note that you can query status inside the
         * implementation of this method to determine whether this task
         * has been cancelled.
         */
        protected void done() { }
    
        /**
         * Sets the result of this Future to the given value unless
         * this future has already been set or has been cancelled.
         * This method is invoked internally by the <tt>run</tt> method
         * upon successful completion of the computation.
         * @param v the value
         */
        protected void set(V v) {
            sync.innerSet(v);
        }
    
        /**
         * Causes this future to report an <tt>ExecutionException</tt>
         * with the given throwable as its cause, unless this Future has
         * already been set or has been cancelled.
         * This method is invoked internally by the <tt>run</tt> method
         * upon failure of the computation.
         * @param t the cause of failure
         */
        protected void setException(Throwable t) {
            sync.innerSetException(t);
        }
    
        // The following (duplicated) doc comment can be removed once
        //
        // 6270645: Javadoc comments should be inherited from most derived
        //          superinterface or superclass
        // is fixed.
        /**
         * Sets this Future to the result of its computation
         * unless it has been cancelled.
         */
        public void run() {
            sync.innerRun();
        }
    
        /**
         * Executes the computation without setting its result, and then
         * resets this Future to initial state, failing to do so if the
         * computation encounters an exception or is cancelled.  This is
         * designed for use with tasks that intrinsically execute more
         * than once.
         * @return true if successfully run and reset
         */
        protected boolean runAndReset() {
            return sync.innerRunAndReset();
        }
    
        /**
         * Synchronization control for FutureTask. Note that this must be
         * a non-static inner class in order to invoke the protected
         * <tt>done</tt> method. For clarity, all inner class support
         * methods are same as outer, prefixed with "inner".
         *
         * Uses AQS sync state to represent run status
         */
        private final class Sync extends AbstractQueuedSynchronizer {
            private static final long serialVersionUID = -7828117401763700385L;
    
            /** State value representing that task is running */
            private static final int RUNNING   = 1;
            /** State value representing that task ran */
            private static final int RAN       = 2;
            /** State value representing that task was cancelled */
            private static final int CANCELLED = 4;
    
            /** The underlying callable */
            private final Callable<V> callable;
            /** The result to return from get() */
            private V result;
            /** The exception to throw from get() */
            private Throwable exception;
    
            /**
             * The thread running task. When nulled after set/cancel, this
             * indicates that the results are accessible.  Must be
             * volatile, to ensure visibility upon completion.
             */
            private volatile Thread runner;
    
            Sync(Callable<V> callable) {
                this.callable = callable;
            }
    
            private boolean ranOrCancelled(int state) {
                return (state & (RAN | CANCELLED)) != 0;
            }
    
            /**
             * Implements AQS base acquire to succeed if ran or cancelled
             */
            protected int tryAcquireShared(int ignore) {
                return innerIsDone()? 1 : -1;
            }
    
            /**
             * Implements AQS base release to always signal after setting
             * final done status by nulling runner thread.
             */
            protected boolean tryReleaseShared(int ignore) {
                runner = null;
                return true;
            }
    
            boolean innerIsCancelled() {
                return getState() == CANCELLED;
            }
    
            boolean innerIsDone() {
                return ranOrCancelled(getState()) && runner == null;
            }
    
            V innerGet() throws InterruptedException, ExecutionException {
                acquireSharedInterruptibly(0);
                if (getState() == CANCELLED)
                    throw new CancellationException();
                if (exception != null)
                    throw new ExecutionException(exception);
                return result;
            }
    
            V innerGet(long nanosTimeout) throws InterruptedException, ExecutionException, TimeoutException {
                if (!tryAcquireSharedNanos(0, nanosTimeout))
                    throw new TimeoutException();
                if (getState() == CANCELLED)
                    throw new CancellationException();
                if (exception != null)
                    throw new ExecutionException(exception);
                return result;
            }
    
            void innerSet(V v) {
    	    for (;;) {
    		int s = getState();
    		if (s == RAN)
    		    return;
                    if (s == CANCELLED) {
    		    // aggressively release to set runner to null,
    		    // in case we are racing with a cancel request
    		    // that will try to interrupt runner
                        releaseShared(0);
                        return;
                    }
    		if (compareAndSetState(s, RAN)) {
                        result = v;
                        releaseShared(0);
                        done();
    		    return;
                    }
                }
            }
    
            void innerSetException(Throwable t) {
    	    for (;;) {
    		int s = getState();
    		if (s == RAN)
    		    return;
                    if (s == CANCELLED) {
    		    // aggressively release to set runner to null,
    		    // in case we are racing with a cancel request
    		    // that will try to interrupt runner
                        releaseShared(0);
                        return;
                    }
    		if (compareAndSetState(s, RAN)) {
                        exception = t;
                        result = null;
                        releaseShared(0);
                        done();
    		    return;
                    }
    	    }
            }
    
            boolean innerCancel(boolean mayInterruptIfRunning) {
    	    for (;;) {
    		int s = getState();
    		if (ranOrCancelled(s))
    		    return false;
    		if (compareAndSetState(s, CANCELLED))
    		    break;
    	    }
                if (mayInterruptIfRunning) {
                    Thread r = runner;
                    if (r != null)
                        r.interrupt();
                }
                releaseShared(0);
                done();
                return true;
            }
    
            void innerRun() {
                if (!compareAndSetState(0, RUNNING))
                    return;
                try {
                    runner = Thread.currentThread();
                    if (getState() == RUNNING) // recheck after setting thread
                        innerSet(callable.call());
                    else
                        releaseShared(0); // cancel
                } catch (Throwable ex) {
                    innerSetException(ex);
                }
            }
    
            boolean innerRunAndReset() {
                if (!compareAndSetState(0, RUNNING))
                    return false;
                try {
                    runner = Thread.currentThread();
                    if (getState() == RUNNING)
                        callable.call(); // don't set result
                    runner = null;
                    return compareAndSetState(RUNNING, 0);
                } catch (Throwable ex) {
                    innerSetException(ex);
                    return false;
                }
            }
        }
    }

RunnableFuture就是拓展了Runable和Future接口的接口：

    package java.util.concurrent;

    /**
     * A {@link Future} that is {@link Runnable}. Successful execution of
     * the <tt>run</tt> method causes completion of the <tt>Future</tt>
     * and allows access to its results.
     * @see FutureTask
     * @see Executor
     * @since 1.6
     * @author Doug Lea
     * @param <V> The result type returned by this Future's <tt>get</tt> method
     */
    public interface RunnableFuture<V> extends Runnable, Future<V> {
        /**
         * Sets this Future to the result of its computation
         * unless it has been cancelled.
         */
        void run();
    }

但是其实这种方式还是一种轮询的方式，比较适合的应用场景是：

1. 主线程并不关心子线程返回的结果，它只是想将任务分给子线程异步操作而已，自己好处理其他事情。也就是说主线程不会polling这个Future结果，而是忙着做其他事情去了。例如：

        /**
         * Asynchronous close the {@link AsyncHttpProvider} by spawning a thread and avoid blocking.
         */
        public void closeAsynchronously() {
            final ExecutorService e = Executors.newSingleThreadExecutor();
            e.submit(new Runnable() {
                public void run() {
                    try {
                        close();
                    } catch (Throwable t) {
                        logger.warn("", t);
                    } finally {
                        e.shutdown();
                    }
                }
            });
        }

2. 主线程把很多任务分配给了子线程去处理，等待所有的子线程处理完成之后，对他们的处理结果进行合并。类似于Map-Reduce的做法。其实就是ExecutorService的常见做法。等待多个子线程都处理完，最常见的做法就是搞一个`CountDownLatch`计时器。


在异步操作中，如果把异步操作结果返回给主线程（调用方）是一个最关键的问题。除了这种轮询的方式之后，还有一个场景的方式就是异步通知，但是往往是使用 队列+信号量 的做法，主线程虽然没有polling而浪费CPU，但是却也阻塞(sleep)在等待信号通知上了。

实际例子
-------

像mule ESB、ning's AsyncHttpClient的异步接口都是实现java.util.concurrent.Future接口。

也有自己定义`Future`接口的，比如mina：

    package org.apache.mina.core.future;
    
    import java.util.concurrent.TimeUnit;
    
    import org.apache.mina.core.session.IoSession;
    
    /**
     * Represents the completion of an asynchronous I/O operation on an 
     * {@link IoSession}.
     * Can be listened for completion using a {@link IoFutureListener}.
     * 
     * @author The Apache MINA Project (dev@mina.apache.org)
     * @version $Rev:671827 $, $Date:2008-06-26 09:49:48 +0100 (jeu., 26 juin 2008) $
     */
    public interface IoFuture {
    	
        /**
         * Returns the {@link IoSession} which is associated with this future.
         */
        IoSession getSession();
    
        /**
         * Wait for the asynchronous operation to complete.
         * The attached listeners will be notified when the operation is 
         * completed.
         */
        IoFuture await() throws InterruptedException;
    
        /**
         * Wait for the asynchronous operation to complete with the specified timeout.
         *
         * @return <tt>true</tt> if the operation is completed.
         */
        boolean await(long timeout, TimeUnit unit) throws InterruptedException;
    
        /**
         * Wait for the asynchronous operation to complete with the specified timeout.
         *
         * @return <tt>true</tt> if the operation is completed.
         */
        boolean await(long timeoutMillis) throws InterruptedException;
    
        /**
         * Wait for the asynchronous operation to complete uninterruptibly.
         * The attached listeners will be notified when the operation is 
         * completed.
         * 
         * @return the current IoFuture
         */
        IoFuture awaitUninterruptibly();
    
        /**
         * Wait for the asynchronous operation to complete with the specified timeout
         * uninterruptibly.
         *
         * @return <tt>true</tt> if the operation is completed.
         */
        boolean awaitUninterruptibly(long timeout, TimeUnit unit);
    
        /**
         * Wait for the asynchronous operation to complete with the specified timeout
         * uninterruptibly.
         *
         * @return <tt>true</tt> if the operation is finished.
         */
        boolean awaitUninterruptibly(long timeoutMillis);
    
        /**
         * Returns if the asynchronous operation is completed.
         */
        boolean isDone();
    
        /**
         * Adds an event <tt>listener</tt> which is notified when
         * this future is completed. If the listener is added
         * after the completion, the listener is directly notified.
         */
        IoFuture addListener(IoFutureListener<?> listener);
    
        /**
         * Removes an existing event <tt>listener</tt> so it won't be notified when
         * the future is completed.
         */
        IoFuture removeListener(IoFutureListener<?> listener);
    }

并且提供了一个默认的实现，使用了一个ready状态变量和相应的对象锁（用于wait方法）。还有死锁检测和监听器回调。

   
    package org.apache.mina.core.future;
    
    import java.util.ArrayList;
    import java.util.List;
    import java.util.concurrent.TimeUnit;
    
    import org.apache.mina.core.polling.AbstractPollingIoProcessor;
    import org.apache.mina.core.service.IoProcessor;
    import org.apache.mina.core.session.IoSession;
    import org.apache.mina.util.ExceptionMonitor;
    
    
    /**
     * A default implementation of {@link IoFuture} associated with
     * an {@link IoSession}.
     * 
     * @author The Apache MINA Project (dev@mina.apache.org)
     * @version $Rev:671827 $, $Date:2008-06-26 09:49:48 +0100 (jeu., 26 juin 2008) $
     */
    public class DefaultIoFuture implements IoFuture {
    
        /** A number of seconds to wait between two deadlock controls ( 5 seconds ) */
        private static final long DEAD_LOCK_CHECK_INTERVAL = 5000L;
    
        /** The associated session */
        private final IoSession session;
        
        /** A lock used by the wait() method */
        private final Object lock;
        private IoFutureListener<?> firstListener;
        private List<IoFutureListener<?>> otherListeners;
        private Object result;
        private boolean ready;
        private int waiters;
    
        /**
         * Creates a new instance associated with an {@link IoSession}.
         *
         * @param session an {@link IoSession} which is associated with this future
         */
        public DefaultIoFuture(IoSession session) {
            this.session = session;
            this.lock = this;
        }
    
        /**
         * {@inheritDoc}
         */
        public IoSession getSession() {
            return session;
        }
    
        /**
         * @deprecated Replaced with {@link #awaitUninterruptibly()}.
         */
        @Deprecated
        public void join() {
            awaitUninterruptibly();
        }
    
        /**
         * @deprecated Replaced with {@link #awaitUninterruptibly(long)}.
         */
        @Deprecated
        public boolean join(long timeoutMillis) {
            return awaitUninterruptibly(timeoutMillis);
        }
    
        /**
         * {@inheritDoc}
         */
        public IoFuture await() throws InterruptedException {
            synchronized (lock) {
                while (!ready) {
                    waiters++;
                    try {
                        // Wait for a notify, or if no notify is called,
                        // assume that we have a deadlock and exit the 
                        // loop to check for a potential deadlock.
                        lock.wait(DEAD_LOCK_CHECK_INTERVAL);
                    } finally {
                        waiters--;
                        if (!ready) {
                            checkDeadLock();
                        }
                    }
                }
            }
            return this;
        }
    
        /**
         * {@inheritDoc}
         */
        public boolean await(long timeout, TimeUnit unit)
                throws InterruptedException {
            return await(unit.toMillis(timeout));
        }
    
        /**
         * {@inheritDoc}
         */
        public boolean await(long timeoutMillis) throws InterruptedException {
            return await0(timeoutMillis, true);
        }
    
        /**
         * {@inheritDoc}
         */
        public IoFuture awaitUninterruptibly() {
            try {
                await0(Long.MAX_VALUE, false);
            } catch ( InterruptedException ie) {
                // Do nothing : this catch is just mandatory by contract
            }
            
            return this;
        }
    
        /**
         * {@inheritDoc}
         */
        public boolean awaitUninterruptibly(long timeout, TimeUnit unit) {
            return awaitUninterruptibly(unit.toMillis(timeout));
        }
    
        /**
         * {@inheritDoc}
         */
        public boolean awaitUninterruptibly(long timeoutMillis) {
            try {
                return await0(timeoutMillis, false);
            } catch (InterruptedException e) {
                throw new InternalError();
            }
        }
    
        /**
         * Wait for the Future to be ready. If the requested delay is 0 or 
         * negative, this method immediately returns the value of the 
         * 'ready' flag. 
         * Every 5 second, the wait will be suspended to be able to check if 
         * there is a deadlock or not.
         * 
         * @param timeoutMillis The delay we will wait for the Future to be ready
         * @param interruptable Tells if the wait can be interrupted or not
         * @return <code>true</code> if the Future is ready
         * @throws InterruptedException If the thread has been interrupted
         * when it's not allowed.
         */
        private boolean await0(long timeoutMillis, boolean interruptable) throws InterruptedException {
            long endTime = System.currentTimeMillis() + timeoutMillis;
    
            synchronized (lock) {
                if (ready) {
                    return ready;
                } else if (timeoutMillis <= 0) {
                    return ready;
                }
    
                waiters++;
                try {
                    for (;;) {
                        try {
                            long timeOut = Math.min(timeoutMillis, DEAD_LOCK_CHECK_INTERVAL);
                            lock.wait(timeOut);
                        } catch (InterruptedException e) {
                            if (interruptable) {
                                throw e;
                            }
                        }
    
                        if (ready) {
                            return true;
                        } else {
                            if (endTime < System.currentTimeMillis()) {
                                return ready;
                            }
                        }
                    }
                } finally {
                    waiters--;
                    if (!ready) {
                        checkDeadLock();
                    }
                }
            }
        }
    
        
        /**
         * 
         * TODO checkDeadLock.
         *
         */
        private void checkDeadLock() {
            // Only read / write / connect / write future can cause dead lock. 
            if (!(this instanceof CloseFuture || this instanceof WriteFuture ||
                  this instanceof ReadFuture || this instanceof ConnectFuture)) {
                return;
            }
            
            // Get the current thread stackTrace. 
            // Using Thread.currentThread().getStackTrace() is the best solution,
            // even if slightly less efficient than doing a new Exception().getStackTrace(),
            // as internally, it does exactly the same thing. The advantage of using
            // this solution is that we may benefit some improvement with some
            // future versions of Java.
            StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
    
            // Simple and quick check.
            for (StackTraceElement s: stackTrace) {
                if (AbstractPollingIoProcessor.class.getName().equals(s.getClassName())) {
                    IllegalStateException e = new IllegalStateException( "t" );
                    e.getStackTrace();
                    throw new IllegalStateException(
                        "DEAD LOCK: " + IoFuture.class.getSimpleName() +
                        ".await() was invoked from an I/O processor thread.  " +
                        "Please use " + IoFutureListener.class.getSimpleName() +
                        " or configure a proper thread model alternatively.");
                }
            }
    
            // And then more precisely.
            for (StackTraceElement s: stackTrace) {
                try {
                    Class<?> cls = DefaultIoFuture.class.getClassLoader().loadClass(s.getClassName());
                    if (IoProcessor.class.isAssignableFrom(cls)) {
                        throw new IllegalStateException(
                            "DEAD LOCK: " + IoFuture.class.getSimpleName() +
                            ".await() was invoked from an I/O processor thread.  " +
                            "Please use " + IoFutureListener.class.getSimpleName() +
                            " or configure a proper thread model alternatively.");
                    }
                } catch (Exception cnfe) {
                    // Ignore
                }
            }
        }
    
        /**
         * {@inheritDoc}
         */
        public boolean isDone() {
            synchronized (lock) {
                return ready;
            }
        }
    
        /**
         * Sets the result of the asynchronous operation, and mark it as finished.
         */
        public void setValue(Object newValue) {
            synchronized (lock) {
                // Allow only once.
                if (ready) {
                    return;
                }
    
                result = newValue;
                ready = true;
                if (waiters > 0) {
                    lock.notifyAll();
                }
            }
    
            notifyListeners();
        }
    
        /**
         * Returns the result of the asynchronous operation.
         */
        protected Object getValue() {
            synchronized (lock) {
                return result;
            }
        }
    
        /**
         * {@inheritDoc}
         */
        public IoFuture addListener(IoFutureListener<?> listener) {
            if (listener == null) {
                throw new NullPointerException("listener");
            }
    
            boolean notifyNow = false;
            synchronized (lock) {
                if (ready) {
                    notifyNow = true;
                } else {
                    if (firstListener == null) {
                        firstListener = listener;
                    } else {
                        if (otherListeners == null) {
                            otherListeners = new ArrayList<IoFutureListener<?>>(1);
                        }
                        otherListeners.add(listener);
                    }
                }
            }
    
            if (notifyNow) {
                notifyListener(listener);
            }
            return this;
        }
    
        /**
         * {@inheritDoc}
         */
        public IoFuture removeListener(IoFutureListener<?> listener) {
            if (listener == null) {
                throw new NullPointerException("listener");
            }
    
            synchronized (lock) {
                if (!ready) {
                    if (listener == firstListener) {
                        if (otherListeners != null && !otherListeners.isEmpty()) {
                            firstListener = otherListeners.remove(0);
                        } else {
                            firstListener = null;
                        }
                    } else if (otherListeners != null) {
                        otherListeners.remove(listener);
                    }
                }
            }
    
            return this;
        }
    
        private void notifyListeners() {
            // There won't be any visibility problem or concurrent modification
            // because 'ready' flag will be checked against both addListener and
            // removeListener calls.
            if (firstListener != null) {
                notifyListener(firstListener);
                firstListener = null;
    
                if (otherListeners != null) {
                    for (IoFutureListener<?> l : otherListeners) {
                        notifyListener(l);
                    }
                    otherListeners = null;
                }
            }
        }
    
        @SuppressWarnings("unchecked")
        private void notifyListener(IoFutureListener l) {
            try {
                l.operationComplete(this);
            } catch (Throwable t) {
                ExceptionMonitor.getInstance().exceptionCaught(t);
            }
        }
    }











