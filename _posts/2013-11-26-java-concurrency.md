---
layout: post
title: Java并发学习笔记
---


线程
----

### 线程VS进程


> ### lightweight processes
>
* fork is expensive. Memory is copied from the parent to the child, all descriptors are duplicated in the child, and so on. Current implementations use a technique called copy-on-write, which avoids a copy of the parent's data space to the child until the child needs its own copy. But, regardless of this optimization, fork is expensive.
>
* IPC is required to pass information between the parent and child after the fork. Passing information from the parent to the child before the fork is easy, since the child starts with a copy of the parent's data space and with a copy of all the parent's descriptors. But, returning information from the child to the parent takes more work.


同个进程中的所有线程共享如下数据：

1. Process instructions
2. Most data
3. Open files (e.g., descriptors)
4. Signal handlers and signal dispositions
5. Current working directory
6. User and group IDs

但是每个线程有自己的私有数据：

1. Thread ID
2. Set of registers, including program counter and stack pointer
3. Stack (for local variables and return addresses)
4. errno
5. Signal mask
6. Priority

### 基本线程操作

#### 1. 创建线程

    Thread thread = new Thread(); // Creating a thread
    thread.start();  // start the thread 

但是上面的代码没有任何意思，因为新创建的子线程没有执行任何代码就退出了。我们要为它指定要执行的代码，在Java中有如下两种方式：

**1. Thread Subclass**

创建Thread的子类，重载run()方法
    
	public class MyThread extends Thread {
    
	    public void run(){
	       System.out.println("MyThread running"); 
	    }
	}

	MyThread myThread = new MyThread();
    myTread.start();

在Java中，创建匿名子类很常见：

    Thread thread = new Thread(){
	    public void run(){
	        System.out.println("Thread Running");  
	    }
  	}
 
    thread.start();

**2. Runnable Interface Implemention (Prefer)**

传递一个实现了Runnable接口的对象给线程构造函数

	public class MyRunnable implements Runnable {

	    public void run(){
	       System.out.println("MyRunnable running");
	    }
    }

    Thread thread = new Thread(new MyRunnable());
    thread.start(); 

同样，可以创建匿名runable对象：

	Runnable myRunnable = new Runnable(){

	    public void run(){
	        System.out.println("Runnable running");
        }
    }
  
    Thread thread = new Thread(myRunnable);
    thread.start();

**注意** 

1. run() VS start()

	    Thread newThread = new Thread(MyRunnable());
	    thread.run();  //should be start();

	run()方法是在当前的线程执行，而不是新创建的线程!!!(it is NOT executed by the new thread you just created. Instead the run() method is executed by the thread that created the thread. In other words, the thread that executed the above two lines of code.)

2. Java的线程可以有名称

		Thread thread = new Thread("New Thread") 

		// ...
		MyRunnable runnable = new MyRunnable();
   		Thread thread = new Thread(runnable, "New Thread"); 

   		// ...
   		System.out.println(thread.getName());
        String threadName = Thread.currentThread().getName();


3. pthread的线程创建函数

		int pthread_create(pthread_t *tid, const pthread_attr_t *attr, void *(*func) (void *), void *arg);

####


#### 2. 等待线程结束



	int pthread_join (pthread_t tid, void ** status);


#### 3. 线程间通信

**1. 共享变量**

由于线程之间共享同一个进程空间，所以只要他们引用同一个变量，就可以通过读写这个变量实现信息交流了。不过要注意并发访问问题。


**2. wait(), nofity(), notifyAll()**


	public class MonitorObject{
	}

	public class MyWaitNotify{

	  MonitorObject myMonitorObject = new MonitorObject();

	  public void doWait(){
	    synchronized(myMonitorObject){
	      try{
	        myMonitorObject.wait();
	      } catch(InterruptedException e){...}
	    }
	  }

	  public void doNotify(){
	    synchronized(myMonitorObject){
	      myMonitorObject.notify();
	    }
	  }
	}


> As you can see both the waiting and notifying thread calls wait() and notify() from within a synchronized block. This is mandatory! A thread cannot call wait(), notify() or notifyAll() without holding the lock on the object the method is called on. If it does, an IllegalMonitorStateException is thrown.


### 线程池

线程池可以看成是一个待执行任务队列和一个执行线程队列的集合。我们可以往线程池里面扔任务，线程池中的执行线程会从任务队列中获取任务执行。

下面是一个简单的线程池实现：


	public class ThreadPool {

	  private BlockingQueue taskQueue = null;
	  private List<PoolThread> threads = new ArrayList<PoolThread>();
	  private boolean isStopped = false;

	  public ThreadPool(int noOfThreads, int maxNoOfTasks){
	    taskQueue = new BlockingQueue(maxNoOfTasks);

	    for(int i=0; i<noOfThreads; i++){
	      threads.add(new PoolThread(taskQueue));
	    }
	    for(PoolThread thread : threads){
	      thread.start();
	    }
	  }

	  public void synchronized execute(Runnable task){
	    if(this.isStopped) throw
	      new IllegalStateException("ThreadPool is stopped");

	    this.taskQueue.enqueue(task);
	  }

	  public synchronized void stop(){
	    this.isStopped = true;
	    for(PoolThread thread : threads){
	      thread.stop();
	    }
	  }

	}

	public class PoolThread extends Thread {

	  private BlockingQueue taskQueue = null;
	  private boolean       isStopped = false;

	  public PoolThread(BlockingQueue queue){
	    taskQueue = queue;
	  }

	  public void run(){
	    while(!isStopped()){
	      try{
	        Runnable runnable = (Runnable) taskQueue.dequeue();
	        runnable.run();
	      } catch(Exception e){
	        //log or otherwise report exception,
	        //but keep pool thread alive.
	      }
	    }
	  }

	  public synchronized void stop(){
	    isStopped = true;
	    this.interrupt(); //break pool thread out of dequeue() call.
	  }

	  public synchronized void isStopped(){
	    return isStopped;
	  }
	}

### 线程执行结果——Callable和Future

**同步返回结果**

Runable并不能返回执行结果。如果你想要线程执行完毕后返回执行结果，可以使用java.util.concurrent.Callable。

**异步返回结果**

在java中，异步操作往往喜欢返回一个Future对象，顾名思义，表示结果在未来的某个时刻才完成，虽然future是立马返回的，但是其实是一个未知的结果。调用方可以（需要）查询这个future对象，看是不是已经出结果了。其实是一种polling方式来的。

这些Future对象，一般都是实现java.util.concurrent.Future接口：

Future返回对象可以是Callable的返回值：


	import java.util.concurrent.Callable;

	public class MyCallable implements Callable<Long> {
	  @Override
	  public Long call() throws Exception {
	    long sum = 0;
	    for (long i = 0; i <= 100; i++) {
	      sum += i;
	    }
	    return sum;
	  }
	} 
	

	public class CallableFutures {
	  private static final int NTHREDS = 10;

	  public static void main(String[] args) {

	    ExecutorService executor = Executors.newFixedThreadPool(NTHREDS);
	    List<Future<Long>> list = new ArrayList<Future<Long>>();
	    for (int i = 0; i < 20000; i++) {
	      Callable<Long> worker = new MyCallable();
	      Future<Long> submit = executor.submit(worker);
	      list.add(submit);
	    }
	    long sum = 0;
	    System.out.println(list.size());
	    // now retrieve the result
	    for (Future<Long> future : list) {
	      try {
	        sum += future.get();
	      } catch (InterruptedException e) {
	        e.printStackTrace();
	      } catch (ExecutionException e) {
	        e.printStackTrace();
	      }
	    }
	    System.out.println(sum);
	    executor.shutdown();
	  }
	} 


### JMM(Java内存模型)

1. 可见性
2. 访问控制


参考资料
--------

1. [Java Concurrency/Multithreading Tutorial](http://tutorials.jenkov.com/java-concurrency/index.html)
2. [java.util.concurrent - Java 5 Concurrency Utilities](http://tutorials.jenkov.com/java-util-concurrent/index.html)
3. [Multithreaded Servers in Java](http://tutorials.jenkov.com/java-multithreaded-servers/index.html)
4. [Java concurrency (multi-threading) - tutorial](http://www.vogella.com/articles/JavaConcurrency/article.html#concurrencyjava_synchronized)