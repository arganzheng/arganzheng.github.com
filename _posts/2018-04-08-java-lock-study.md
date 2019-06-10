---
title: Java各种锁介绍
layout: post
tags: [java, concurrency]
catalog: true
---

### Lock 接口 VS Synchronized 关键字

在Lock接口出现以前，Java程序是靠synchronized关键字实现锁功能的，而Java SE 5之后，并发包中新增了 Lock 接口（以及相关实现类）来实现锁的功能，它提供了与 Synchronized 关键字类似的同步功能。只是在使用的时候，需要显示地获取和释放锁。虽然它缺少了通过 Synchronized 块或者方法所提供的隐式获取释放锁的便捷性，但是却拥有了锁获取与释放的可操作性（灵活性）、可中断的获取锁、非阻塞的获取锁 以及 带超时获取锁 等多种 Synchronized 关键字所不具备的同步特性。

使用 Synchronized 关键字将会隐式的获取锁，但是它将锁的获取和释放固化了，也即是先获取，再释放。当然这种方式简化了锁的管理，可是扩展性没有显示的获取锁和释放来的好。例如，针对一个场景，手把手进行锁的获取和释放，先获得锁A，然后再获取锁B，当锁B获得后，释放锁A同时获取锁C，当锁C获得后，再释放B同时获取锁D，以此类推。这种场景下，Synchronized 关键字就不是那么容易实现了，而使用 Lock 却容易许多。

总结: Lock接口提供的 Synchronized 关键字所不具备的主要特性:

1. 灵活性: 灵活的决定锁的粒度和顺序
2. 尝试非阻塞的获取锁: 当线程尝试获取锁，如果这一时刻锁没有被其他线程获取到，则成功获取并持有锁
3. 能被中断的获取锁: 与synchronized不同，获取到锁的线程能够响应中断，当获取到锁的线程被中断时，中断异常将会被抛出，同时锁会被释放
4. 超时获取锁: 在指定的截止时间之前获取锁，如果截止时间到了仍旧无法获取锁，则返回


**Lock接口的API**

* `void lock()` : 获取锁，调用该方法当前线程会获取锁，当锁获得后，该方法返回
* `void lockInterruptibly() throws InterruptedException` : 可中断的获取锁，和lock()方法不同之处在于该方法会响应中断，即在锁的获取中可以中断当前线程
* `boolean tryLock()` : 尝试非阻塞的获取锁，调用该方法后立即返回。如果能够获取到返回true，否则返回false
* `boolean tryLock(long time, TimeUnit unit) throws InterruptedException` : 超时获取锁 
* `void unlock()` : 释放锁 
* `Condition newCondition()` : 获取等待通知组件，该组件和当前的锁绑定，当前线程只有获取了锁，才能调用该组件的`wait()`方法，而调用后，当前线程将释放锁


**JDK Lock接口实现**

Lock接口有三个实现类，分别是 `ReentrantLock`(可重入锁)、`ReadLock`(读锁)、`WriteLock`(写锁)。`Segment`继承了`ReentrantLock`。

* Lock
    * ReentrantLock
        * Segment
    * ReadLock
    * WriteLock


### 可重入锁

所谓可重入锁(ReentrantLock)，是指这种锁是可以反复进入的。当然，这里的反复进入仅仅局限于同一个线程。例如，下面的代码是work的:

```java
lock.lock()
lock.lock()
try {
    i++;
} finally {
    lock.unlock();
    lock.unlock();
}
```

在这种情况下，一个线程连续两次获取同一把锁。这是允许的。如果不允许这种操作，那么同一个线程在第2次获取锁时，将会和自己产生死锁。程序就会卡死在第2次申请锁的过程中。但是需要注意的是，如果同一个线程多次获取锁，那么在释放的时候，也必须释放相同的次数。如果释放的次数多了，那么会得到一个 `java.lang.illegalMonitorStateException` 异常；反之，如果释放的次数少了，那么相当于线程还持有这个锁，因此，其他线程也无法进入临界区。

### 公平锁

正常情况下，锁的申请都是非公平的。也就是说，线程1首先申请了锁A，接着线程2也申请了锁A，那么当锁A可用时，是线程1还是线程2获得锁呢？这是不一定的。系统只会从这个锁的等待队列中随机挑选一个。因此不能保证其公平性。而公平锁，则会按照时间的先后顺序，保证先到者先得。公平锁最主要的优点在于它不会导致饥饿现象，只要你排队，总是会轮到你的。Synchronized 关键字锁控制，那么产生的锁就是非公平锁。可重入锁（ReentrantLock）支持公平锁（FairSync）和非公平锁（NonfairSync），它有个构造函数允许我们对其公平性进行设置:

```java
public ReentrantLock(boolean fair)
```

```java
//默认的构造方法
public ReentrantLock() {
    sync = new NonfairSync(); //非公平锁
}
```

公平锁看起来很优美，但是要实现公平锁必须需要系统维护一个有序队列，因此公平锁的实现成本比较高，性能也相对低下，因此默认情况下，锁都是非公平的。

从实际测试来看，系统在锁申请上，一个线程会倾向于再次获取已经持有的锁（即多个线程同时尝试获取一个锁时，可能会多次被同一个线程获取），这种方式是高效的，但是明显不是公平的（均衡的）。

**TIPS** 怎样避免死锁

1. 保证加锁顺序
2. 可中断的锁等待: 如使用 ReentrantLock 的 lockInterruptibly() 方法，在其他地方可以通过 Thread 的 interrupt() 方法对其进行中断
3. 限时锁等待: 使用 tryLock() 方法进行限时的锁等待。


### Condition接口

在没有 Lock 之前，我们使用 Synchronized 来控制同步，配合 Object 的监视器方法监视器方法（wait()、notify()以及 notifyAll() ） 可以实现等待/通知模式。在 Java SE5 后，Java 提供了 Lock 接口，相对于 Synchronized 而言，Lock 提供了条件 Condition（`Lock.newCondition()`），对线程的等待、唤醒操作更加详细和灵活。

下面是 Object 的监视器方法与 Condition 接口的对比：

| 对比项 | Object Monitor Methods | Condition |
| --- | --- | --- |
| 前置条件 | 获取对象的锁 | 调用 Lock.lock() 获取锁； 调用 Lock.newCondition() 绑定 Condition 对象 |
| 调用方式 | 直接调用，如 object.wait() | 直接调用，如: condition.await()  |
| 等待队列个数 | 一个 | 多个。一个 Lock 对象可以创建多个 Condition 实例，所以可以支持多个等待队列。|
| 当前线程释放锁并进入等待状态 | 支持 | 支持 |
| 当前线程释放锁并进入等待状态，在等待状态中不响应中断 | 不支持 | 支持 |
| 当前线程释放锁并进入超时等待状态 | 支持 | 支持 |
| 当前线程释放锁并进入等待状态到将来的某个时间 | 不支持 | 支持 |
| 唤醒等待队列中的一个线程 | 支持 | 支持 |
| 唤醒等待队列中的全部线程 | 支持 | 支持 |

Condition接口的方法与Object的监视器主要方法对比

| Condition | Object | 作用 |
| --- | --- | --- |
| await() | wait() | 造成当前线程在接到信号或被中断之前一直处于等待状态。|
| await(long time, TimeUnit unit) | wait(long timeout) | 造成当前线程在接到信号、被中断或到达指定等待时间之前一直处于等待状态。|
| awaitUninterruptibly() | 无 | 当前线程进入等待状态直到被通知，不可中断。|
| awaitNanos(long nanosTimeout) | 无 | 当前线程在接到信号、被中断或到达指定等待时间之前一直处于等待状态。返回值表示剩余时间，如果在nanosTimesout之前唤醒，那么返回值 = nanosTimeout – 消耗时间，如果返回值 <= 0，则可以认定它已经超时了。 |
| awaitUntil(Date deadline) |  无 | 当前线程在接到信号、被中断或到达指定最后期限之前一直处于等待状态。如果没有到指定时间就被通知，则返回true，否则表示到了指定时间，返回返回false。 |
| signal() | notify() | 唤醒一个等待线程。 |
| signalAll() | notifyAll() | 唤醒所有等待线程。 |

Condition是一种广义上的条件队列。他为线程提供了一种更为灵活的等待/通知模式，线程在调用await方法后执行挂起操作，直到线程等待的某个条件为真时才会被唤醒。

Condition必须要配合锁一起使用，因为对共享状态变量的访问发生在多线程环境下。一个Condition的实例必须与一个Lock绑定，因此Condition一般都是作为Lock的内部实现。

类似于wait、notify 和 notifyAll必须在同步代码块中使用，Condition对象的方法也必须要写在Lock.lock与Lock.unLock()代码之间:

```java
ReentrantLock lock=new ReentrantLock();
Condition condition = lock.newCondition();
lock.lock();
/*获取到锁之后才能调用以下方法
condition.await();
condition.signal();
condition.signalAll();*/
lock.unlock();
```

**Condtion的实现**

Condition 为一个接口，其下仅有一个实现类 ConditionObject，由于 Condition 的操作需要获取相关的锁，而AQS则是同步锁的实现基础，所以 ConditionObject 则定义为 AQS 的内部类。每个 Condition 对象都包含着一个FIFO队列，该队列是 Condition 对象通知/等待功能的关键。在队列中每一个节点都包含着一个线程引用，该线程就是在该 Condition 对象上等待的线程。

一个线程获取锁后，通过调用 Condition的await() 方法，会将当前线程先加入到条件队列中，然后释放锁，最后通过isOnSyncQueue(Node node)方法不断自检看节点是否已经在CLH同步队列了，如果是则尝试获取锁，否则一直挂起。当线程调用signal()方法后，程序首先检查当前线程是否获取了锁，然后通过doSignal(Node first)方法唤醒CLH同步队列的首节点。被唤醒的线程，将从await()方法中的while循环中退出来，然后调用acquireQueued()方法竞争同步状态。


### CountDownLatch

CountDownLatch 类位于 `java.util.concurrent` 包下，利用它可以实现类似计数器的功能。比如有一个任务A，它要等待其他4个任务执行完毕之后才能执行，此时就可以利用 CountDownLatch 来实现这种功能了。

CountDownLatch 类只提供了一个构造器：

```java
public CountDownLatch(int count) { ... };  //参数count为计数值
```

下面这3个方法是CountDownLatch类中最重要的方法：

* public void await() throws InterruptedException : 调用await()方法的线程会被挂起，它会等待直到count值为0才继续执行
* public boolean await(long timeout, TimeUnit unit) throws InterruptedException : 和await()类似，只不过等待一定的时间后count值还没变为0的话就会继续执行 
* public void countDown() : 将count值减1

CountDownLatch 经常用于实现父线程等待多个子线程都完成任务之后再继续执行的场景（fork-join），类似于多线程的 join 操作。

###  CyclicBarrier

CyclicBarrier 字面意思 循环栅栏，通过它可以实现让一组线程等待至某个状态之后再全部同时执行。Cyclic 意为循环，也就是说当所有等待线程都被释放以后，CyclicBarrier可以被重用。我们暂且把这个状态就叫做barrier，当调用await()方法之后，线程就处于barrier了。

CyclicBarrier类位于java.util.concurrent包下，CyclicBarrier提供2个构造器：

```java
public CyclicBarrier(int parties, Runnable barrierAction) { ... }
public CyclicBarrier(int parties) { ... }
```

参数parties指让多少个线程或者任务等待至barrier状态；参数barrierAction为当这些线程都达到barrier状态时会执行的内容。

CyclicBarrier中最重要的方法就是await方法，它有2个重载版本：

* `int await() throws InterruptedException, BrokenBarrierException` : 挂起当前线程，直至所有线程都到达barrier状态再同时执行后续任务
* `int await(long timeout, TimeUnit unit) throws InterruptedException,BrokenBarrierException,TimeoutException` : 让线程等待至一定的时间，如果还有线程没有到达barrier状态就直接让到达barrier的线程执行后续任务


细心的读者可能注意到，CyclicBarrier 跟 CountDownLatch 非常的类似，都可以用来实现线程间的计数等待。但它的功能比 CountDownLatch 要更加复杂也稍微强大。第一个是 CyclicBarrier 可以循环计数，可以被重复使用。第二个是 CyclicBarrier 可以接收一个参数作为 barrierAction。所谓 barrierAction 就是当计数器一次计数完成后，系统会执行的动作。实际测试当所有的线程都到达barrier状态后，系统会从这些线程中选择最后一个执行完的线程去执行 barrierAction。 

### Semaphore

Semaphore 翻译成字面意思为 信号量，Semaphore 可以控同时访问的线程个数，通过 acquire() 获取一个许可，如果没有就等待，而 release() 释放一个许可。

Semaphore类位于 `java.util.concurrent` 包下，它提供了2个构造器：

* `Semaphore(int permits)` : 参数 permits 表示许可数目，即同时可以允许多少线程进行访问
* `Semaphore(int permits, boolean fair)` : 参数 fair 表示是否是公平的，即等待时间越久的越先获取许可


Semaphore 最重要的几个方法如下：

* `void acquire() throws InterruptedException` : 获取一个许可，若无许可能够获得，则会一直等待，直到获得许可。
* `void acquire(int permits) throws InterruptedException` : 获取 permits 个许可，若无许可能够获得，则会一直等待，直到获得许可。
* `void release()` : 释放一个许可。注意，在释放许可之前，必须先获获得许可。
* `void release(int permits)` : 释放 permits 个许可。注意，在释放许可之前，必须先获获得许可。

这4个方法都会被阻塞，如果想立即得到执行结果，可以使用下面几个非堵塞方法：

* `boolean tryAcquire()` : 尝试获取一个许可，若获取成功，则立即返回true，若获取失败，则立即返回false
* `boolean tryAcquire(long timeout, TimeUnit unit) throws InterruptedException` : 尝试获取一个许可，若在指定的时间内获取成功，则立即返回true，否则则立即返回false
* `boolean tryAcquire(int permits)` : 尝试获取permits个许可，若获取成功，则立即返回true，若获取失败，则立即返回false
* `boolean tryAcquire(int permits, long timeout, TimeUnit unit) throws InterruptedException` : 尝试获取 permits 个许可，若在指定的时间内获取成功，则立即返回true，否则则立即返回false

另外还可以通过 `availablePermits()` 方法得到可用的许可数目（一般只做调试使用）:

* `int availablePermits()` : the current number of permits available in this semaphore
          
由于 Semaphore 适用于限制访问某些资源的线程数目，因此可以使用它来做限流。

下面通过一个例子来看一下Semaphore的具体使用：

假若一个工厂有5台机器，但是有8个工人，一台机器同时只能被一个工人使用，只有使用完了，其他工人才能继续使用。那么我们就可以通过Semaphore来实现：

```java
public class Test {
    public static void main(String[] args) {
        int N = 8;            //工人数
        Semaphore semaphore = new Semaphore(5); //机器数目
        for(int i=0;i<N;i++)
            new Worker(i,semaphore).start();
    }
     
    static class Worker extends Thread{
        private int num;
        private Semaphore semaphore;
        public Worker(int num,Semaphore semaphore){
            this.num = num;
            this.semaphore = semaphore;
        }
         
        @Override
        public void run() {
            try {
                semaphore.acquire();
                System.out.println("工人"+this.num+"占用一个机器在生产...");
                Thread.sleep(2000);
                System.out.println("工人"+this.num+"释放出机器");
                semaphore.release();           
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
```

执行结果:

```sh
工人0占用一个机器在生产...
工人1占用一个机器在生产...
工人2占用一个机器在生产...
工人4占用一个机器在生产...
工人5占用一个机器在生产...
工人0释放出机器
工人2释放出机器
工人3占用一个机器在生产...
工人7占用一个机器在生产...
工人4释放出机器
工人5释放出机器
工人1释放出机器
工人6占用一个机器在生产...
工人3释放出机器
工人7释放出机器
工人6释放出机器
```



参考文章
------

1. [Java 各种锁的小结](http://www.imooc.com/article/277500)
