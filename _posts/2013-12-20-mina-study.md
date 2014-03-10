---
title: mina学习笔记
layout: post
---

![mina](/media/images/mina.jpg)


Mina框架
--------

mina的设计非常简单，由于几个模块组成：

1. IoService (Acceptor | Connector)
2. IoProcessor，如果没有指定，默认是NioProcessor。
3. IoFilterChain
4. IoHandler

其中IoAccetor主要负责监听绑定的端口，接收客户端连接。它有一个selector，对listener端口进行轮询select，如果有OP_ACCEPT事件发生，它会创建一个IoSession，然后调用IoProcessor的add方法，把这个IoSession放进IoProcessor的newSessions队列中，然后唤醒IoProcessor处理

NioSocketAcceptor是处理TCP的IoService，它有一个Exector，一个Acceptor runable对象，两个队列（registerQueue，cancelQueue，一个socketAddress和Handler的映射map: `Map<SocketAddress, H> boundHandles`，一个IoProcessor的引用，一个IoHandler的引用（用户提供），一个filterChainBuilder，一个IoServiceListener，还有一个selector）。

IoProcessor则是负责Connected端口的读写和关闭（当读到EOF的时候）操作。它也有一个Selector，不过这个selector监听的是OP_READ事件。它会select所有accept的连接(IoSession)，如果某个连接有 OP_READ 事件产生，则进行一连串的pipeline(Processor=>FilterChain=>Handler)处理。如果对端关闭了连接，它会移除这个session。session的增加和移除也会触发pipleline处理。

![Selector](http://image.slidesharecdn.com/javanio-100701052238-phpapp02/95/slide-17-728.jpg?1277979880)

在mina中，这两者都是一个Runable的对象，而且都放在各自的Executor线程池中（如果没有指定，就是Executors.newCachedThreadPool()）。对于Acceptor，虽然是个线程池，但是这个线程池中只有一个Runable对象。而对于IoProcessor可以指定要创建多个。mina提供了一个简单的实现，叫做：`SimpleIoProcessorPool`，默认是CPU核数+1: Runtime.getRuntime().availableProcessors() + 1;），所有的IoProcessor共用一个线程池。

SimpleIoProcessorPool的add(s session)方法很简单。把IoAcceptor扔过来的IoSession随机分配给其中的一个IoProcessor处理。并且会把该processor放置在IoSession中，避免重复分配。

    /**
     * {@inheritDoc}
     */
    public final void add(S session) {
        getProcessor(session).add(session);
    }

     /**
     * Find the processor associated to a session. If it hasen't be stored into
     * the session's attributes, pick a new processor and stores it.
     */
    @SuppressWarnings("unchecked")
    private IoProcessor<S> getProcessor(S session) {
        IoProcessor<S> processor = (IoProcessor<S>) session.getAttribute(PROCESSOR);
        
        if (processor == null) {
            if (disposed || disposing) {
                throw new IllegalStateException("A disposed processor cannot be accessed.");
            }

            processor = pool[Math.abs((int) session.getId()) % pool.length];
            
            if (processor == null) {
                throw new IllegalStateException("A disposed processor cannot be accessed.");
            }
            
            session.setAttributeIfAbsent(PROCESSOR, processor);
        }

        return processor;
    }

也就是说Acceptor把IoSession随机放在一个IoProcessor的newSessions队列中。回顾一下每个NioProcessor有一个的Executor，一个Processor runable对象，四个Queue（newSessions，removingSessions，flushingSessions，trafficControllingSessions），还有一个监听OP_READ事件的Selector。除了Executor一般就是来自于SimpleIoProcessorPool的Executor，所以所有的NioProcessor公用一个线程池之外，其他的都是私有成员变量。而Processor这个runable的任务就是循环的poll selector，处理活跃的session。

也就是说对于一个普通的mina应用，将有一个Acceptor，多个IoProcessor。这两者都运行在单独的线程池中。

mina这里为了避免Processor空转（队列没有session的时候不断轮询队列），还动态创建Processor Runable对象。
    
    private class Processor implements Runnable {
        public void run() {
            int nSessions = 0;
            lastIdleCheckTime = System.currentTimeMillis();

            for (;;) {
                try {
                    int selected = select(1000);

                    nSessions += add();
                    updateTrafficMask();

                    if (selected > 0) {
                        process();
                    }

                    long currentTime = System.currentTimeMillis();
                    flush(currentTime);
                    nSessions -= remove();
                    notifyIdleSessions(currentTime);

                    if (nSessions == 0) {
                        synchronized (lock) {
                            if (newSessions.isEmpty() && isSelectorEmpty()) {
                                processor = null;
                                break;
                            }
                        }
                    }
                    ...
                    }
                } catch (Throwable t) {
                   ...
                }
                ...
            }
        }
    }

可以看到当newSessions队列为空并且当前没有活跃的sessionKeys需要处理。mina会把Processor runnable关闭，退出循环，结束Processor线程。

同时提供了一个启动Processor runnable的方法，供调用：

	private void startupWorker() {
	        synchronized (lock) {
	            if (processor == null) {
	                processor = new Processor();
	                executor.execute(new NamePreservingRunnable(processor, threadName));
	            }
	        }
	        wakeup();
    }

这个方法，在add(T session)和dispose()和remove(T session)的时候被调用：
	
    public final void add(T session) {
        if (isDisposing()) {
            throw new IllegalStateException("Already disposed.");
        }

        // Adds the session to the newSession queue and starts the worker
        newSessions.add(session);
        startupWorker();
    }

	public final void dispose() {
        if (disposed) {
            return;
        }

        synchronized (disposalLock) {
            if (!disposing) {
                disposing = true;
                startupWorker();
            }
        }

        disposalFuture.awaitUninterruptibly();
        disposed = true;
    }
    
    public final void remove(T session) {
        scheduleRemove(session);
        startupWorker();
    }

可以看出mina在性能方面做的细节处理。不过这种处理方式在特殊情况下——消费者的消费速度与生产者的生产速度差不多的情况下，会出现Runable的反复创建与销毁。不过这里Runable不像Thread，仅仅是一个普通对象，创建销魂成本还是比较小的。

如果不增加ExecutorFilter，那么所有的Filter、IoHandler与IoProcessor是共用一个线程池。事实上是同一个线程。

另外，IoProcessor其实在Mina内部也是一个IoFileter来的，mina把它定义为TailFilter，放在filterChain的最后。所以IoHandler总是最后调用。DefaultIoFilterChain还默认创建了一个HeadFilter，放在最前面。

IoProcessor本身是一个线程池加任务队列的做法。任务队列放的是每个创建的IoSession，由acceptor作为生产者存放进去。

IoSession相当于一个Connection，里面有所有的上下文信息：

	public interface IoSession {

	    long getId();

	    IoService getService();

	    IoHandler getHandler();

	    IoSessionConfig getConfig();

	    IoFilterChain getFilterChain();

	    WriteRequestQueue getWriteRequestQueue();

	    TransportMetadata getTransportMetadata();

	    ReadFuture read();

	    WriteFuture write(Object message);

	    WriteFuture write(Object message, SocketAddress destination);

	    CloseFuture close(boolean immediately);
	   
	    Object getAttribute(Object key);
	    
	    Object getAttribute(Object key, Object defaultValue);

	    Object setAttribute(Object key, Object value);
	  
	    Object setAttribute(Object key);
	  
	    Object setAttributeIfAbsent(Object key, Object value);

	    Object setAttributeIfAbsent(Object key);

	    Object removeAttribute(Object key);

	    boolean removeAttribute(Object key, Object value);
	  
	    boolean replaceAttribute(Object key, Object oldValue, Object newValue);

	    boolean containsAttribute(Object key);

	    Set<Object> getAttributeKeys();

	    boolean isConnected();

	    boolean isClosing();

	    CloseFuture getCloseFuture();

	    SocketAddress getRemoteAddress();

	    SocketAddress getLocalAddress();

	    SocketAddress getServiceAddress();

	    // ...
	}

另一个值得一说的就是前面提到的ExecutorFilter。他可以让产生的事件在一个独立的线程池异步执行，不占用IoProcessor的线程池资源。

Mina定义了一个IoEvent对象，其实也是一个Runable来的：


	public class IoEvent implements Runnable {
	    private final IoEventType type;

	    private final IoSession session;

	    private final Object parameter;

	    public IoEvent(IoEventType type, IoSession session, Object parameter) {
	        if (type == null) {
	            throw new NullPointerException("type");
	        }
	        if (session == null) {
	            throw new NullPointerException("session");
	        }
	        this.type = type;
	        this.session = session;
	        this.parameter = parameter;
	    }

	    public IoEventType getType() {
	        return type;
	    }

	    public IoSession getSession() {
	        return session;
	    }

	    public Object getParameter() {
	        return parameter;
	    }
	    
	    public void run() {
	        fire();
	    }

	    public void fire() {
	        switch (getType()) {
	        case MESSAGE_RECEIVED:
	            getSession().getFilterChain().fireMessageReceived(getParameter());
	            break;
	        case MESSAGE_SENT:
	            getSession().getFilterChain().fireMessageSent((WriteRequest) getParameter());
	            break;
	        case WRITE:
	            getSession().getFilterChain().fireFilterWrite((WriteRequest) getParameter());
	            break;
	        case CLOSE:
	            getSession().getFilterChain().fireFilterClose();
	            break;
	        case EXCEPTION_CAUGHT:
	            getSession().getFilterChain().fireExceptionCaught((Throwable) getParameter());
	            break;
	        case SESSION_IDLE:
	            getSession().getFilterChain().fireSessionIdle((IdleStatus) getParameter());
	            break;
	        case SESSION_OPENED:
	            getSession().getFilterChain().fireSessionOpened();
	            break;
	        case SESSION_CREATED:
	            getSession().getFilterChain().fireSessionCreated();
	            break;
	        case SESSION_CLOSED:
	            getSession().getFilterChain().fireSessionClosed();
	            break;
	        default:
	            throw new IllegalArgumentException("Unknown event type: " + getType());
	        }
	    }
	}

IoFilterEvent稍微封装了一下IoEvent：

	public class IoFilterEvent extends IoEvent {

	    private final NextFilter nextFilter;

	    public IoFilterEvent(NextFilter nextFilter, IoEventType type,
	            IoSession session, Object parameter) {
	        super(type, session, parameter);

	        if (nextFilter == null) {
	            throw new NullPointerException("nextFilter");
	        }
	        this.nextFilter = nextFilter;
	    }

	    public NextFilter getNextFilter() {
	        return nextFilter;
	    }

	    @Override
	    public void fire() {
	        switch (getType()) {
	        case MESSAGE_RECEIVED:
	            getNextFilter().messageReceived(getSession(), getParameter());
	            break;
	        case MESSAGE_SENT:
	            getNextFilter().messageSent(getSession(), (WriteRequest) getParameter());
	            break;
	        case WRITE:
	            getNextFilter().filterWrite(getSession(), (WriteRequest) getParameter());
	            break;
	        case CLOSE:
	            getNextFilter().filterClose(getSession());
	            break;
	        case EXCEPTION_CAUGHT:
	            getNextFilter().exceptionCaught(getSession(), (Throwable) getParameter());
	            break;
	        case SESSION_IDLE:
	            getNextFilter().sessionIdle(getSession(), (IdleStatus) getParameter());
	            break;
	        case SESSION_OPENED:
	            getNextFilter().sessionOpened(getSession());
	            break;
	        case SESSION_CREATED:
	            getNextFilter().sessionCreated(getSession());
	            break;
	        case SESSION_CLOSED:
	            getNextFilter().sessionClosed(getSession());
	            break;
	        default:
	            throw new IllegalArgumentException("Unknown event type: " + getType());
	        }
	    }
	}

当有事件发生的时候，ExecutorFilter会创建一个IoFilterEvent对象(Runable对象)，然后扔到他自己的Excutor线程池执行。

### 收获

1. 数据结构优于流程图 把这几个类的成员变量仔细看一下比看方法要更容易明白。深刻的体会到数据结构优于流程图这句话的高瞻远瞩。
2. 对于有pipeline处理机制的代码，有个context的概念会使得编程方便很多。可以方便的在pipeline的各个组件中传递参数和处理结果。Mina中这个context就是IoSession对象，一般除了明确的成员之外，还会提供map接口，可以放置自定义属性。

