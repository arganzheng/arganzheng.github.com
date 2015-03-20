---
title: 服务器编程模型
layout: post
---

1. 多进程模型
-------------

1.fork子进程在accept阻塞等待连接。
2. 业务和网络层分离为不同进程模型。
3. 负载均衡模型，多个进程绑定不同的端口。


2. 多线程模型
-------------

1.主线程在accept处阻塞，当有新连接时实时创建新线程处理，处理完关闭。
2.创建线程池，都在accept处阻塞，当有新连接时notify一个线程去处理，处理完归还。
3.线程(池)关联一个线程消息队列，主线程在accept处阻塞，将新连接放入消息队列，线程(池)读取线程消息队列处理。

Java最经典的处理就是一个链接一个线程处理：


	class Server implements Runnable {
		public void run() {
			try {
				ServerSocket ss = new ServerSocket(PORT);
				while (!Thread.interrupted()){
					new Thread(new Handler(ss.accept())).start();
					// or, single-threaded, or a thread poo
				}
			} catch (IOException ex) { /* ... */ }
		}
				
		static class Handler implements Runnable {
			final Socket socket;
			Handler(Socket s) { 
				socket = s; 
			}
		
			public void run() {
				try {
					byte[] input = new byte[MAX_INPUT];
					socket.getInputStream().read(input);
					byte[] output = process(input);
					socket.getOutputStream().write(output);
				} catch (IOException ex) { /* ... */ }
			}
				
			private byte[] process(byte[] cmd) { /* ... */ }
		}
	}


3. 线程池
---------


4. 多路复用(Reactor)模型
------------------------

1.事件多路分离与分派。
2.实用的网络模型都是多路分离的(select/poll/epoll)，而且是非阻塞的。
3.Java NIO

一般地，I/O多路复用机制都依赖于一个事件多路分离器(Event Demultiplexer)。分离器对象可将来自事件源的I/O事件分离出来，并分发到对应的read/write事件处理器(Event Handler)。开发人员预先注册需要处理的事件及其事件处理器（或回调函数）；事件分离器负责将请求事件传递给事件处理器。两个与事件分离器有关的模式是Reactor和Proactor。Reactor模式采用同步IO，而Proactor采用异步IO。

	class Reactor implements Runnable {
		final Selector selector;
		final ServerSocketChannel serverSocket;

		Reactor(int port) throws IOException {
			selector = Selector.open();
			serverSocket = ServerSocketChannel.open();
			serverSocket.socket().bind(
			new InetSocketAddress(port));
			serverSocket.configureBlocking(false);
			SelectionKey sk = serverSocket.register(selector, SelectionKey.OP_ACCEPT);
			sk.attach(new Acceptor());
		}
		/*
		Alternatively, use explicit SPI provider:
		SelectorProvider p = SelectorProvider.provider();
		selector = p.openSelector();
		serverSocket = p.openServerSocketChannel();
		*/

		
		public void run() { // normally in a new Thread
			try {	
				while (!Thread.interrupted()) {
				selector.select();
				Set selected = selector.selectedKeys();
				Iterator it = selected.iterator();
				while (it.hasNext())
					dispatch((SelectionKey)(it.next());
					selected.clear();
				}
			} catch (IOException ex) { /* ... */ }
		}
		
		void dispatch(SelectionKey k) {
			Runnable r = (Runnable)(k.attachment());
			if (r != null)
			r.run();
		}

		class Acceptor implements Runnable { // inner class
			public void run() {
				try {
					SocketChannel c = serverSocket.accept();
					if (c != null){
						new Handler(selector, c);
					}
				}catch(IOException ex) { /* ... */ }
			}
		}
	}

	final class Handler implements Runnable {
		final SocketChannel socket;
		final SelectionKey sk;
		ByteBuffer input = ByteBuffer.allocate(MAXIN);
		ByteBuffer output = ByteBuffer.allocate(MAXOUT);
		static final int READING = 0, SENDING = 1;
		int state = READING;

		Handler(Selector sel, SocketChannel c) throws IOException {
			socket = c; 
			c.configureBlocking(false);
			// Optionally try first read now
			sk = socket.register(sel, 0);
			sk.attach(this);
			sk.interestOps(SelectionKey.OP_READ);
			sel.wakeup();
		}

		boolean inputIsComplete() { /* ... */ }
		boolean outputIsComplete() { /* ... */ }
		void process() { /* ... */ }

		// class Handler continued
		public void run() {
			try {
				if (state == READING) {
					read();
				} else if (state == SENDING) {
					send();
				}
			} catch (IOException ex) { /* ... */ }
		}

		void read() throws IOException {
			socket.read(input);
			if (inputIsComplete()) {
				process();
				state = SENDING;
				// Normally also do first write now
				sk.interestOps(SelectionKey.OP_WRITE);
			}
		}
		
		void send() throws IOException {
			socket.write(output);
			if (outputIsComplete()) {
				sk.cancel();
			}
		}
	}


5. 异步IO(Proactor)
-------------------


参考文章
--------

1. [Multithreaded Servers in Java](http://tutorials.jenkov.com/java-multithreaded-servers/index.html)