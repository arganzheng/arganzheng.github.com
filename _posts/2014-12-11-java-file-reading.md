---
title: Java文件读取支持timeout
layout: post
---

我们知道系统的IO操作一般分为阻塞IO和异步IO两种。对于Java来说，1.4之前的IO是Blocking IO，1.4引入的NIO其实也是，但是它有两个好处：
1. select可以同时监听多个文件描述符，而这些文件描述符其中的任意一个进入读就绪状态，select()函数就可以返回。这也是它被称为IO复用的原因。
2. 如果select返回的时候，调用相应的Blocking IO操作一般不会“阻塞”，因为文件描述符已经是ready了。需要的时间只是将数据从内核copy到用户空间（或者相反）

但是对于NIO的IO复用只支持网络IO，对于文件系统的读写，还是传统的Blocking IO方式。这种情况，直到Java 1.7的出现才解决。Java 7引入的AsynchronousFileChannel可以支持文件系统的异步IO。在使用新的异步IO时，主要有两种方式——Future轮询和Callback回调。

但是其实注意到Java对于Blocking IO的支持其实是不完整的——文件读取不支持Timeout！这个问题很严重，因为没有timeout，有可能线程完全堵塞在read调用了。

如果你已经升级到1.7+，那么可以直接使用AIO，不需要timeout，反正也不会hold住线程。如果不是，那么有如下几种解决方案：

#### 法一、另启一个线程，异步读取，然后用Future.get(timeout)实现带timeout的异步read操作：

	int readByte = 1;
    // Read data with timeout
    Callable<Integer> readTask = new Callable<Integer>() {
        @Override
        public Integer call() throws Exception {
            return inputStream.read();
        }
    };
    while (readByte >= 0) {
        Future<Integer> future = executor.submit(readTask);
        try{
        	readByte = future.get(1000, TimeUnit.MILLISECONDS);
        }catch(TimeoutException e){
        	executor.shutdownNow(); 
        }
        if (readByte >= 0)
            System.out.println("Read: " + readByte);
    }


Future.get如果读取超时，会抛出一个java.util.concurrent.TimeoutException。但是需要注意的是这时候读取子线程还block在那里，需要在catch中将读取线程退出，executor.shutdownNow()未必能够成功。总之，需要多测试一下。

也可以另起一个子线程去做堵塞读取，然后让父线程调用thread.join(timeout)等待一段时间，如果timeout就interrupt子线程。

#### 法二、依赖于inputStream.available()只读取拥有的数据，相当于fast failed机制。

	byte[] inputData = new byte[1024];
    int result = is.read(inputData, 0, is.available());  
    // result will indicate number of bytes read; -1 for EOF with no data read.


#### 法三、两者的结合

有时候我们不希望新起一个线程异步执行，同时也希望能够有超时机制，那么可以这么处理：

	// InputStream => InputStreamReader(with charset) => String
	private static String getStreamAsStringWithTimeout(InputStream is, String charset,
			int timeoutMillis) throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(is, charset));
		StringWriter writer = new StringWriter();

		char[] chars = new char[256];

		int count = 0;
		long maxTimeMillis = System.currentTimeMillis() + timeoutMillis;
		while (System.currentTimeMillis() < maxTimeMillis) {
			int readLength = java.lang.Math.min(is.available(), 256);
			count = reader.read(chars, 0, readLength);
			if (count == -1) {
				break;
			}
			writer.write(chars, 0, count);
		}
		return writer.toString();
	}


### 参考文档

1. [Is it possible to read from a InputStream with a timeout?](http://stackoverflow.com/questions/804951/is-it-possible-to-read-from-a-inputstream-with-a-timeout)。
