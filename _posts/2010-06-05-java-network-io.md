---
title: Java网络IO编程
layout: post
---

在前面的文章 [Java IO概述](http://blog.arganzheng.me/posts/java-io.html) 主要介绍了Java的文件IO。在这一篇，我将继续介绍Java的网络IO编程。


Prequirement
------------

1. 在继续阅读这篇文章之前，请务必先阅读前面这篇[Java IO概述](http://blog.arganzheng.me/posts/java-io.html)，因为Java把所有的IO都统一成流(Stream)了。
2. TCP/IP协议栈。知道IP、端口、DNS、Socket、URL、TCP、UDP、HTTP等网络相关知识。


一些重要的类
------------

### InetAddress类

java.net.InetAddress类是Java对IP地址(包括IPv4和IPv6)的封装。一般来说，它同时包含主机名(hostname)和IP地址。


#### 1. 创建方式（工厂方法）

1. public static InetAddress getByName(String hostName) throws UnknownHostException;
2. public static InetAddress[] getAllByName(String hostName) throws UnknownHostException;
3. public static InnetAddress getLocalHost() throws UnknownHostException;


**说明**

1. 所有这三个方法都可能在必要的时候连接本地DNS服务器，进行域名解析。
2. 由于DNS查找成本相对比较高，InetAddress类会缓存查找的结果。不成功的DNS查找缓存默认是10s。这两个缓存时间可以通过networkaddress.cache.ttl和networkaddress.cache.negative.ttl控制。
3. 虽然名称写着getByName(hostName)，看起来是使用DNS查找给定hostName对应的IP地址。但是其实这个方法是可以接收包含点分四段或者十六进制形式的IP地址字符串的。如`InetAddress address = InetAddress.getByName("8.8.8.8");`。JDK1.4及以后的版本，对这种情况提供了单独的接口: 1. public static InetAddress getByName(byte[] address) throws UnknownHostException; 2. public static InetAddress getByName(String hostName, byte[] address) throws UnknownHostException; 数组的长度必须是4或者16(即IPv4或者IPv6)。
4. 当使用IP地址字符串作为参数调用getByName()时，是不需要检查DNS的。这表示可能为实际上不存在也无法连接的主机创建InetAddress对象。但是，当显式地通过getHostName()请求此主机名时，会进行实际主机名的DNS查询。但是这时候DNS查找失败，不会抛UnknownHostException异常。
5. getAllByname(hostName)返回所有对应hostName的地址。虽然一个域名绑定多个IP并不少见，但是对于客户端来说往往只需要连接其中的一个即可，所以这个方法并不常用。
6. getLocalHost()方法返回运行机器的InetAddress。如果本机没有固定IP地址或者域名，可能会得到localhost作为域名，127.0.0.1作为IP地址的InetAddress对象。

#### 2. 常用的方法

前面说过InetAddress类是Java对IP地址(包括IPv4和IPv6)的封装。一般来说，它同时包含主机名(hostname)和IP地址。所以通过InetAddress我们可以得到hostName或者ip地址：

1. public String getHostName()
2. public String getHostAddress()
3. public byte[] getAddress()

**说明**

1. 没有setter方法，原因很明显，不多说
2. getHostName()方法一般返回主机名，如果这台机器没有主机名或者安全管理器阻止确定主机名，就返回点分四段格式的数字IP地址。
3. 如果InetAddress是通过getByName(数字IP地址字符串)构造的，那么getHostName会在这里进行DNS查找。
4. getHostAddress()返回包括点分四段格式IP地址的字符串。
5. getAddress()返回网络字节顺序的IP地址。返回的字节是无符号的，因为Java都是有符号的。值大于127的字节会被当作负数。因此，如果要对getAddress()返回的字节数值进行操作，需要把字节提升为int，进行适当的调整。比如：`int unsignedByte = signedByte < 0 ? signedByte + 256 : signedByte;`

#### 3. Inet4Address 和 Inet6Address

Java 1.4引入了两个新类，Inet4Address和Inet6Address，以此区分IPv4和IPv6地址：

	public final class Inet4Address extends InetAddress
	public final class Inet6Address extends InetAddress

不过基本上你不需要考虑一个地址是IPv4还是IPv6地址。这两个类属于JDK本身实现细节，你只需要关注基类InetAddress即可，面向对象编程的好处在这里体现出来。	


### NetworkInterface类

Java 1.4 添加了一个[java.net.NetworkInterface](http://docs.oracle.com/javase/tutorial/networking/nifs/definition.html)类，表示计算机与网络的互联点。一个NetworkInterface一般是指网卡地址（Network Interface Card(NIC)），但是不一定是硬件的形式。软件模拟的网络接口也是用NetworkInterface表示。例如loopback interface(127.0.0.1 for IPv4或者::1 for IPv6)。总而言之，NetworkInterface用来表示物理和虚拟的网卡地址。下面这段代码：

	Socket soc = new java.net.Socket();
	soc.connect(new InetSocketAddress(address, port));

操作系统会为我们选择一个网络接口发送和接收数据。但是我们也可以告诉操作系统使用哪个网卡：

	NetworkInterface nif = NetworkInterface.getByName("eth0");
	Enumeration<InetAddress> nifAddresses = nif.getInetAddresses();

	Socket soc = new java.net.Socket();
	soc.bind(new InetSocketAddress(nifAddresses.nextElement(), 0));
	soc.connect(new InetSocketAddress(address, port));


#### 工厂方法

1. public static NetworkInterface getByName(String name) throws SocketException
2. public static NetworkInterface getByInetAddress(InetAddress address) throws SocketException
3. public static Enumeration getNetworkInterfaces() throws SocketException

**说明**

1. 网络接口名称与平台相关，unix系统一般命名为eth0、eth1等等，本地回路地址名可能是类似于lo的。
2. 记住下面的映射关系：一台机器可能有多个网络接口(NetworkInterface)，一个网络接口(NetworkInterface)可能绑定了多个IP地址(InetAddress)。


### TCP

Socket是两台主机之间的一个连接。它可以进行七项基本操作：

1. 连接远程机器
2. 发送数据
3. 接收数据
4. 关闭连接
5. 绑定端口
6. 监听入站数据
7. 在所绑定的端口上接收来自远程机器的连接

说明

1. Java的Socket类可同时用于客户端和服务器，它有对应于前四项操作的方法。后三项只有服务器才需要，这些操作通过ServerSocket类实现。
2. TCP是面向字节流的协议，所以数据的发送和接收通过socket关联的输入输出流进行，操作起来跟文件是类似的。

#### Socket类

java.net.Socket类是Java执行客户端TCP操作的基础类。其他进行TCP网络连接的面向客户端的类，如URL、URLConnection等，最终都会调用到Socket类的方法。

Socket在数据结构上，是 <IP, port> 的组合。其中IP可以通过InetAddress进行主机名和IP地址的转换和表示，port是端口号，必须在0到65535之间。

**构造函数**

1. public Socket(String host, int port) throws UnknownHostException, IOException
2. public Socket(InetAddress host, int port) throws IOException
3. public Socket(String host, int port, InetAddress interface, int localPort) throws IOException, UnknownHostException
4. public Socket(InetAddress host, int port, InetAddress interface, int localPort) throws IOException, UnknownHostException
5. public Socket()
6. public Socket(SocketImpl impl)
7. public Socket(Proxy proxy)

**说明**

1. 上面前四个构造函数，在创建Socket的时候就会尝试连接指定的服务器。后三个构造函数用于创建未连接的socket对象。
2. 第三和第四个构造函数，连接到前两个参数指定的主机和端口，从后两个参数指定的本机网络接口和端口进行连接。如果0传递给localPort参数，Java会随机选择一个1024~65535之间的可用端口。

**常用方法**

1. public InetAddress getInetAddress()
2. public int getPort()
3. public int getLocalPort()
4. public InetAddress getLocalAddress()
5. public InputStream getInputStream() throws IOException
6. public OutputStream getOutputStream() throws IOException
7. public void close() throws IOException
8. public void shutdownInput() throws IOException
9. public void shutdownOutput() throws IOException

**说明**

1. TCP是面向字节流的协议，Socket其实就是一个文件句柄，所以通过它关联的输入输出流实现数据的读写。
2. shutdownInput和shutdownOutput仅仅关闭连接的一半。需要注意的是shutdown方法只影响socket的流，并不释放与socket相关的资源，如占用的端口等。所以仍然需要在结束使用后close掉该socket。


**设置Socket选项**

* TCP_NODELAY
* SO_BINDADDR
* SO_TIMEOUT
* SO_LINGER
* SO_REUSEADDR
* SO_SNDBUF
* SO_RCVBUF
* SO_KEEPALIVE
* OOBINLINE

**说明**

1. 正常情况下,小的包在发送前会组合为大点的包。在发送另一个包之前，本地主机要等待远程系统对前一个包的回应，这称之为Nagle算法。Nagle算法主要是为了解决“糊涂窗口综合症”的。但是Nagle算法也会带来一些问题。如果远程系统没有尽可能快地将回应发送会本地系统，那么依赖于小数据量信息稳定传播的应用程序会变得很慢。设置TCP_NODELAY为true可以打破这种缓冲模式，这样所有的包一就绪就能发送。
2. SO_LINGER选项规定，当socket关闭时如何处理尚未发送的数据包。默认情况下，close()方法将立即返回，但系统仍会尝试发送剩余的数据。如果延迟时间设置为0，那么当socket关闭时，所有为发送的数据将都被丢弃。如果延迟hi见设置为任意正数，那么close()方法将会阻塞指定秒数，等待数据发送和接收回应，然后socket就会被关闭，所有剩余的数据都不会发送，也不会收到回应。这个参数和SO_REUSEADDR选项可以在一定程度上解决time_wait状态占用端口问题，但是同样会带来一些问题。具体可以参考笔者前面写的一篇文章[如何解决time_wait状态占用端口问题](http://blog.arganzheng.me/posts/how-to-avoid-time-wait-state.html)。
3. 当socket关闭时，为了确保收到所有寻址到此端口的延迟数据，会等待一段时间，也就是进入所谓的time_wait状态。系统不会对后接收的任何包进行操作，只是希望确保这些数据不会偶然地传入绑定于相同端口的新进程。如果开启SO_REUSEADDR（默认情况是关闭），就允许另一个socket绑定到一个尚未释放的端口，尽管此时仍有可能存在前一个socket未接收的数据。
4. 如果启用SO_KEEPALIVE，客户端会偶尔通过一个空闲连接发送一个数据包（一般两小时一次），以确保服务器为崩溃。如果服务器没有响应此包，客户端会尝试11分钟多的时间，知道接收到响应为止。如果在12分钟内未收到响应，客户端就关闭socket。没有SO_KEEPALIVE，不活动的客户端可能会永久存在下去，而不会注意到服务器已经崩溃。SO_KEEPALIVE默认值是false。


**SocketAddress**

SocketAddress类的主要用途是为暂时的socket连接信息(IP地址和端口)提供方便的存储，这些信息可以重用以创建新的socket，即使最初的socket已断开并被垃圾回收。为此，Socket类提供了两个返回SocketAddress的方法:

1. getRemoteSocketAddress
2. getLocalSocketAddress

对于未连接的socket，通过connect()方法进行连接时就必须使用SocketAddress:

	public void connect(SocketAddress endpoint) throws IOException
	public void connect(SocketAddress endpoint, int timeout) throws IOException


#### ServerSocket

对于接收连接的服务器，Java提供了服务器Socket的ServerSocket类。

构造函数

1. public ServerSocket(int port) throws BindException, IOException
2. public ServerSocket(int port, int queueLength) throws BindException, IOException
3. public ServerSocket(int port, int queueLenght, InetAddress bindAddress) throws IOException
4. public ServerSocket() throws IOException

无参构造函数需要在以后使用bind()进行绑定：

1. public void bind(SocketAddress endpoint) throws IOException
2. public void bind(SocketAddress endpoint, int queueLength) throws IOException

主要用途是，允许程序在绑定端口前设置服务器Socket的选项。

	ServerSocket ss = new ServerSocket();
	// 设置socket选项
	SocketAddress http = new InetSocketAddress(80);
	ss.bind(http);

接受和关闭连接

1. public Socket accept() throws IOException
2. public void close() throws IOException

### UDP

#### DatagramSocket

#### DatagramPacket


### 非阻塞IO


### URLConnection




