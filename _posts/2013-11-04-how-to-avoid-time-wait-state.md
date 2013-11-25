---
layout: post
title: 如何解决time_wait状态占用端口问题
---

问题 
----

在restart服务端的时候，总是发现启动失败，端口被占用。需要先stop，然后等待一段时间，再start。分析了一下，确定是time_wait导致端口缓慢释放问题。

相关背景知识
------------

一、Linux服务器上11种网络连接状态:

![TCP状态机](http://farm8.staticflickr.com/7302/10221890304_4b25a4fdb2.jpg)

通常情况下: 一个正常的TCP连接，都会有三个阶段:
 1. TCP三次握手
 2. 数据传送
 3. TCP四次挥手 

同时会有三种特殊的数据包

 1. SYN:(同步序列编号,Synchronize Sequence Numbers)该标志仅在三次握手建立TCP连接时有效。表示一个新的TCP连接请求。
 2. ACK:(确认编号,Acknowledgement Number)是对TCP请求的确认标志,同时提示对端系统已经成功接收所有数据。 
 3. FIN:(结束标志,FINish)用来结束一个TCP回话.但对应端口仍处于开放状态,准备接收后续数据。

产生11种状态

 1. LISTEN:首先服务端需要打开一个socket进行监听，状态为LISTEN. /* The socket is listening for incoming connections. 侦听来自远方TCP端口的连接请求 */
 2. SYN_SENT:客户端通过应用程序调用connect进行active open.于是客户端tcp发送一个SYN以请求建立一个连接.之后状态置为SYN_SENT. /*The socket is actively attempting to establish a connection. 在发送连接请求后等待匹配的连接请求 */
 3. SYN_RECV:服务端应发出ACK确认客户端的SYN,同时自己向客户端发送一个SYN. 之后状态置为SYN_RECV  /* A connection request has been received from the network. 在收到和发送一个连接请求后等待对连接请求的确认 */
 4. ESTABLISHED: 代表一个打开的连接，双方可以进行或已经在数据交互了。/* The socket has an established connection. 代表一个打开的连接，数据可以传送给用户 */
 5. FIN_WAIT1:主动关闭(active close)端应用程序调用close，于是其TCP发出FIN请求主动关闭连接，之后进入FIN_WAIT1状态./* The socket is closed, and the connection is shutting down. 等待远程TCP的连接中断请求，或先前的连接中断请求的确认 */
 6. CLOSE_WAIT:被动关闭(passive close)端TCP接到FIN后，就发出ACK以回应FIN请求(它的接收也作为文件结束符传递给上层应用程序),并进入CLOSE_WAIT. /* The remote end has shut down, waiting for the socket to close. 等待从本地用户发来的连接中断请求 */
 7. FIN_WAIT2:主动关闭端接到ACK后，就进入了FIN-WAIT-2 ./* Connection is closed, and the socket is waiting for a shutdown from the remote end. 从远程TCP等待连接中断请求 */
 8. LAST_ACK:被动关闭端一段时间后，接收到文件结束符的应用程序将调用CLOSE关闭连接。这导致它的TCP也发送一个 FIN,等待对方的ACK.就进入了LAST-ACK . /* The remote end has shut down, and the socket is closed. Waiting for acknowledgement. 等待原来发向远程TCP的连接中断请求的确认 */
 9. TIME_WAIT:在主动关闭端接收到FIN后，TCP就发送ACK包，并进入TIME-WAIT状态。/* The socket is waiting after close to handle packets still in the network.等待足够的时间以确保远程TCP接收到连接中断请求的确认 */
 10. CLOSING: 比较少见./* Both sockets are shut down but we still don't have all our data sent. 等待远程TCP对连接中断的确认 */
 11. CLOSED: 被动关闭端在接受到ACK包后，就进入了closed的状态。连接结束./* The socket is not being used. 没有任何连接状态 */

**TIME_WAIT状态**

可以看出，TIME_WAIT状态的形成只发生在主动关闭连接的一方。

主动关闭方在接收到被动关闭方的FIN请求后，发送成功给对方一个ACK后,将自己的状态由FIN_WAIT2修改为TIME_WAIT，而必须再等2倍的MSL（`Maximum Segment Lifetime`，MSL是一个数据报在internetwork中能存在的时间）时间之后双方才能把状态 都改为CLOSED以关闭连接。目前RHEL里保持TIME_WAIT状态的时间为60秒。

*TIPS* 上述很多TCP状态在系统里都有对应的解释或设置,可见man tcp。

 
二、关于长连接和短连接

通俗点讲：短连接就是一次TCP请求得到结果后，连接马上结束。而长连接并不马上断开，而一直保持着，直到长连接TIMEOUT（具体程序都有相关参数说明）。长连接可以避免不断的进行TCP三次握手和四次挥手。

长连接(keepalive)是需要靠双方不断的发送探测包来维持的，keepalive期间服务端和客户端的TCP连接状态是ESTABLISHED。目前http 1.1版本里默认都是keepalive（1.0版本默认是不keepalive的）。

一个应用至于到底是该使用短连接还是长连接，应该视具体情况而定。一般的应用应该使用长连接。

1、Linux的相关keepalive参数

* tcp_keepalive_time - INTEGER
:   How often TCP sends out keepalive messages when keepalive is enabled.
    Default: 2hours.
* tcp_keepalive_probes - INTEGER
:   How many keepalive probes TCP sends out, until it decides that the
    connection is broken. Default value: 9.
* tcp_keepalive_intvl - INTEGER
:   How frequently the probes are send out. Multiplied by
    tcp_keepalive_probes it is time to kill not responding connection,
    after probes started. Default value: 75sec i.e. connection
    will be aborted after ~11 minutes of retries.

2、F5负载均衡上的相关参数说明

* Keep Alive Interval
:   Specifies, when enabled, how frequently the system sends data over an idle TCP connection, to determine whether the connection is still valid.
    Specify: Specifies the interval at which the system sends data over an idle connection, to determine whether the connection is still valid. The default is 1800 milliseconds.
* Time Wait
:   Specifies the length of time that a TCP connection remains in the TIME-WAIT state before entering the CLOSED state.
    Specify: Specifies the number of milliseconds that a TCP connection can remain in the TIME-WAIT state. The default is 2000.
* Idle Timeout
:   Specifies the length of time that a connection is idle (has no traffic) before the connection is eligible for deletion.
    Specify: Specifies a number of seconds that the TCP connection can remain idle before the system deletes it. The default is 300 seconds.

3、Apache的相关参数说明

以下是Apache/2.0.61版本的默认参数和说明

* KeepAlive:
:   default On.Whether or not to allow persistent connections (more than one request per connection). Set to "Off" to deactivate.
* MaxKeepAliveRequests:
:   default 100.The maximum number of requests to allow
    during a persistent connection. Set to 0 to allow an unlimited amount.
    We recommend you leave this number high, for maximum performance.
* KeepAliveTimeout:
:   default 15. Number of seconds to wait for the next request from the same client on the same connection.

4、JAVA1.6的相关参数说明

* http.keepAlive - boolean
:   default: true 
    Indicates if keep alive (persistent) connections should be supported.
* http.maxConnections - int
:   default: 5
    Indicates the maximum number of connections per destination to be kept alive at any given time

解决方案
--------

设置`SO_LINGER`选项，可以让TCP链接直接走捷径关闭，避免了四次挥手过程，从而避免了time_wait缓慢状态。不过会导致缓冲区的数据丢失，所以不是很推荐。

java中一般这样设置：

    private Socket initSocket(InetSocketAddress addr) throws IOException, SocketException {
        Socket socket = new Socket();
        
        socket.setReuseAddress(true);
        socket.setSoLinger(true, 0);

        socket.connect(addr, CONNECTION_TIME_OUT_MS);
        socket.setSoTimeout(TIMEOUT_MS);    
        
        return socket;
    }

试验结果，貌似java必须加上`socket.setReuseAddress(true);`。而且只需要加上这个。

说明: 关于`SO_LINGER`和`SO_REUSEADDR`选项参考《UNIX网络编程卷一：套接字联网API》。就明白了。

参考文章
--------

1. [Java network server and TIME_WAIT](http://stackoverflow.com/questions/922951/java-network-server-and-time-wait)
2. [What is the reason for time_wait connection increasing in java?](http://stackoverflow.com/questions/10726049/what-is-the-reason-for-time-wait-connection-increasing-in-java)
3. [TIME_WAIT and its design implications for protocols and scalable client server systems](http://www.serverframework.com/asynchronousevents/2011/01/time-wait-and-its-design-implications-for-protocols-and-scalable-servers.html)
4. [如何解决大量JAVA客户端Socket关闭时TIME_WAIT的问题？](http://www.zhihu.com/question/20129467)