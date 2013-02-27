负载均衡
=======

高性能高可用性的系统一般会有多层次的负载均衡。(High-performance systems may use multiple layers of load balancing.)。负责均衡的实现机制有硬件和软件两种。下面我们以top down的方式一层层介绍下去。

## 前端web服务器负载均衡

一般的做法是：

首先是使用DNS做第一层的负载均衡(DNS based load balancing cluster)，同一个域名可能映射到不同的服务器（IP地址）。一般采用简单[Round-robin](http://en.wikipedia.org/wiki/Round-robin_DNS)DNS策略
 
其次域名一般不会直接映射到具体的某个web服务器，而是先到一个负载均衡服务器。负载均衡服务器根据实现方式不同一般有两种：

1. 硬件负载均衡：如F5。
    
2. 软件负载均衡：如[HAProxy](http://haproxy.1wt.eu/)，[LVS](http://www.linux-vs.org/)，等。
    
    软件负载均衡服务器也有不同的实现机制：
    1. 有用户空间的负载均衡程序
        
        如HAProxy, nginx，等。这种一般是建立在TCP/IP协议之上的负载均衡，所以也称为application-level load balancing，或者称为Layer-7 switching。
        缺点就就是性能比较差，但是功能会比较多，除了负载均衡，还能作Web服务器，可以Cache，可以压缩等等。
    
    2. 还有编译进入内核的负载均衡程序
     
        如LVS，这种一般涉及到修改IP层package，所以也称为IP-based load balancers，或者Layer-4 switching）。用户空间的负载均衡程序一般有两种实现：1. event-driven, single-process model；或者Multi-process or multi-threaded models。
    
### 负载均衡的三角策略

负载均衡涉及到三个角色：client，balancer和backend-server。
请求路径非常明确：client==>balancer==>backend-server
但是根据负载均衡实现机制的不同，返回路径有两种：
1. backend-server==>balancer==>client
2. backend-server==>client

对于第一种方式，一般是使用一种称之为NAT(http://www.linux-vs.org/VS-NAT.html)的路由策略。所有应用层的load balancer(如HAProxy，Nginx，等)也只能是这种方式。

对于第二种，由于返回的时候backend-server直接绕过load balancer，说明以下几点：

1. backend-server有真实的外部IP地址，也就是说backend-server也是在一个LAN/WAN网络里。而不是一个私有网络。
2. backend-server必须有client的IP信息，否则它没法发送给client，也就是说balancer发给它的package中除了包含它自己的IP信息（IP协议要求）还必须包含client的IP信息。

解决这个问题的方案就是IP tunneling，也称为[IP encapsulation](http://www.networksorcery.com/enp/protocol/ip-ip.htm)，其实是一个非常简单协议，就是为了解决发送端IP地址与返回地址不是同一个的问题，要保留两种，那么简单再增加一个IP头部就是了。不过要求balancer和backend-servers都支持IP分装协议(IP encapsulation protocol)。
具体的实现机制和原理参加LVS的这篇文档：[Virtual Server via IP Tunneling](http://www.linux-vs.org/VS-IPTunneling.htmls)。

另一种做法就是直接路由方式(Direct Routing)：就是load balancer和backend-server处于同样的地位：双IP，其中一个IP都是配置同样的VIP，处于同一个局域网内。当请求过来的时候，balancer直接将package转发给挑选到的backend-server，因为两者都有同样的VIP，所以就不需要修改IP头或者做IP tunnele了。具体做法参见：[Virtual Server via Direct Routing](http://www.linux-vs.org/VS-DRouting.html)。

返回消息不经过load balancer能够带来极大的性能提高，因为请求消息一般小而快，而返回消息一般比较大而慢。让backend-server直接返回给client，极大的解放了load balancer。

关于 LVS提供的三种routing方式，这篇文章介绍的非常详细：[Load Balancing The UK National JANET Web Cache Service——Using Linux Virtual Servers](http://www.linux-vs.org/docs/PilotService.html)

**疑问：load balancer本身不会成为瓶颈和单点吗？**

由于load balancer基本就是一个forward逻辑，所以很快，吞吐量非常高。如果采用backend-server直接返回client的模式，那么不需要改写response信息，吞吐量会进一步提高。再配合NDS负载均衡，一般不会有问题。对于单点问题，一般的作法是配对一个backup，与active server做heart beat。如果有状态的话，一般是共享存储，或者在backup server之间做数据同步。如果当期的活动server宕掉了，那么迅速切换到backup server，可能需要一定的人工接入。


在这些负载均衡服务器会配置需要负载均衡的实际web服务器，一般是nginx或者apache。负载均衡的策略一般是最简单也是最有效的随机或者轮休策略。而事实上，由于现在大部分网站都是动态内容，这些web服务器往往也是一个有cache功能的load balancer，为后台的应用服务器（tomcat、JBoss等）做负载均衡。

## 后端应用服务器负责均衡

后端应用服务器之间的通讯一般都是服务调用的方式，一般还会跨语言。所以现在业界普遍的做法就是采用 配置中心+客户端负载均衡的方式实现服务路由和负载均衡。试用Trift或者protocalbuf实现跨语言交互。

## 后端数据库负载均衡

1. [USING HAPROXY FOR MYSQL FAILOVER AND REDUNDANCY](http://www.alexwilliams.ca/blog/2009/08/10/using-haproxy-for-mysql-failover-and-redundancy/)
2. [A more stable MySQL with HAProxy](http://flavio.tordini.org/a-more-stable-mysql-with-haproxy/comment-page-1)

### Sharding

### LVS学习笔记

LVS(Linux Virtual Servers)是目前最好的IP-based load balancers。它的优点是编译进内核，速度非常快。缺点就是平台和内核版本相关，目前只支持linux。配合[Keepalived](http://www.keepalived.org/)来监控后台servers的情况，效果非常好。

[Linux Virtual Server (LVS) Project](http://www.austintek.com/LVS/)

[Load Balancing The UK National JANET Web Cache Service——Using Linux Virtual Servers](http://www.linux-vs.org/docs/PilotService.html)

## 实际企业级应用实例

Alibaba
Tencent
Quara


## 负载均衡策略


#### 服务端负载均衡：如nginx等方向代理负载均衡。

#### 客户端负载均衡：
如Cobar，DNS，等

## 负载均衡的范围

1. 全局负载均衡
 
    GSLB 是英文Gobal Server Load Balance的缩写，意思是全局负载均衡。 作用：实现在广域网（包括互联网）上不同地域的服务器间的流量调配，保证使用最佳的服务器服务离自己最近的客户，从而确保访问质量。

2. 局部负载均衡（一般是同个IDC，同个业务服务）

