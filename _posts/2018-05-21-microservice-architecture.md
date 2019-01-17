---
title: 微服务架构学习
layout: post
catalog: true
tags: [微服务, 架构]
category: [技术]
---

微服务架构
--------

微服务的概念在2014年3月由Martin Fowler首次提出。

微服务架构解决的核心问题及其相应的开源组件如下所示：

* RPC框架 (Service-to-service calls)
    * Spring Boot/Spring MVC
    * Dubbo
    * gRPC
    * thrift
* 服务注册和发现 (Service registration and discovery)
    * 注册中心
        * Eureka: 在 Netflix 经过大规模生产验证，支持跨数据中心。
        * Consul: 天然支持跨数据中心，还支持 KV 模型存储和灵活健康检查能力
        * ZooKeeper
        * Redis
        * Nacos
* 负载均衡 (Load balancing)
    * Ribbon/Feign
* 容错限流 (Circuit Breakers)
    * Hystrix
    * Nginx/Kong + RateLimit
    * Sentinel
* 认证授权 (Security)
    * Spring Security/Spring Cloud Security
    * OAuth 
    * OpenID connect
* 服务网关和路由 (Routing)
    * Zuul: 在 Netflix 经过大规模生产验证，支持灵活的动态过滤器脚本机制，异步性能不足（基于 Netty 的异步 Zuul 迟迟未能推出正式版）
    * Kong: Nginx/OpenResty 的 API 网关。
* 配置中心 (Distributed/versioned configuration)
    * Spring Cloud Config
    * Apollo@携程
* 监控告警
    * 日志监控 (Logging)
        * ELK 
    * 调用链监控 (Tracing)
        * CAT@点评
        * Zipkin@Twitter
        * Pinpoint@Naver
    * Metrics 监控 (stats, metrics)
        * 存储 (TSDB)
            * OpenTSDB(HBase) + Argus，KariosDB(Cassandra) + ZMon
            * InfluxDB 
            * Prometheus
        * 告警 
            * Argus 是 Salesforce 开源的基于 OpenTSDB 的统一监控告警平台，支持丰富的告警函数和灵活的告警配置。
            * ZMon 后台采用 KairosDB 存储，如果企业已经采用 KariosDB 作为时间序列数据库
        * 报表
            * Grafana 是 Metrics 报表展示的社区标配
    * 健康检查
    * 告警通知
        * Elastalert 是 Yelp 开源的针对 ELK 的告警通知模块
* 后台任务
* 事件驱动
    * Spring Cloud Stream
* 其它
    * Global locks
    * Leadership election and cluster state
* 部署微服务 (CICD & CNCF)
    * CICD
    * A/B rollout
    * dark launches
    * auto scale
    * [docker](https://www.docker.com/)
    * [Cloud Foundry](https://www.cloudfoundry.org/)
    * [Kubernetes](https://kubernetes.io/)
    * [Mesos](http://mesos.apache.org/)

业界主流微服务解决方案
------------------

### 1、[Netflix](https://netflix.github.io/)

* [Eureka](https://github.com/Netflix/eureka) for service discovery
* [Archaius](https://github.com/Netflix/archaius) for distributed configuration
* [Ribbon](https://github.com/Netflix/ribbon) for resilient and intelligent inter-process and service communication
* [Hystrix](https://github.com/Netflix/hystrix) for latency and fault tolerance at run-time
* [Prana](https://github.com/Netflix/prana) as a sidecar for non-JVM based services.
* [Zuul](https://github.com/Netflix/zuul) (which integrates Hystrix, Eureka, and Ribbon as part of its IPC capabilities) provides dyamically scriptable proxying at the edge of the cloud deployment.
* [Fenzo](https://github.com/Netflix/Fenzo) as a scheduler Java library for Apache Mesos frameworks that supports plugins for scheduling optimizations and facilitates cluster autoscaling.

### 2、[Spring Cloud](http://spring.io/projects/spring-cloud): Tools for building common patterns in distributed systems with Spring.

Spring Cloud为开发者提供了快速构建分布式系统的通用模型的工具（包括配置管理、服务发现、熔断器、智能路由、微代理、控制总线、一次性令牌、全局锁、领导选举、分布式会话、集群状态等）。 主要项目包括：

* Spring Cloud Config：由Git存储库支持的集中式外部配置管理。配置资源直接映射到Spring Environment，但是如果需要可以被非Spring应用程序使用。
* Spring Cloud Netflix：与各种Netflix OSS组件（Eureka，Hystrix，Zuul，Archaius等）集成。
* Spring Cloud Bus：用于将服务和服务实例与分布式消息传递联系起来的事件总线。用于在集群中传播状态更改（例如配置更改事件）。
* Spring Cloud for Cloudfoundry：将您的应用程序与Pivotal Cloudfoundry集成。提供服务发现实现，还可以轻松实现通过 SSO 和 OAuth 2 保护资源，还可以创建Cloudfoundry服务代理。
* Spring Cloud - Cloud Foundry Service Broker：提供构建管理一个Cloud Foundry中服务的服务代理的起点。
* Spring Cloud Cluster：领导选举和通用状态模型（基于ZooKeeper，Redis，Hazelcast，Consul的抽象和实现）。
* Spring Cloud Consul：结合Hashicorp Consul的服务发现和配置管理
* Spring Cloud Security：在Zuul代理中为负载平衡的OAuth 2休眠客户端和认证头中继提供支持。
* Spring Cloud Sleuth：适用于Spring Cloud应用程序的分布式跟踪，与Zipkin，HTrace和基于日志（例如ELK）跟踪兼容。
* Spring Cloud Data Flow：针对现代运行时的可组合微服务应用程序的云本地编排服务。易于使用的DSL，拖放式GUI和REST-API一起简化了基于微服务的数据管道的整体编排。
* Spring Cloud Stream：轻量级事件驱动的微服务框架，可快速构建可连接到外部系统的应用程序。使用Apache Kafka或RabbitMQ在Spring Boot应用程序之间发送和接收消息的简单声明式模型。
* Spring Cloud Stream Application Starters：Spring Cloud任务应用程序启动器是Spring Boot应用程序，可能是任何进程，包括不会永远运行的Spring Batch作业，并且它们在有限时间的数据处理之后结束/停止。
* Spring Cloud ZooKeeper：ZooKeeper的服务发现和配置管理。
* Spring Cloud for Amazon Web Services：轻松集成托管的Amazon的Web Services服务。它通过使用Spring的idioms和APIs便捷集成AWS服务，例如缓存或消息API。开发人员可以围绕托管服务，不必关心基础架构来构建应用。
* Spring Cloud Connectors：使PaaS应用程序在各种平台上轻松连接到后端服务，如数据库和消息代理（以前称为“Spring Cloud”的项目）。
* Spring Cloud Starters：作为基于Spring Boot的启动项目，降低依赖管理（在Angel.SR2后，不在作为独立项目）。
* Spring Cloud CLI：插件支持基于Groovy预言快速创建Spring Cloud的组件应用。

### 3、[spring cloud alibaba](https://github.com/spring-cloud-incubator/spring-cloud-alibaba): Spring Cloud Alibaba provides a one-stop solution for application development for the distributed solutions of Alibaba middleware.

Spring Cloud for Alibaba，它是由一些阿里巴巴的开源组件和云产品组成的。这个项目的目的是为了让大家所熟知的 Spring 框架，其优秀的设计模式和抽象理念，以给使用阿里巴巴产品的 Java 开发者带来使用 Spring Boot 和 Spring Cloud 的更多便利。
 
其中阿里巴巴开源组件的命名前缀为spring-cloud-alibaba，提供了如下特性：

* 服务注册与发现 : 适配 Spring Cloud 服务注册与发现标准，默认集成了 Ribbon 的支持。
* 分布式配置管理：支持分布式系统中的外部化配置，配置更改时自动刷新。
* 消息驱动能力：基于 Spring Cloud Stream 为微服务应用构建消息驱动能力。
* 服务限流降级 : 默认支持 Servlet、Feign、RestTemplate、Dubbo 和 RocketMQ 限流降级功能的接入，可以在运行时通过控制台实时修改限流降级规则，还支持查看限流降级 Metrics 监控。

阿里云的产品组件的命名前缀为 spring-cloud-alicloud ，提供了如下特性：

* 应用配置管理 : 阿里云配置管理服务ACM，加强了安全的配置管理，并且还包含了完整的推送轨迹查询。
* 对象存储服务 : 阿里云提供的海量、安全、低成本、高可靠的云存储服务。支持在任何应用、任何时间、任何地点存储和访问任意类型的数据。
* 分布式任务调度 : 提供秒级、精准、高可靠、高可用的定时（基于 Cron 表达式）任务调度服务。同时提供分布式的任务执行模型，如网格任务。网格任务支持海量子任务均匀分配到所有 Worker（schedulerx-client）上执行。

下面是使用到的一些组件:

* [Sentinel](https://github.com/alibaba/Sentinel) : 把流量作为切入点，从流量控制、熔断降级、系统负载保护等多个维度保护服务的稳定性。
* [Nacos](https://github.com/alibaba/Nacos) : 一个更易于构建云原生应用的动态服务发现、配置管理和服务管理平台。
* [RocketMQ](https://rocketmq.apache.org/) : 一款开源的分布式消息系统，基于高可用分布式集群技术，提供低延时的、高可靠的消息发布与订阅服务。
* [Alibaba Cloud ACM](https://www.aliyun.com/product/acm) : 一款在分布式架构环境中对应用配置进行集中管理和推送的应用配置中心产品。
* [Alibaba Cloud OSS](https://www.aliyun.com/product/oss) : 阿里云对象存储服务（Object Storage Service，简称 OSS），是阿里云提供的海量、安全、低成本、高可靠的云存储服务。您可以在任何应用、任何时间、任何地点存储和访问任意类型的数据。
* [Alibaba Cloud SchedulerX](https://help.aliyun.com/document_detail/43136.html?spm=a2c4g.11186623.6.709.baef7da9QVICiD) : 阿里中间件团队开发的一款分布式任务调度产品，提供秒级、精准、高可靠、高可用的定时（基于 Cron 表达式）任务调度服务。

但是显然，这个并不是一个完全的开源项目，因为阿里云的服务是要收费的。

### 4、Service Mesh

微服务的概念在2014年3月由Martin Fowler首次提出，而Service Mesh的概念则是在2016年左右提出，Service Mesh至今也经历了第二代的发展。

Service Mesh又译作“服务网格”，作为服务间通信的基础设施层。如果用一句话来解释什么是Service Mesh，可以将它比作是应用程序或者说微服务间的TCP/IP，负责服务之间的网络调用、限流、熔断和监控。对于编写应用程序来说一般无须关心TCP/IP这一层（比如通过 HTTP 协议的 RESTful 应用），同样使用Service Mesh也就无须关系服务之间的那些原来是通过应用程序或者其他框架实现的事情，比如Spring Cloud、OSS，现在只要交给Service Mesh就可以了。

Service Mesh有如下几个特点：

* 应用程序间通讯的中间层
* 轻量级网络代理
* 应用程序无感知
* 解耦应用程序的重试/超时、监控、追踪和服务发现

Service Mesh的架构如下图所示：


Service Mesh作为Sidebar运行，对应用程序来说是透明，所有应用程序间的流量都会通过它，所以对应用程序流量的控制都可以在Service Mesh中实现。

目前流行的开源 Service Mesh 框架有：

* [Linkerd](https://linkerd.io/)
* [Envoy](https://www.envoyproxy.io/)
* [Istio](https://istio.io/)
* [Conduit](https://conduit.io/)


#### 1、[Linkerd 1.x](https://linkerd.io/)

Linkerd 1.x 基于Twitter的Fingle，使用Scala编写，是业界第一个开源的Service Mesh方案，在长期的实际生产环境中获得验证。

它的主要特性有：

* 负载均衡：Linkerd提供了多种负载均衡算法，它们使用实时性能指标来分配负载并减少整个应用程序的尾部延迟。
* 熔断：Linkerd包含自动熔断，将停止将流量发送到被认为不健康的实例，从而使他们有机会恢复并避免连锁反应故障。
* 服务发现：Linkerd 与各种服务发现后端集成，通过删除特定的(ad-hoc)服务发现实现来帮助您降低代码的复杂性。
* 动态请求路由：Linkerd 启用动态请求路由和重新路由，允许您使用最少量的配置来设置分段服务（staging service），金丝雀（canaries），蓝绿部署（blue-green deploy），跨DC故障切换和黑暗流量（dark traffic）。
* 重试次数和截止日期：Linkerd可以在某些故障时自动重试请求，并且可以在指定的时间段之后让请求超时。
* TLS：Linkerd可以配置为使用TLS发送和接收请求，您可以使用它来加密跨主机边界的通信，而不用修改现有的应用程序代码。
* HTTP代理集成：Linkerd可以作为HTTP代理，几乎所有现代HTTP客户端都广泛支持，使其易于集成到现有应用程序中。
* 透明代理：您可以在主机上使用iptables规则，设置通过Linkerd的透明代理。
* gRPC：Linkerd支持HTTP/2和TLS，允许它路由gRPC请求，支持高级RPC机制，如双向流，流程控制和结构化数据负载。
* 分布式跟踪：Linkerd支持分布式跟踪和度量仪器，可以提供跨越所有服务的统一的可观察性。
* 仪器仪表：Linkerd支持分布式跟踪和度量仪器，可以提供跨越所有服务的统一的可观察性。

说明：Conduit 后面合并到 Linkerd 2.0，因此本质上 Linkerd 2.0 = Conduit。

#### 2、[Envoy](https://www.envoyproxy.io/)

Envoy底层基于C++，性能上优于使用Scala的Linkerd。同时，Envoy社区成熟度较高，商用稳定版本面世时间也较长。

Envoy是一个面向服务架构的L7代理和通信总线而设计的，这个项目诞生是出于以下目标：

对于应用程序而言，网络应该是透明的，当发生网络和应用程序故障时，能够很容易定位出问题的根源。

Envoy可提供以下特性：

* 外置进程架构：可与任何语言开发的应用一起工作；可快速升级。
* 基于新C++11编码：能够提供高效的性能。
* L3/L4过滤器：核心是一个L3/L4网络代理，能够作为一个可编程过滤器实现不同TCP代理任务，插入到主服务当中。通过编写过滤器来支持各种任务，如原始TCP代理、HTTP代理、TLS客户端证书身份验证等。
* HTTP L7过滤器：支持一个额外的HTTP L7过滤层。HTTP过滤器作为一个插件，插入到HTTP链接管理子系统中，从而执行不同的任务，如缓冲，速率限制，路由/转发，嗅探Amazon的DynamoDB等等。
* 支持HTTP/2：在HTTP模式下，支持HTTP/1.1、HTTP/2，并且支持HTTP/1.1、HTTP/2双向代理。这意味着HTTP/1.1和HTTP/2，在客户机和目标服务器的任何组合都可以桥接。
* HTTP L7路由：在HTTP模式下运行时，支持根据content type、runtime values等，基于path的路由和重定向。可用于服务的前端／边缘代理。
* 支持gRPC：gRPC是一个来自谷歌的RPC框架，使用HTTP/2作为底层的多路传输。HTTP/2承载的gRPC请求和应答，都可以使用Envoy的路由和LB能力。
* 支持MongoDB L7：支持获取统计和连接记录等信息。
* 支持DynamoDB L7：支持获取统计和连接等信息。
* 服务发现：支持多种服务发现方法，包括异步DNS解析和通过REST请求服务发现服务。
* 健康检查：含有一个健康检查子系统，可以对上游服务集群进行主动的健康检查。也支持被动健康检查。
* 高级LB：包括自动重试、断路器，全局限速，阻隔请求，异常检测。未来还计划支持请求速率控制。
* 前端代理：可作为前端代理，包括TLS、HTTP/1.1、HTTP/2，以及HTTP L7路由。
* 极好的可观察性：对所有子系统，提供了可靠的统计能力。目前支持statsd以及兼容的统计库。还可以通过管理端口查看统计信息，还支持 第三方的分布式跟踪机制。
* 动态配置：提供分层的动态配置API，用户可以使用这些API构建复杂的集中管理部署。

#### 3、[Istio](https://istio.io/)

Istio是Google、IBM和Lyft合作的开源项目，是目前最主流的Service Mesh方案，也是事实上的第二代Service Mesh标准。在Istio中，直接把Envoy作为Sidecar。除了Sidecar，Istio中的控制面组件都是使用Go语言编写。

Istio在服务网络中主要提供了以下关键功能：

* 流量管理：控制服务之间的流量和API调用的流向，使得调用更可靠，并使网络在恶劣情况下更加健壮。
* 可观察性：了解服务之间的依赖关系，以及它们之间流量的本质和流向，从而提供快速识别问题的能力。
* 策略执行：将组织策略应用于服务之间的互动，确保访问策略得以执行，资源在消费者之间良好分配。策略的更改是通过配置网格而不是修改应用程序代码。
* 服务身份和安全：为网格中的服务提供可验证身份，并提供保护服务流量的能力，使其可以在不同可信度的网络上流转。
* 平台支持：Istio旨在在各种环境中运行，包括跨云、Kubernetes、Mesos等。最初专注于Kubernetes，但很快将支持其他环境。
* 集成和定制：策略执行组件可以扩展和定制，以便与现有的ACL、日志、监控、配额、审核等解决方案集成。

Istio服务网格逻辑上分为数据面板和控制面板：

* 数据面板由一组智能代理（Envoy）组成，代理部署为边车 (Sidecar)，调解和控制微服务之间所有的网络通信。
* 控制面板负责管理和配置代理来路由流量，以及在运行时执行策略。

下图为Istio的架构设计图，主要包括了Envoy、Pilot、Mixer和Istio-Auth等。 

![istio-arch](/img/in-post/istio-arch.jpg)

* Envoy: 扮演Sidecar的功能，协调服务网格中所有服务的出入站流量，并提供服务发现、负载均衡、限流熔断等能力，还可以收集与流量相关的性能指标。
* Pilot: 负责部署在Service Mesh中的Envoy实例的生命周期管理。本质上是负责流量管理和控制，将流量和基础设施扩展解耦，这是Istio的核心。可以把Pilot看做是管理Sidecar的Sidecar, 但是这个特殊的Sidacar并不承载任何业务流量。Pilot让运维人员通过Pilot指定它们希望流量遵循什么规则，而不是哪些特定的pod/VM应该接收流量。有了Pilot这个组件，我们可以非常容易的实现 A/B 测试和金丝雀Canary测试。
* Mixer: Mixer在应用程序代码和基础架构后端之间提供通用中介层。它的设计将策略决策移出应用层，用运维人员能够控制的配置取而代之。应用程序代码不再将应用程序代码与特定后端集成在一起，而是与Mixer进行相当简单的集成，然后Mixer负责与后端系统连接。Mixer可以认为是其他后端基础设施（如数据库、监控、日志、配额等）的Sidecar Proxy。
* Istio-Auth: 提供强大的服务间认证和终端用户认证，使用交互TLS，内置身份和证书管理。可以升级服务网格中的未加密流量，并为运维人员提供基于服务身份而不是网络控制来执行策略的能力。Istio的未来版本将增加细粒度的访问控制和审计，以使用各种访问控制机制（包括基于属性和角色的访问控制以及授权钩子）来控制和监视访问服务、API或资源的访问者。

#### 4、[Conduit](https://conduit.io/)

Conduit 是开源 Linkerd 的公司 Buoyant 基于 Kubernetes 设计的一个超轻型服务网格服务。它可透明地管理在Kubernetes上运行的服务的运行时通信，使得它们更安全可靠。Conduit提供了可见性、可靠性和安全性的功能，而无需更改代码。

Conduit各方面的设计理念与Istio非常类似，作者使用Rust语言重新编写了Sidecar, 叫做Conduit Data Plane, 控制面则由Go语言编写的Conduit Control Plane接管。从Conduit的架构看，作者号称Conduit吸取了很多Linkerd的教训，比Linkerd更快、更轻、更简单，控制面功能更强。与Istio比较，Conduit的架构一方面比较简单，另一方面对于要解决的问题足够聚焦。

Conduit service mesh也是由数据面板和控制面板组成。数据面板承载应用实际的网络流量。控制面板驱动数据面板，并对外提供北向接口。

![conduit-arch](/img/in-post/conduit-arch.png)

其中控制面板 (control plane) 主要由下面四个组件构成:

* Controller : The controller deployment consists of multiple containers (public-api, proxy-api, destination, tap) that provide the bulk of the control plane’s functionality.
* Web : The web deployment provides the Linkerd dashboard.
* Prometheus : All of the metrics exposed by Linkerd are scraped via Prometheus and stored here. This is an instance of Prometheus that has been configured to work specifically with the data that Linkerd generates. There are instructions if you would like to integrate this with an existing Prometheus installation.
* Grafana : Linkerd comes with many dashboards out of the box. The Grafana component is used to render and display these dashboards. You can reach these dashboards via links in the Linkerd dashboard itself.

而数据面板 (Data Plane) 则是简单的包括你的 application 和透明的 sidecar proxy (默认是用 Rust 语言编写的 linkerd-proxy)。

最后，Conduit 还提供了一个本地命令行工具 (CLI)，用于跟控制面板和数据面板交互。和一个 Dashboard，用于服务监控和治理。可以通过 `linkerd dashboard` 命令启动。

![conduit-stat](/img/in-post/conduit-stat.png)

说明：Conduit 后面合并到 Linkerd 2.0，因此本质上 Linkerd 2.0 = Conduit。


**对比**

Linkerd 1.x 和 Envoy 比较相似，都是一种面向服务通信的网络代理，均可实现诸如服务发现、请求路由、负载均衡等功能。它们的设计目标就是为了解决服务之间的通信问题，使得应用对服务通信无感知，这也是Service Mesh的核心理念。

Linkerd 1.x和 Envoy 像是分布式的 Sidebar，多个类似 Linkerd 1.x 和 Envoy 的 proxy 互相连接，就组成了 service mesh。

而 Istio 和 Conduit (Linkerd 2.x) 则是站在了一个更高的角度，它将Service Mesh分为了Data Plane和Control Plane。Data Plane负责微服务间的所有网络通信，而Control Plane负责管理Data Plane Proxy:

![service-mesh-arch](/img/in-post/service-mesh-arch.jpg)

而且Istio和Conduit引入了Kubernetes，这也弥合了应用调度框架与Service Mesh之间的空隙。


### 5、Serverless

Serverless被翻译为『无服务器架构』，这个概念在2012年时便已经存在，比微服务和Service Mesh的概念出现都要早，但是直到微服务概念大红大紫之后，Serverless才重新又被人们所关注。

Serverless 的意思并不是无服务器，而是去除有关对服务器运行状态的关心和担心，代码在哪里运行，需要多少容量，它们是否在工作，应用是否跑起来正常运行。

“Write your code, tell the system when to run it and you’re done."

所有服务器的管理和扩缩容对开发者是透明的，开发者只需要关心业务逻辑的开发，其他的一切交给云服务提供者，比如Amazon Web Services (AWS Lambda), Google Cloud (Google Cloud Functions) 或者 Microsoft Azure (Azure Functions) 。

> In this model, organizations only pay for the resources they use — actual consumption — rather than pre-purchased services based on guesswork. The server management and capacity planning decisions are completely hidden from the developer, thus the term “serverless.” The servers are fully abstracted away. Instead, a cloud provider, like Amazon Web Services (AWS Lambda), Google Cloud (Google Cloud Functions) or Microsoft Azure (Azure Functions) dynamically manages the assignment and distribution of resources.

但是其实这块强调的是云平台带来的机器资源自由调度带来的便利。跟 service mesh 2.0 其实没有本质上的区别，从这个意义上来说，serverless 是目标，service mesh 是解决方案。所以基本上，目前开源的 serverless 框架，基本都是基于 k8s/mesos/docker compose这样的容器编排和资源调度框架和 Service Mesh 框架实现。主要有如下这些：

1. [Knative](https://github.com/knative) 谷歌开源的一个基于Kubernetes和Istio实现的构建、部署和管理的现代serverless workloads。
2. [kubeless](https://kubeless.io/) The Kubernetes Native Serverless Framework - Build advanced applications with FaaS on top of Kubernetes
3. [Fission](https://fission.io/) Fission is a framework for serverless functions on Kubernetes.
4. [OpenWhisk](https://openwhisk.apache.org/) Open Source Serverless Cloud Platform

### 最终的微服务解决方案 = Dev + Ops = Service Mesh  + Kubernetes ？

这就是为什么有了 Linkerd 和 Envoy 之后，还会进一步进化出 Istio 和 Conduit。它们相对于老的 serive mesh 框架最大的特点就是基于 Kubernetes 设计，补足了Kubernetes在微服务间服务通讯上的短板。虽然Dubbo、Spring Cloud等都是成熟的微服务框架，但是它们或多或少都会和具体语言或应用场景绑定，并只解决了微服务Dev层面的问题。若想解决Ops问题，它们还需和诸如[Cloud Foundry](https://www.cloudfoundry.org/)、[Mesos](http://mesos.apache.org/)、或[Kubernetes](https://kubernetes.io/)这类资源调度框架做结合：

![sevicemesh-with-k8s](/img/in-post/sevicemesh-with-k8s.png)

Kubernetes本身就是一个和开发语言无关的、通用的容器管理平台，它可以支持运行云原生和传统的容器化应用。并且它覆盖了微服务的Dev和Ops阶段，结合Service Mesh，它可以为用户提供完整端到端的微服务体验。

因此我们有理由推测，未来的微服务架构和技术栈可能是如下形式:

![msa-stack](/img/in-post/msa-stack.jpg)

云平台(或者自建机房) 为微服务提供了资源能力（计算、存储和网络等），容器 作为最小工作单元被 Kubernetes 调度和编排，Service Mesh 管理微服务的服务通信，最后通过 API Gateway 向外暴露微服务的业务接口。


参考文章
------

1. [一篇文章带你快速理解微服务架构，由浅入深带你走进微服务架构的核心](https://blog.csdn.net/javaxuexi123/article/details/79500619)
2. [Understanding Microservices: From Idea To Starting Line](https://medium.freecodecamp.org/microservices-from-idea-to-starting-line-ae5317a6ff02)
3. [轻舟微服务](https://www.163yun.com/product-nsf) 网易云的轻舟微服务
4. [Serverless Service Mesh With Kubeless And Istio](https://engineering.bitnami.com/articles/serverless-service-mesh-with-kubeless-and-istio.html)
5. [《microservice & serverless》by蔡超的一点感想](https://segmentfault.com/a/1190000012944359)




