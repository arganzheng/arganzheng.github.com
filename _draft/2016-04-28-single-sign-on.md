---
title: 单点登陆实现
layout: post
catalog: true
---


CAS
---

[JASIG CAS server](https://www.apereo.org/projects/cas)，一个开源的CAS server实现。
包含客户端(CAS Client)和服务端(CAS Server)两个物理组件，两者之间通过各种认证协议(Authentication Protocols)进行交互。

![cas_architecture](/img/in-post/cas_architecture.png)

其中CAS Server包含三个层次的子系统：

* Web/API: 与外部系统(主要是client)的交互入口，将请求委托给下面的ticketing子系统。JASIC CAS实现是建立在Spring MVC/Spring Webflow框架上。
* [Ticketing](http://jasig.github.io/cas/4.2.x/installation/Configuring-Ticketing-Components.html): 主要负责生成票据，一般使用Cache-Based Ticket Registries。
* [Authentication](http://jasig.github.io/cas/4.2.x/installation/Configuring-Authentication-Components.html): 对票据进行验证。

最新版的CAS支持多种认证协议：

* [CAS](http://jasig.github.io/cas/4.2.x/protocol/CAS-Protocol.html)
* [SMAL](http://jasig.github.io/cas/4.2.x/protocol/SAML-Protocol.html)
* [OpenID](http://jasig.github.io/cas/4.2.x/protocol/OpenID-Protocol.html)
* [OAuth](http://jasig.github.io/cas/4.2.x/protocol/OAuth-Protocol.html)。


具体交互流程如下（以CAS协议为例子）：

![cas_flow_diagram](/img/in-post/cas_flow_diagram.png)


参考文档
-------

1. [SSO with CAS or OAuth?](http://stackoverflow.com/questions/2033026/sso-with-cas-or-oauth)
