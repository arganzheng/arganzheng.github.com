---
layout: post
title: 如何实现用户认证授权统
---


需求
----

1. 用户注册
2. 登陆(Authentication)
3. 登出
4. 访问权限控制（Authorization，匿名用户、普通用户、管理员）

实现
----

### Authentication

一旦用户注册之后，用户信息就保存在服务器端（DB/Cache）。关键在于用户需要提供身份凭证，一般是用户名和密码。即常见的登陆页面：用户输入username和password，勾选Remember Me（可选，一般是记住一周），点击登陆，提交请求到服务端（这里一般是走HTTPS）。服务端根据用户名和密码到数据库查询是否匹配，如果匹配的话，说明身份认证成功。这是一次普通的身份认证过程。非常好理解。

关键在于HTTP是无状态的，用户登陆过一次，但是如果你没有做一些状态管理操作的话，用户登陆后请求同一个页面，服务器仍然要求其登陆。

这时候就需要做一些状态处理了。

一般来说，我们要达到这样子的目的：

1. 用户登陆之后，在其关闭浏览器之前，不需要登陆。
2. 如果用户登陆时勾选了 Remember Me，那么在记住的这段时候内用户访问该网站都不需要登陆。

对于第一个需求，其实就是说在session期间不需要登陆，那么使用session作为状态管理是最合适的选择了。

关于session有几个注意事项：

1. sessionId其实是需要来回传递的，一般来说是通过[Session cookie](http://en.wikipedia.org/wiki/HTTP_cookie#Session_cookie)来实现（REWRITE URL或者Hidden Field安全性和可操作性都不强，比较少见）。
2. sesionId一般是服务器随机生成的一个ID，基本来说是不可能被猜测出来的，所以这方面的安全还是有一定保障的。但是，要防止攻击者获取一个合法的sessionId是相当困难的，这基本上不是开发者所能控制的。相对来说，Session Cookie是比较安全的传递机制，但是也不能完全保证。
3. 在分布式环境下必须保证用户登录一台机器，下次请求路由到另一台web服务器也能够认识这个用户。这一般通过两种方式实现: 1. session复制。2. Session Sticky。这种方式将同一用户的请求转发到特定的web服务器上，避免了集群中Session的复制，缺点是用户只跟一种的一台服务器通信，有单点问题[Cookies, Sessions, and Persistence](http://www.f5.com/pdf/white-papers/cookies-sessions-persistence-wp.pdf)。3. 使用redis这样的集中式的分布式缓存系统实现session集群，这是推荐的方式。

第二个需求，需要使用[Persisitent cookie](http://en.wikipedia.org/wiki/HTTP_cookie#Persistent_cookie)来实现。具体参考这篇文章 [Using Cookies to implement a RememberMe functionality](http://java.dzone.com/articles/using-cookies-implement) 

#### 单点登录（SSO）

实现单点登录，有个最简单的方式。就是顶级域名的cookies + redis这样的session集群。前提就是所有的子系统必须在同样的顶级域名下。

当然，不同域名的子系统也可以实现的。具体可以看一下这篇文章，写的非常好。[Building and implementing a Single Sign-On solution](http://merbist.com/2012/04/04/building-and-implementing-a-single-sign-on-solution/)。

#### session-less

上面的Authentication方式其实是用到了session和cookies。我们知道session这东西是服务端状态，而服务端一旦有状态，就不是很好线性扩展。其实对于身份验证来说，服务端保留的也这是一个简单的value而已，一般是userId，即`session['sessionId']==>userId`。然后再根据userId去DB获取用户的详细信息。如果我们把userId作为一个cookies值放在客户端，然后把用几个cookies值（比如userId）做一个特殊的签名算法得到的token也放在cookie中，即`f(userId, expireTime)==>token`。这样服务端得到用户请求，用同样的签名算法进行计算，如果得到的token是相等，那么证明这个用户是合法的用户。注意这个签名算法的输入因子必须包含过期时间这样的动态因子，否者得到的token永远是固定的。这种实现方式其实是仿造CSRF防御机制[anti-csrf.md](https://gist.github.com/arganzheng/6113349)。是笔者自己想出来的，不清楚有没有人用过，个人感觉行得通。

更进一步的，可以参考API中的签名验证方式，把password作为secretKey，把userId, expireTime, nonce, timestamp作为输入参数，然后用不公开的算法（这个与API签名不同）结合password这个secretKey进行计算，得到一个签名，即f`(userId, expireTime, nonce, tampstamp, password)==>sign`。每次客户端传递userId, expireTime, nonce, tampstamp和sign值，我们根据userId获取到password，然后进行`f(userId, expireTime, nonce, tampstamp, password)==>sign`计算，然后比较两个sign是否一致，如果是，表示通过。这种方式比起上面的方式其实区别在于增加了password作为输入参数。这样当用户修改了password之后，这个token就失效了，更合理安全一些。

google了一下，发现这篇文章跟我的观点不谋而合[Sessionless_Authentication_with_Encrypted_Tokens](http://eversystems.eu/Document/15/Sessionless_Authentication_with_Encrypted_Tokens)。另外看了一下[Spring Security的Remember Me](http://static.springsource.org/spring-security/site/docs/3.0.x/reference/remember-me.html)实现，原来这种方式是[10.2 Simple Hash-Based Token Approach]方式。他的hash因子中也有password，这样当用户修改了password之后，这个token就失效了。不过这种方式没有办法解决token被盗取的问题。要解决token盗取问题，需要使用动态token，这需要持久化token，比较麻烦[10.3 Persistent Token Approach](http://static.springsource.org/spring-security/site/docs/3.0.x/reference/remember-me.html)。


### Authorization

认证之后的下一步就是授权了。因为资源(体现在URL上)往往是有分访问等级的。有些是public的，匿名用户就可以访问。有些是私人的，必须登陆才行，有些是需要管理员才能处理的。所以我们还需要判断这个用户是否有权限对这个资源进行某些操作。这个是通过角色来处理的。

每个用户有个角色，每个资源也可以定义一个角色，就可以简单判断了。资源的访问权限配置可以通过全局的URL匹配，比如/my/xxx的URL必须用户登陆，/admin/xxx的必须是管理员权限。也可以是方法级别的细粒度，这个可以使用anotation简单的做到。

    @Controller
    @RequestMapping("/my")
    public class MyController{
    
        @RequireLogin("role=amdin")
        public void doXXX(){
          xxx
        }
    
    }
  
  
参考文章
--------

1. [你必须了解的Session的本质](http://www.360weboy.com/php/session-cookie/session_essence.html)
2. [你会做Web上的用户登录功能吗？](http://coolshell.cn/articles/5353.html)
3. [The definitive guide to forms based website authentication](http://stackoverflow.com/questions/549/the-definitive-guide-to-forms-based-website-authentication#477579) 这篇帖子真是有意思，回答基本都是长篇博文了[/偷笑]
4. [Building and implementing a Single Sign-On solution](http://merbist.com/2012/04/04/building-and-implementing-a-single-sign-on-solution/) 关于单点登陆笔者目前看到最好的文章了。

