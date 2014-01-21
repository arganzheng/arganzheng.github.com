---
title: 使用Spring-Security进行登录控制的session问题
layout: post
---

这边文章[Spring Security Form Login Using Database](http://www.mkyong.com/spring-security/spring-security-form-login-using-database/)对Spring Security做了非常简单明了的介绍。而且有一个可以down下来运行的示例工程。笔者试验了一下，发现一些有意思的问题。

访问`http://localhost/SpringMVC/login`，填写用户名和密码，点击登录按钮，发生如下请求：

	Request URL:http://localhost/SpringMVC/j_spring_security_check
	Request Method:POST
	Status Code:302 Moved Temporarily

	Request Headers

	Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
	Accept-Encoding:gzip,deflate,sdch
	Accept-Language:en-US,en;q=0.8,zh-CN;q=0.6
	Cache-Control:max-age=0
	Connection:keep-alive
	Content-Length:49
	Content-Type:application/x-www-form-urlencoded
	Cookie:JSESSIONID=C576AF49D77836D13070B7D8C6EB4401; JSESSIONID=6483843684BD92251656560FCD7ABF69
	Host:localhost
	Origin:http://localhost
	Referer:http://localhost/SpringMVC/loginfailed
	User-Agent:Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.18 Safari/537.36
	Form Dataview sourceview URL encoded
	j_username:mkyong
	j_password:123456
	submit:submit

	Response Headers

	Content-Length:0
	Date:Tue, 21 Jan 2014 02:13:06 GMT
	Location:http://localhost/SpringMVC/welcome
	Server:Apache-Coyote/1.1
	Set-Cookie:JSESSIONID=997BB0ABD9108D86354138157F4EF46B; Path=/SpringMVC

------

点击logout链接，发生如下请求：


	Request URL:http://localhost/SpringMVC/j_spring_security_logout
	Request Method:GET
	Status Code:302 Moved Temporarily

	Request Headers

	Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
	Accept-Encoding:gzip,deflate,sdch
	Accept-Language:en-US,en;q=0.8,zh-CN;q=0.6
	Connection:keep-alive
	Cookie:JSESSIONID=997BB0ABD9108D86354138157F4EF46B; JSESSIONID=6483843684BD92251656560FCD7ABF69
	Host:localhost
	Referer:http://localhost/SpringMVC/welcome
	User-Agent:Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.18 Safari/537.36

	Response Headers

	Content-Length:0
	Date:Tue, 21 Jan 2014 02:14:22 GMT
	Location:http://localhost/SpringMVC/logout
	Server:Apache-Coyote/1.1


可以看到是JSESSIONID这个cookies在起作用。
关闭浏览器再次登录没有问题，可见JSESSIONID不是用session cookies保存。但是由于使用了session，会有如下几个问题：

1. 服务器重需要重新登录，这个验证很简单，重启服务就可以了。
2. 如果用户修改了密码，已经成功登录的session还是能够继续访问。也就是只有有那个session cookies，就不会进行DB校验了。
3. 当有多台服务器就不行了。这点可以通过nginx本地实验一下。

	#user  nobody;
	worker_processes  1;

	#error_log  logs/error.log;
	#error_log  logs/error.log  notice;
	#error_log  logs/error.log  info;

	#pid        logs/nginx.pid;


	events {
	    worker_connections  1024;
	}


	http {
	    include       mime.types;
	    default_type  application/octet-stream;

	    #log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
	    #                  '$status $body_bytes_sent "$http_referer" '
	    #                  '"$http_user_agent" "$http_x_forwarded_for"';

	    sendfile        on;
	    #tcp_nopush     on;

	    #keepalive_timeout  0;
	    keepalive_timeout  65;


	    server {
	        listen       80;

	        server_name   security.arganzheng.me;

	        client_max_body_size 10M;

	        location  / {
	            proxy_pass http://security.arganzheng.me;
	        }
	    }

	    upstream security.arganzheng.me{
	        server 127.0.0.1:8090 max_fails=3 fail_timeout=15s;
	        server 127.0.0.1:8080 max_fails=3 fail_timeout=15s;
	    }
	}
	    

然后hosts绑定一下：

	127.0.0.1 security.arganzheng.me


TIPS: 如果你的浏览器配置了局域网代理，记得要去掉，否则浏览器会优先走代理。

我们先在`http://localhost:8080/SpringMVC/login`上进行登录，登录成功后会跳转到`http://localhost:8080/SpringMVC/welcome`页面，告诉我们登录成功。然后我们再用浏览器访问`http://localhost:8090/SpringMVC/welcome`，由于我们没有在8090机器登录，所有会重定向到`http://localhost:8090/SpringMVC/login`页面让我们登录。再试验一下走nginx会怎样，访问`http://security.arganzheng.me/SpringMVC/welcome`，发现一个诡异的现象，就是基本上百分之百都会重定向到登录页面。把8090服务关闭之后，再访问`http://security.arganzheng.me/SpringMVC/welcome`，发现卡了半天。看nginx的配置，就是每次都是先尝试8090机器，但是max_fails参数没有起作用，3次之后还是一样的卡。怀疑是windows下nginx的bug在笔者的mac下试验一下。

