---
title: JS跨域问题及解决方案
layout: post
---


问题
---

### [Same-origin policy](https://en.wikipedia.org/wiki/Same-origin_policy)


Same-origin policy(SOP): 只有当两个资源同源的时候，浏览器才允许他们之间的脚本互相访问，包括请求各自的服务端，修改DOM结构和读取cookies，localStorage等。

源(origin): 一个页面的源(origin)是由三部分(triple)组成的 {protocol, host, port}。 当且只当这个三元组的值完全相同的时候，两个资源才被认为是同源的。


SOP可以防止恶意脚本读取你的信息（比如cookies信息）然后发送到他们的服务器，增加了安全性。

**NOTES**

二级域名不同也是跨域。


解决方案
-------

### 1. [Cross Origin Resource Sharing](https://en.wikipedia.org/wiki/Cross-origin_resource_sharing)

CORS允许服务端决定允许哪些域对资源发起的访问请求。提供比较细粒度的安全控制，对客户端来说基本透明，服务端的开发工作也非常小，是W3C推荐的标准跨域访问方式。

**TIPS** 现代浏览器在跨域访问的时候会提示如下信息：

	XMLHttpRequest cannot load http://external.service/. No 'Access-Control-Allow-Origin' header is present on the requested resource. Origin 'http://my.app' is therefore not allowed access.

#### 原理

当需要请求一个跨域的AJAX请求时，支持CORS的现代浏览器会插入一个额外的"preflight"请求来判断是否有权限进行这个操作：

	OPTIONS /
	Host: bar.com
	Origin: http://foo.com

如果bar.com愿意接收这个请求，那么它会响应如下头部:

	Access-Control-Allow-Origin: http://foo.com
	Access-Control-Allow-Methods: PUT, DELETE


#### 相关HTTP头部

跟CORS相关的HTTP头部如下：

请求头部(Request headers):

* Origin
* Access-Control-Request-Method
* Access-Control-Request-Headers

响应头部(Response headers):

* Access-Control-Allow-Origin
* Access-Control-Allow-Credentials
* Access-Control-Expose-Headers
* Access-Control-Max-Age
* Access-Control-Allow-Methods
* Access-Control-Allow-Headers

#### 历史

In May 2006 the first W3C Working Draft was submitted. In March 2009 the draft was renamed to "Cross-Origin Resource Sharing" and in January 2014 it was accepted as a W3C Recommendation.


**说明** CORS vs JSONP

* It supports HTTP verbs other than GET
* Error handling with JSON-P is tricky or impossible. CORS has access to the HTTP status and the response body just like XHR does
* CORS supports many types of authorization (like Basic Auth or OAuth). JSON-P only supports cookies.
* JSONP can cause cross-site scripting (XSS) issues where the external site is compromised, CORS allows websites to manually parse responses to ensure security.

当然，另一方面，JSONP works on legacy browsers which predate CORS support. CORS is supported by most modern web browsers. 


**TIPS** 

注意，Access-Control-Allow-Origin响应头只能有一个，并且只支持grob表达式。如果要支持多个域名，那么需要在服务端动态的根据请求的域名进行判断，在Access-Control-Allow-Origin头部写上相应的允许域名。例如，假如我们要允许某个接口被`blog.arganzheng.me`和`magiforrest.com`这两个域名极其子域名访问，那么可以在nginx中做如下配置：

    ## cross domain with cros
    location /api/v1/activity {
        if ( $http_origin ~* (http://(.+\.)?(blog.arganzheng.me|magiforrest.com)$) ) {
            add_header "Access-Control-Allow-Origin" "$http_origin";
        }
        proxy_pass http://backenserver;
    }


### 2. [JSONP](https://en.wikipedia.org/wiki/JSONP)

JSONP通过script标签来实现从不同域的服务端请求带callback的JSON数据，从而实现跨域请求的目的。

#### 原理

假设原网页是在 {http, foo.com, 80} 源下，需要请求`http://server.example.com/Users/1234`。根据SOP原则，是访问不了的。但是script标签是没有跨域限制的，所以可以动态的插入script标签，来实现跨域请求：

	<script type="application/javascript"
        src="http://server.example.com/Users/1234?callback=parseResponse">
	</script>

为了能够解析返回的数据，这里对服务端放回的数据有要求，即`${callback}(${JSON});` 其中callback函数是本页面已经定义好的回调函数，这样，拿到JSON数据之后就会自动回调callback函数，达到解析返回JSON数据的目的。

	parseResponse({"Name": "Foo", "Id": 1234, "Rank": 7});


**注意** 因为JSONP是利用script标签，所以其实它只是实现了跨域GET请求。如果需要跨域POST请求，那么就不能使用JSONP。


### 3. [Cross-document messaging](https://en.wikipedia.org/wiki/Web_Messaging)

Web Messaging或者cross-document messaging 是HTML5引入的一个新的API。专门用来解决处于不同域上的文档(windows/frames)之间的通讯，并且是非阻塞的异步通信机制。另外这里的通讯并不涉及到网络通信，因为其实是走浏览器内部的API的。

#### 例子

假设我们有一个文档A位于example.net，另一个文档B位于example.com，并且B包含在一个iframe或者popup窗体。如果我们想要A向B发送消息，那么可以在A页面这么写：

	var o = document.getElementsByTagName('iframe')[0];
	o.contentWindow.postMessage('Hello B', 'http://example.com/');

然后B就可以这样子接收消息了：

	function receiver(event) {
		if (event.origin == 'http://example.net') {
			if (event.data == 'Hello B') {
				event.source.postMessage('Hello A, how are you?', event.origin);
			}
			else {
				alert(event.data);
			}
		}
	}
	window.addEventListener('message', receiver, false);


YouTube就是使用这种方式做跨域访问的。

### 4. [WebSockets](https://en.wikipedia.org/wiki/WebSocket)

WebSocket是一个基于TCP的协议，它提供了一种建立在TCP链接之上的全双工(full-duplex)通信机制。

所以WebSocket的本意并不是为了解决跨域问题，而是提供一种客户端和服务端保持双向通信链接的机制。这样，可以实现服务端的实时推送，避免低效率的轮询。

我们知道，HTTP协议被设计为一个简单的请求-响应协议。虽然HTTP 1.1提供了保持链接的机制，但是即使保持了链接，服务端是没有办法主动推送消息给客户端的。所以在WebSocket之前，服务端如果要推送消息给客户端，只能通过一些特殊的技巧(hacks)，我们称之为[Comet (programming)](https://en.wikipedia.org/wiki/Comet_(programming))。Comet其实是一个编程模型，有多种具体实现，这些实现一般都落到下面两大分类——streaming和long polling：

* Streaming
	* Hidden iframe
	* XMLHttpRequest
* Long polling
	* XMLHttpRequest long polling
	* Script tag long polling

#### 实现

![WebSockets](/media/images/websocket.png)

可以看到，在双向通信之前，WebSocket需要先握手建立持久链接。这个握手协议是建立在HTTP协议之上的：

	GET /chat HTTP/1.1
	Host: server.example.com
	Upgrade: websocket
	Connection: Upgrade
	Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==
	Sec-WebSocket-Protocol: chat, superchat
	Sec-WebSocket-Version: 13
	Origin: http://example.com


服务端回复：

	HTTP/1.1 101 Switching Protocols
	Upgrade: websocket
	Connection: Upgrade
	Sec-WebSocket-Accept: HSmrc0sMlYUkAGmm5OPpG2HaGWk=
	Sec-WebSocket-Protocol: chat

一旦链接建立，双方就会切换到双向的二进制通信协议。


#### 例子

**客户端**

1、建立一个WebSocket链接：

	var connection = new WebSocket('ws://html5rocks.websocket.org/echo', ['soap', 'xmpp']);

2、监听回调事件：

	// When the connection is open, send some data to the server
	connection.onopen = function () {
	  connection.send('Ping'); // Send the message 'Ping' to the server
	};

	// Log errors
	connection.onerror = function (error) {
	  console.log('WebSocket Error ' + error);
	};

	// Log messages from the server
	connection.onmessage = function (e) {
	  console.log('Server: ' + e.data);
	};

3、发送消息给服务端

	// Sending String
	connection.send('your message');

	// Sending canvas ImageData as ArrayBuffer
	var img = canvas_context.getImageData(0, 0, 400, 320);
	var binary = new Uint8Array(img.data.length);
	for (var i = 0; i < img.data.length; i++) {
	  binary[i] = img.data[i];
	}
	connection.send(binary.buffer);

	// Sending file as Blob
	var file = document.querySelector('input[type="file"]').files[0];
	connection.send(file);	


**服务端(以Node.js为例)**

	// Require HTTP module (to start server) and Socket.IO
	var http = require('http'), io = require('socket.io');

	// Start the server at port 8080
	var server = http.createServer(function(req, res){ 

		// Send HTML headers and message
		res.writeHead(200,{ 'Content-Type': 'text/html' }); 
		res.end('<h1>Hello Socket Lover!</h1>');
	});
	server.listen(8080);

	// Create a Socket.IO instance, passing it our server
	var socket = io.listen(server);

	// Add a connect listener
	socket.on('connection', function(client){ 
		
		// Success!  Now listen to messages to be received
		client.on('message',function(event){ 
			console.log('Received message from client!',event);
		});
		client.on('disconnect',function(){
			clearInterval(interval);
			console.log('Server has disconnected');
		});

	});


**TIPS** [Socket.IO](http://socket.io/)

Socket.IO is a WebSocket API created by Guillermo Rauch, CTO of LearnBoost and lead scientist of LearnBoost Labs. Socket.IO will use feature detection to decide if the connection will be established with WebSocket, AJAX long polling, Flash, etc., making creating realtime apps that work everywhere a snap. Socket.IO also provides an API for Node.js which looks very much like the client side API.


### 5. 其他一些解决方案


#### 1、server-side proxy

这个原理非常简单，因为SOP只是浏览器的限制，服务端是可以任意请求任何域名的。所以可以利用服务端来做代理中转。但是这种模式只能用于不需要cookies验证的情况。

具体例子可以参考这篇文档：[JavaScript: Use a Web Proxy for Cross-Domain XMLHttpRequest Calls](https://developer.yahoo.com/javascript/howto-proxy.html)


#### 2、[document.domain](http://javascript.info/tutorial/same-origin-security-policy)


对于顶级域名相同的跨域情况，还可以使用document.domain实现跨域请求。

假设我们有一个窗体(window)在http://site.com域名，和两个iframes。第一个iframe来自于http://john.site.com, 另一个来自于http://peter.site.com。那么我们可以将他们的document.domain属性都设置为site.com，这样他们之间就可以实现跨域访问了。

**注意** 

1. 新的document.domain值必须设置为共同的父级域名
2. 所有需要通讯的窗体/iframe都需要设置document.domain。即使主窗体的domain值已经是site.com，你仍然需要重新设置一下，可以用`document.domain=document.domain`设置。


补充：关于Cookies的跨域问题
------------------------

因为HTTP是无状态的，所以我们经常需要用到Cookie。但是处于安全考虑，Cookie一样具有跨域问题。

每个Cookie都是绑定到一个单独的域名，或者它的子域名。也就是说Cookie可以支持子域，但是不能完全跨域。

如果在服务端执行如下JSP代码：

	<body>
	        <%
	        Cookie cookie = new Cookie("username", "arganzheng");
	        cookie.setDomain("www.google.com");
	        response.addCookie(cookie);
	        %>
	</body>

发现服务器实际上返回了一个 `Set-Cookie:username=arganzheng; Domain=www.google.com` 的HTTP头。

但是浏览器接收到这个Set-Cookie头部，并不会盲目的在本地设置cookie。而是会检查该cookie与当前的域名是不是同一个（或者父子关系）。如果不是，那么它会忽略这个请求。所以这时候你去请求www.google.com/ncr。发现并没有带上这个cookie，就是这个原因。

请求页面的时候也是会做类似的判断，只有同域或者子域的cookie才会被浏览器发送过去。

### 解决方案

cookies的跨域问题可以而且基本都需要建立在JS跨域的基础上。只要你能够发出一个跨域的JS请求到跨域的服务端，让其设置你想要发送的cookies，就可以达到跨域设置cookies的目的。比如假设B站有一个设置cookies的接口：

	<?php
	    setcookie("visited", "arganzheng's site", time()+3600);
	?>

然后在A站的HTML页面就可以使用img标签的做跨域设置cookies的请求：

	<!DOCTYPE html>
	<html>
	 <head>
	  <script>
	   //place set cookie for own domain here
	   //
	   //
	  </script>
	 </head>
	 <body>

	  <!-- setting cookies to other domains -->
	  <img src="http://anotherdomain.com/cookies.php" style="display:none;" />
	  <!-- setting cookies to others domains ends -->

	  <!-- place other htmls here -->
	 </body>
	</html>

这样，当用户访问A站的这个页面的时候，就会自动发送一个设置cookies的请求到B站服务端。从而到达用户同时"签到"的效果。这就是Google等公司如何同时设置多个域名的cookies。这样当你登陆gmail.com，它会同时往youtube.com、google.com等其他域名设置相同的cookies。

另一种现代的解决方案就是使用前面提到的CROS。

服务端需要支持：

	header("Access-Control-Allow-Origin: *");
	header("Access-Control-Allow-Credentials: true");
	header("Access-Control-Allow-Methods: GET, POST");
	header("Access-Control-Allow-Headers: Content-Type, *");

然后客户端需要多提交如下两个参数:

	crossDomain: true
	xhrFields: { withCredentials: true }

这样，不同域名的cookies也会被浏览器发送到允许的服务端去。

具体可以参考: [Cookies With My CORS](https://quickleft.com/blog/cookies-with-my-cors/)。


参考文章
-------

1. [Cross-Domain requests in Javascript](https://jvaneyck.wordpress.com/2014/01/07/cross-domain-requests-in-javascript/)
2. [Using CORS](http://www.html5rocks.com/en/tutorials/cors/)
3. [Cross-window messaging with postMessage](http://javascript.info/tutorial/cross-window-messaging-with-postmessage)
4. [WebSocket and Socket.IO](https://davidwalsh.name/websocket)
