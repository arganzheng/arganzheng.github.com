---
title: nginx重定向问题
layout: post
---


背景
----

经常有这样的需求，对于没有登陆的用户如果访问了某个需求登陆才能访问的链接，系统需要自动将其重定向到登陆页面。登陆成功之后再跳回原来请求的链接。


实现
----

Java中实现重定向非常简单，HttpServletResponse.sendRedirect就可以了：

	try {
		response.sendRedirect(HttpServletRequestTool.getLoginUrl(request));
	} catch (IOException e) {
		log.error("Error redirecting user to login.", ex);
	}

关键是拿到登陆的URL和returnUrl：

	import javax.servlet.http.HttpServletRequest;
	import javax.servlet.http.HttpUtils;

	import org.apache.commons.lang.StringUtils;
	import org.apache.velocity.tools.generic.EscapeTool;
	
	public class HttpServletRequestTool {

		public static String getLoginUrl(HttpServletRequest request) {
			EscapeTool escape = new EscapeTool();  // returnUrl需要转义
			String url = HttpServletRequestTool.getRequestURLForRedirect(request, "/user/login");
			if (request.getMethod().equalsIgnoreCase("GET")) {
				String requestUrl = getRequestURL(request);
				int i = requestUrl.indexOf("returnUrl=");
				if (i > 0) {
					requestUrl = requestUrl.substring(0, i);
				}
				if (requestUrl.startsWith(url) == false) {
					url += "?" + "returnUrl=" + escape.url(requestUrl);
				}
			}

			return url;
		}

		public static String getRequestURLForRedirect(HttpServletRequest req, String urlPath) {
			StringBuffer url = new StringBuffer();
			String scheme = req.getScheme();
			int port = req.getServerPort();

			String contextPath = req.getContextPath();

			url.append(scheme); // http, https
			url.append("://");
			String serverName = req.getServerName();

			url.append(serverName);
			if ((scheme.equals("http") && port != 80) || (scheme.equals("https") && port != 443)) {
				url.append(':');
				url.append(port);
			}

			if (StringUtils.isNotBlank(contextPath)) {
				url.append(contextPath);
			}

			url.append(urlPath);

			return url.toString();
		}

		/**
		 * 获取当前的URL，包括queryString。主要用于构造returnUrl。
		 * 
		 * @param req
		 * @return
		 */
		@SuppressWarnings("deprecation")
		public static String getRequestURL(HttpServletRequest req) {
			StringBuffer sb = HttpUtils.getRequestURL(req);
			if (StringUtils.isNotEmpty(req.getQueryString())) {
				sb.append("?").append(req.getQueryString()).toString();
			}
			return sb.toString();
		}
	}

其实还是有很多细节需要考虑的，比如对returnUrl的转义，还有是否需要添加端口等等。在本机运行了一下，是没有问题的。


问题
----

但是部署到线上就出问题了。

nginx配置如下：

    upstream reading {
                server xx01-xxx-mob00.xx01:9999;
    }

	server {
        listen       8080;
        server_name  reading.arganzheng.me;

        location / {
            proxy_pass http://reading;
            proxy_set_header  X-Real-IP  $remote_addr;
            proxy_set_header X-Forward-For $remote_addr;
        }
    }


如果直接访问tomcat端口，则没有问题。但是如果通过域名reading.arganzheng.me走nginx，则每次重定向总是重定向到 `http://reading.arganzheng.me:8080/user/login?returnUrl=http%3A%2F%2Freading.arganzheng.me%2Fbook%2Fshare`。

一开始怀疑是req.getServerPort()返回的不是80，而是8080，于是想到一个简单的解决方案，就是判断是否域名，如果是域名肯定不能带端口：

	public static String getRequestURLForRedirect(HttpServletRequest req, String urlPath) {
		StringBuffer url = new StringBuffer();
		String scheme = req.getScheme();
		int port = req.getServerPort();

		String contextPath = req.getContextPath();

		url.append(scheme); // http, https
		url.append("://");
		String serverName = req.getServerName();

		url.append(serverName);
		if (isIpAddress(serverName)) { // 如果是域名是不需要增加端口的
			if ((scheme.equals("http") && port != 80) || (scheme.equals("https") && port != 443)) {
				url.append(':');
				url.append(port);
			}
		}
		if (StringUtils.isNotBlank(contextPath)) {
			url.append(contextPath);
		}

		url.append(urlPath);

		return url.toString();
	}

	private static boolean isIpAddress(String serverName) {
		String[] parts = StringUtils.split(serverName, ".");
		for (String part : parts) {
			if (!StringUtils.isNumeric(part)) {
				return false;
			}
		}
		return true;
	}

但是发现结果还是一样的！难道req.getServerName()返回的不是reading.arganzheng.me？不可能吧，还能返回IP？于是打了log看了下serverName和serverPort，以及最后的redirectUrl，结果令人跌破眼镜：serverName=reading，serverPort=80，redirectUrl=http://reading/user/login?returnUrl=http%3A%2F%2Freading%2Fbook%2Fshare`。Nani..这都行。。Google一下，有人建议说配置一个nginx header，然后拿那个header就可以。于是增加了这么一个配置：

    proxy_set_header Host $http_host;

在代码中通过String serverName = req.getHeader("Host"); 结果发现拿到的serverName还是reading！

难道因为我nginx的upstream名称为reading的原因！

	location / {
        proxy_pass http://reading;
	}

那么最简单的方式就是将其改成跟server_name一致：

    upstream reading.arganzheng.me {
                server xx01-xxx-mob00.xx01:9999;
    }

	server {
        listen       8080;
        server_name  reading.arganzheng.me;

        location / {
            proxy_pass http://reading.arganzheng.me;
            proxy_set_header  X-Real-IP  $remote_addr;
            proxy_set_header X-Forward-For $remote_addr;
        }
    }

果然，这样拿到的serverName就是reading.arganzheng.me了。蛋疼啊，为什么这样子设计，有什么好处吗。。算了，但是还是有问题，仍然是重定向到：`http://reading.arganzheng.me:8080/user/login?returnUrl=http%3A%2F%2Freading.arganzheng.me%2Fbook%2Fshare`。看打印出来的loginUrl是`http://reading.arganzheng.me/user/login?returnUrl=http%3A%2F%2Freading.arganzheng.me%2Fbook%2Fshare`。两者的差别仅仅是端口号而已。这个端口号是谁给加上的呢？8080，不就是nginx的监听端口吗？难道又是nginx搞的鬼？Google了一下，果然有人遇到类似的问题: [Nginx redirects to port 8080 when accessing url without slash](http://serverfault.com/questions/351212/nginx-redirects-to-port-8080-when-accessing-url-without-slash) 和 [Removing port from nginx redirect](http://stackoverflow.com/questions/16725372/removing-port-from-nginx-redirect)。其中有个配置项十分可疑：`port_in_redirect`。查看了一下官方文档：


> Syntax:	port_in_redirect on | off;
> Default:	port_in_redirect on;
> Context:	http, server, location
> Enables or disables specifying the port in redirects issued by nginx.
>
> The use of the primary server name in redirects is controlled by the server_name_in_redirect directive.

> Syntax:	server_name_in_redirect on | off;
> Default:	server_name_in_redirect off;
> Context:	http, server, location
> Enables or disables the use of the primary server name, specified by the server_name directive, in redirects issued by nginx. When the use of the primary server name is disabled, the name from the “Host” request header field is used. If this field is not present, the IP address of the server is used.
>
>The use of a port in redirects is controlled by the port_in_redirect directive.

没看懂。。特别是`issued by nginx`这句。。anyway，把默认on改成off试试不就知道了：

    upstream reading.arganzheng.me {
                server xx01-xxx-mob00.xx01:9999;
    }

 	server {
        listen       8080;
        server_name  reading.arganzheng.me;

        port_in_redirect off;

        location / {
            proxy_pass http://reading.arganzheng.me;
            proxy_set_header  X-Real-IP  $remote_addr;
            proxy_set_header X-Forward-For $remote_addr;
        }
    }

发现还是不行！！为什么呢！！！哥已经黔驴技穷了啊。。但是这时候无意中观察到请求的头部，一个信息引起了我的注意：

	Remote Address:119.63.198.134:80
	Request URL:http://reading.arganzheng.me/book/share
	Request Method:GET
	Status Code:301 Moved Permanently (from cache)

就是这句：Status Code:301 Moved Permanently (from cache)。为什么有个from cache的括号？！默认是浏览器做了缓存？！换个浏览器访问，果然可以了！折腾了两个多小时，肉流满面啊。
	