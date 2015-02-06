---
title: nginx日志格式
layout: post
---


最近我们要调用IDL的一个人脸识别接口，但是人家那个接口是部署在南京机房，并且只能内部调用。从最小化成本的考虑，对于海外用户，我们在香港搭建了一个nginx走专线中转进行提速。在国内则是在北京机房。


中转的配置很简单，就是一个简单的proxypass:

	http {
	    upstream idl {
	        server 10.205.xxx.xx:8788;
	    }

	    upstream idl-test {
	        server 10.46.xxx.xx:8787;
	    }

	    server {
	        listen       8080;
	        server_name  face.arganzheng.me;

	        location /idl/api/v1/recognition {
	            rewrite /idl/api/v1/recognition /1 last;
	        }

	        location /1 {
		    proxy_pass http://idl;
	        }

	        location /idl-test/api/v1/recognition {
	            rewrite /idl-test/api/v1/recognition /faceapi last;
	        }

	        location /faceapi {
		    proxy_pass http://idl-test;
	    }
	}

用户是对 http://face.arganzheng.me/idl/api/v1/recognition 发送 HTTP POST 请求就能够转发给南京的IDL服务。其中POST的内容是一张压缩后的图片的BASE64编码。

用WIFI环境下体验都挺顺畅的，但是在2G或者3G环境下，会发现有时候识别1分钟都没有识别出来。显然问题应该是出现在网络质量上，但是有没有数据支撑呢？Nginx的access_log应该支持吧。查看了一下nginx的log配置说明，果然是有相关的参数的。

对于前端用户请求的信息：[Module ngx_http_core_module - Embedded Variables](http://nginx.org/en/docs/http/ngx_http_core_module.html#variables)

* $arg_name
* $args
* $binary_remote_addr
* $body_bytes_sent：number of bytes sent to a client, not counting the response header; this variable is compatible with the “%B” parameter of the mod_log_config Apache module
* $bytes_sent：number of bytes sent to a client (1.3.8, 1.2.5)
* $connection
* $connection_requests
* $content_length
* $content_type
* $cookie_name
* $document_root
* $document_uri
* $host
* $hostname
* $http_name
* $https
* $is_args
* $limit_rate
* $msec
* $nginx_version
* $pid
* $pipe
* $proxy_protocol_addr
* $query_string
* $realpath_root
* $remote_addr：client address
* $remote_port
* $remote_user
* $request：ull original request line
* $request_body：不要轻易log这个选项，会打很多日志的。
* $request_body_file
* $request_completion
* $request_filename
* $request_length：request length (including request line, header, and request body) (1.3.12, 1.2.7)
* $request_method：request method, usually “GET” or “POST”
* $request_time：request processing time in seconds with a milliseconds resolution (1.3.9, 1.2.6); time elapsed since the first bytes were read from the client
* $request_uri
* $scheme
* $sent_http_name
* $server_addr
* $server_name
* $server_port
* $server_protocol
* $status：response status (1.3.2, 1.2.2)
* $tcpinfo_rtt, $tcpinfo_rttvar, $tcpinfo_snd_cwnd, $tcpinfo_rcv_space
* $time_iso8601
* $time_local：local time in the Common Log Format (1.3.12, 1.2.7)
* $uri


对于后端upstream的信息: [Module ngx_http_upstream_module - Embedded Variables](http://nginx.org/en/docs/http/ngx_http_upstream_module.html#variables)

* $upstream_addr
* $upstream_cache_status
* $upstream_cookie_name
* $upstream_header_time
* $upstream_http_name
* $upstream_response_length
* $upstream_response_time
* $upstream_status


还有其他一些变量 [Alphabetical index of variables](http://nginx.org/en/docs/varindex.html)。再结合日志格式说明：[Module ngx_http_log_module - log format](http://nginx.org/en/docs/http/ngx_http_log_module.html#log_format)。于是我们可以这么配置：

    log_format  main  '$remote_addr [$time_local] $upstream_response_time - $request_time "$request" $request_length '
                      '$status - $upstream_status $bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for" ' ;

另外，因为我们的POST数据比较大，增大buffer的大小可以提到一些性能：

	client_max_body_size 10m;
	client_body_buffer_size 2m;

超时方面，默认的60s应该足够，这里就不做调整。


最终的nginx配置如下：

	#user  nobody;
	worker_processes  8;

	pid        logs/nginx.pid;

	events {
	    use epoll;
	    worker_connections  102400;
	}


	http {
	    include       mime.types;
	    default_type  application/octet-stream;

	    log_format  main  '$remote_addr [$time_local] $upstream_response_time - $request_time "$request" $request_length '
	                      '$status - $upstream_status $bytes_sent "$http_referer" '
	                      '"$http_user_agent" "$http_x_forwarded_for" ' ;
	                      
	    access_log  logs/access.log  main;

	    sendfile        on;
	    #tcp_nopush     on;

	    keepalive_timeout  65;

	    # Have nginx do the compression.
	    gzip  on;
	    gzip_comp_level     1;
	    gzip_disable        msie6;
	    gzip_proxied        any;

	    # text/html mime type is automatically included for gzip, have to add the rest
	    gzip_types          text/plain text/css application/x-javascript text/xml application/xml application/rss+xml text/javascript;


	 	# Pass some client identification headers back to the backend_server
	    proxy_set_header   Host             $host;
	    proxy_set_header   X-Real-IP        $remote_addr;
	    proxy_set_header   X-Forwarded-For  $proxy_add_x_forwarded_for;

	    client_max_body_size 10m;
	    client_body_buffer_size 2m;

	     upstream idl {
	        server 10.205.xxx.xx:8788;
	    }

	    upstream idl-test {
	        server 10.46.xxx.xx:8787;
	    }

	    server {
	        listen       8080;
	        server_name  face.arganzheng.me;

	        location /idl/api/v1/recognition {
	            rewrite /idl/api/v1/recognition /1 last;
	        }

	        location /1 {
		    	proxy_pass http://idl;
	        }

	        location /idl-test/api/v1/recognition {
	            rewrite /idl-test/api/v1/recognition /faceapi last;
	        }

	        location /faceapi {
		    	proxy_pass http://idl-test;
	    	}

		 	location /face/stat/pv.do {
	     	    access_log logs/face_pv.log main;
	            proxy_pass http://127.0.0.1:8080/face/stat/postokay;
	        }

	        location /face/stat/act.do {
	            access_log logs/face_act.log main;
	            proxy_pass http://127.0.0.1:8080/face/stat/postokay;
	        }

			location /face/stat/postokay {
	            return 200;
	        }

	   }
	}


注意到这里打点统计还可以指定使用那个log，写到哪个日志文件。

**NOTES** $request_time 和 $upstream_response_time

需要注意的是，对于HTTP POST的请求，两者相差特别大。因为Nginx会把HTTP request body缓存住，接受完毕后才会把数据一起发给后端。

[[Tengine-中文]关于nginx中request_time 和upstream_response_time的疑问](http://code.taobao.org/pipermail/tengine-cn/2012-May/000295.html)


推荐阅读
--------

1. [Logging and Monitoring](http://nginx.com/resources/admin-guide/logging-and-monitoring/) Nginx的日志和监控
2. [Tracking Application Response Time with Nginx](https://lincolnloop.com/blog/tracking-application-response-time-nginx/)
3. [How To Optimize Nginx Configuration](https://www.digitalocean.com/community/tutorials/how-to-optimize-nginx-configuration) 对于我们这个案例主要关注Buffers的相关配置。