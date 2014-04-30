---
title: 动态页面缓存方案
layout: post
---


背景
----

像新闻等这样的比较少变化的页面，提供一个静态页面缓存机制，避免每次都动态拉取数据渲染生成HTML。


解决方案
--------

**总体原则**

应用服务器（这里是tomcat）不处理静态资源的缓存，由前端nginx、CDN和browser来处理。应用通过HTTP头控制前端静态资源缓存的缓存时间和机制。应用服务器这边缓存的是数据（redis等cache），而不是静态页面。这样的好处是系统的边界和职责更加明确。原来的数据逻辑也不需要修改。也不需要再次实现一套静态资源缓存机制。


**约定**

1. 只缓存GET请求
2. 隐私内容不适合缓存。即对于同一个URL，不同用户看到的信息是不一样的。一般是通过cookies或者session来区分不同用户请求的。这种情况下不适合缓存，CDN缓存也有同样问题。


应用服务器
----------

### 职责以及相应的实现

1. 动态生成静态页面
	* 使用velocity做模版引擎
2. 控制静态页面的缓存机制
	* 规定`/cache`目录是走nginx缓存的。**说明** 因为nginx的cache模块不支持if配置，所以无法通过查询字符串cache=true控制。
	* 对于静态的配置，可以使用`@CacheControl`注解进行配置。

			@CacheControl(isPublic = true, maxAge = 300, sMaxAge = 300)

	  会自动生成：`Cache-Control: public, max-age=300, s-maxage=300`的HTTP头。
	

    * 对于动态的配置，可以使用CacheControlHeader工具类方便的控制：

    		int age = calculateLeftTiming();
			String cacheControlValue = CacheControlHeader.newBuilder()
					.setCacheType(CacheType.PUBLIC)
					.setMaxAge(age)
					.setsMaxAge(age).build().stringValue();
			if (StringUtils.isNotBlank(cacheControlValue)) {
				response.addHeader("Cache-Control", cacheControlValue);
			}


具体实现细节，可以参考笔者的另一篇文章：[优雅的Builder模式]()。

前端缓存
--------

前端缓存主要包括Nginx缓存，CDN缓存，浏览器缓存。这些缓存都是针对HTTP请求和相应的HTML静态资源。所以都是遵循HTTP协议中关于缓存的控制的。所以我们这里有必要先学习一下HTTP协议对于缓存的控制规范。

### 0. HTTP缓存控制

#### 1. Time-based cache headers

1. [Cache-Control](http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.9)
	* public: public resources can be cached not only by the end-user’s browser but also by any intermediate proxies that may be serving many other users as well.
	* private: private resources are bypassed by intermediate proxies and can only be cached by the end-client.
	* no-cache | no-store：Both values are required as IE uses no-cache, and Firefox uses no-store.
	* max-age: Defines the freshness window in seconds relative to the request time. If less than max-age seconds have passed since the component was request(downloaded), the browser will use the cached version. The max-age header overrides the Expires header.
	* s-maxage: s的意思是shared cache，比如CDN。会覆盖max-age和expires头。基本上所有的CDN都会遵循这个头部。s-maxage directive is always ignored by a private cache.
	* must-revalidate：必须重新刷新cache。
2. Expires：specify a specific point in time the resource is no longer valid. 老的过期时间控制头，只能接收日期字符串。

#### 2. Conditional cache headers

1. Last-Modified: The date the file last changed. Time-based conditional request. 客户端通过If-Modified-Since询问服务端是否有修改，如果没有服务端返回304状态码。
2. Etag: a unique identifier for a particular version of the file (for instance, an MD5 hash). Contend-based conditional request. 客户端通过If-None-Match头询问服务端是否资源有修改，如果没有服务端返回304状态码。


如果要强制每次都验证内容是否过期：可以设置如下HTTP头：

	HTTP/1.1 200 OK
	Cache-Control: private, must-revalidate, max-age=0, proxy-revalidate, s-maxage=0
	Content-Type: text/html; charset=UTF-8
	Date: Mon, 29 Apr 2013 16:38:15 GMT
	ETag: "bbea5db7e1785119a7f94fdd504c546e"
	Last-Modified: Sat, 27 Apr 2013 00:44:54 GMT
	Server: thin 1.4.1 condename Chrome
	X-Rack-Cache: miss

也可以混合使用：

	HTTP/1.1 200 OK
	Cache-Control: no-transform,public,max-age=300,s-maxage=900
	Content-Type: text/html; charset=UTF-8
	Date: Mon, 29 Apr 2013 16:38:15 GMT
	ETag: "bbea5db7e1785119a7f94fdd504c546e"
	Last-Modified: Sat, 27 Apr 2013 00:44:54 GMT
	Server: AmazonS3
	Vary: Accept-Encoding
	X-Cache: HIT 

**NOTES**

1. the "x-cache: HIT" header, indicating the CDN served the request.
2. Once a resource is invalid, a browser has two options. It can download the resource again, or do a conditional GET request which only downloads the file if it has changed. In order to make a conditional GET request, the browser needs a way to specify what version it has in the cache. Once again it is up to the web server to provide that, and once again the HTTP protocol gives possibilities: the Last-Modified header and the ETag header.
3. Google recommend to use the Last-Modified header above ETag because if the date is sufficiently far back, the browser may choose to skip requesting the file altogether.



### 1. [nginx缓存](http://wiki.nginx.org/HttpProxyModule)

这个是通过Nginx的HttpProxyModule实现的。主要配置参数如下：

* proxy_cache：指定用于页面缓存的共享内存
* proxy_cache_bypass：定义nginx不从缓存取响应的条件
* proxy_no_cache: 定义nginx不将响应写入缓存的条件。可以和proxy_cache_bypass指令一起使用
* proxy_cache_key：定义如何生成缓存的键
* proxy_cache_path：设置缓存的路径和其他参数。缓存数据是保存在文件中的，缓存的键和文件名都是在代理URL上执行MD5的结果
* proxy_cache_use_stale：如果后端服务器出现状况，nginx是可以使用过期的响应缓存的
* proxy_cache_valid：为不同的响应状态码设置不同的缓存时间
* proxy_store：开启将文件保存到磁盘上的功能
* proxy_store_access:  设置缓存目录和文件的访问权限


**说明** 

1. 缓存参数也可以直接在响应头中设定。这种方式的优先级高于使用proxy_cache_valid指令设置缓存时间。 “X-Accel-Expires”响应头可以以秒为单位设置响应的缓存时间，如果值为0，表示禁止缓存响应，如果值以@开始，表示自1970年1月1日以来的秒数，响应一直会被缓存到这个绝对时间点。 如果不含“X-Accel-Expires”响应头，缓存参数仍可能被“Expires”或者“Cache-Control”响应头设置。 如果响应头含有“Set-Cookie”，响应将不能被缓存。 这些头的处理过程可以使用指令proxy_ignore_headers忽略。也就是说，“X-Accel-Expires”，“Expires”，“Cache-Control”，和“Set-Cookie” 这些响应头会影响nginx的缓存时间。
2. 保存文件的修改时间根据接收到的“Last-Modified”响应头来设置。响应都是先写到临时文件，然后进行重命名来生成的。从0.8.9版本开始，临时文件和持久化存储可以放在不同的文件系统，但是需要注意这时文件执行的是在两个文件系统间拷贝操作，而不是廉价的重命名操作。因此建议保存文件的路径和proxy_temp_path指令设置的临时文件的路径在同一个文件系统中。
3. Nginx提供了两种缓存方式: proxy_store和proxy_cache。Proxy_store只是作为一个镜像，用于创建静态无更改文件的本地拷贝。而proxy_cache可以提供过期时间设置，也会根据响应头过期处理。


最终Nginx配置如下：

	http {
	    include       mime.types;
	    default_type  application/octet-stream;

	    #在日志格式中加入$upstream_cache_status
	    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
	                      '$status $body_bytes_sent $request_body "$http_referer" '
	                      '"$http_user_agent" "$http_x_forwarded_for"'
	                      '"$upstream_cache_status" $request_time';

	    # Begin Proxy Cache Config
	    #keys_zone=cache1:100m 表示这个zone名称为cache1，分配的内存大小为100MB
	    #/usr/local/nginx/proxy_cache_dir/cache1 表示cache1这个zone的文件要存放的目录
	    #levels=1:2 表示缓存目录的第一级目录是1个字符，第二级目录是2个字符，即/usr/local/nginx/proxy_cache_dir/cache1/a/1b这种形式
	    #inactive=1d 表示这个zone中的缓存文件如果在1天内都没有被访问，那么文件会被cache manager进程删除掉
	    #max_size=10g 表示这个zone的硬盘容量为10GB
	    proxy_cache_path  data/proxy_cache_dir/cache1  levels=1:2 keys_zone=cache1:100m inactive=1d max_size=2g;
	    proxy_temp_path   data/proxy_temp_dir 1 2;

	    #设置缓存的key
	    # Putting the host name in the cache key allows different virtual hosts to share the same cache zone
	    proxy_cache_key "$scheme://$host$request_uri";

	    # Pass some client identification headers back to the backend_server  
	    proxy_set_header   Host             $host;
	    proxy_set_header   X-Real-IP        $remote_addr;
	    proxy_set_header   X-Forwarded-For  $proxy_add_x_forwarded_for;

	    # Cache different return codes for different lengths of time 
	    # We cached normal pages for 10 minutes
	    proxy_cache_valid 200 302  10m;
	    proxy_cache_valid 404      1m;
	    
	    upstream nantianmen {
	       server arganzheng-mob01.hk01:8091;
	       server arganzheng-mob04.hk01:8091;
	       server arganzheng-mob05.hk01:8091;
	    }
	    
	    server {
	        listen 8080;
	        server_name s.arganzheng.me;
	      
	        # $upstream_cache_status表示资源缓存的状态，有HIT MISS EXPIRED UPDATING STALE BYPASS几种状态
	        add_header X-Nginx-Cache $upstream_cache_status;
	        # nginx的proxy_cache不支持if判断，所以规定/cache的走缓存
	        location ~ /cache {
	            #设置资源缓存的zone
	            proxy_cache cache1;
	        
	            # If it's not in the cache pass back to the backend-server 
	            proxy_pass http://nantianmen;
	        }
	        
	        # Proxy all remaining content to the backend-server
	        location / {
	            proxy_pass http://nantianmen;
	        }
	    }
	}


### 2. CDN缓存

CDN对于我们来说其实就是代表浏览器。所以一般所有的CDN都是遵循HTTP缓存控制的。不过HTTP协议有一些头部是专门针对这些中间代理的，这些头部优先级要高于针对浏览器的头部。比如s-maxage要高于max-age。

### 3. browser缓存

参见前面 HTTP缓存控制 部分。


缓存清空机制以及刷新机制
------------------------


1. 浏览器缓存
	* 服务器端更改Resource URL。一般是通过在URL中嵌入版本号或者ETAG
2. CDN缓存
	* 提供清理接口
3. Nginx缓存
	* 手动清理proxy_cache_path目录，重启Nginx服务
	* 使用`proxy_cache_purge`模块清空。我们的nginx没有安装这个模块，而nginx不能动态加载模块，需要重新编译安装，所以这次没有使用这个。

有了这些机制，可以把缓存时间设置长一些。


参考文章和推荐阅读
------------------

1. [14 Header Field Definitions](http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html)
2. [A Beginner's Guide to HTTP Cache Headers](http://www.mobify.com/blog/beginners-guide-to-http-cache-headers/)
3. [HTTP Caching](https://developers.google.com/speed/articles/caching) 谷歌出品，必属精品！
4. [Increasing Application Performance with HTTP Cache Headers](https://devcenter.heroku.com/articles/increasing-application-performance-with-http-cache-headers) 写的很全面，强烈推荐！
5. [Caching Tutorial for Web Authors and Webmasters](http://www.mnot.net/cache_docs/)
6. [https://devcenter.heroku.com/articles/jax-rs-http-caching](https://devcenter.heroku.com/articles/jax-rs-http-caching)
7. [Nginx添加proxy_cache模块](http://derekzhan.iteye.com/blog/1986092)
8. [An Nginx Load Balancing, Caching, Reverse Proxy](http://people.adams.edu/~cdmiller/posts/nginx-reverse-proxy-cache/)
9. [nginx cache查看命中率](http://www.361way.com/nginx-cache/2665.html)
