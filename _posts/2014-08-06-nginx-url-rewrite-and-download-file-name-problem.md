---
title: nginx URL rewrite与下载文件名称问题
layout: post
---


背景
----

我们的APK官方下载链接原来是`http://app.arganzheng.me/official/download?language=xxx`。这个链接是请求到tomcat，然后在应用服务器中遍历某个目录，找到最新的apk版本，让用户下载。

	/**
	 * 下载APK。处理CDN的真正下载请求。其实应该是在静态资源服务器和nginx处理。
	 */
	@RequestMapping("/download")
	public void downloadApk(HttpServletResponse response) {
		File file = getLatestApkFile();
		if (file == null) {
			logger.error("No apk file for download!!!");
			return;
		}

		InputStream inStream = null;
		try {
			inStream = new FileInputStream(file);
			String fileName = file.getName();
			response.setHeader("Content-Disposition", "attachment; filename=" + fileName); // HttpServletResponse
			response.addHeader("Content-Length", "" + inStream.available());
			response.setContentType("application/octet-stream; charset=UTF-8");
			IOUtils.copy(inStream, response.getOutputStream());
		} catch (FileNotFoundException e) {
			logger.error(e.getMessage() + "; FilePath=" + file.getName());
		} catch (IOException e) {
			logger.error("download apk failed!", e);
		} finally {
			IOUtils.closeQuietly(inStream);
		}
	}


这个方案有个好处，就是apk名称可以随意，而且通过设置`Content-Disposition`头部，客户端总是能够显示正确的下载名称。缺点就是没有任何逻辑，属于静态服务器的职责，可以在nginx层就处理掉。所以就考虑在nginx层处理。那么nginx只能处理静态的文件，所以我们把APK包名称约定为`app-release_official.apk`。那么用户请求`http://app.arganzheng.me/download/app-release_official.apk?language=xxx`时，会下载download目录下的app-release_official.apk包。一切都很简单，直到URL rewrite 出现。


问题
----

为了对用户透明，所以我们打算还是使用老的下载URL。那么URL rewrite就是很自然的一种解决方案：

 	location /download {
        proxy_pass http://static-server;
    }

    location /official/download {
            rewrite ^(.*)$ /download/app-release_official.apk last;
    }

`-s reload`运行发现可以正常下载，但是下载的文件的文件名总是为download！

看起来nginx的URL rewrite对用户是透明的，让用户以为下载的就是请求`http://app.arganzheng.me/official/download?language=xxx`就是请求official下的download文件。于是想着改成redirect让用户感知一下:

 	location /download {
        proxy_pass http://static-server;
    }

    location /official/download {
            rewrite ^(.*)$ /download/app-release_official.apk redirect;
    }

现在是返回一个302临时跳转，但是发现一个问题。就是我们的nginx其实是在8080端口的，用户的请求其实前面还有一层类似于LVS的负载均衡，所以请求到nginx的时候其实是`http://app.arganzheng.me:8080/official/download?language=xxx`，重定向之后就变成了`http://app.arganzheng.me:8080/download/app-release_official.apk?language=xxx`。但是显然，我们的nginx并不直接对外提供服务。所以这个请求是有问题的。

**TIPS**

在做URL rewrite的时候，可以先用redirect测试一下，这样你可以看到改写后的URL符不符合期望。比如通过这个redirect，笔者就发现原来URL write会自动带上原来的请求参数的。不需要rewrite的时候加上$query_string在后面。


解决方案
--------

于是想浏览器除了根据URL决定下载文件名称之外，还有没有其他方式呢？答案就是前面的`Content-Disposition`头部。于是马上配置了一下：

	location /download {
        proxy_pass http://static-server;
    }

    location /official/download {
    	add_header Content-Disposition "attachment; filename=app-release_official.apk";
        rewrite ^(.*)$ /download/app-release_official.apk last;
    }

测试了一下，发现不行！这个header经过rewrite之后完全丢弃了，相当于重新发起一次新的请求！那么只能在rewrite之后匹配的location中做处理了，而且rewrite之间只能通过URL来传递是否需要增加头部这个信息。于是改成：

	location /download {
        if ($arg_from_official){
            add_header Content-Disposition "attachment; filename=app-release_official.apk";
        }
        proxy_pass http://static-server;
    }

    location /official/download {
        rewrite ^(.*)$ /download/app-release_official.apk?from_official=true last;
    }


再次测试，还是不行！！！不可能，这不科学啊！！！难道Chrome不支持`Content-Disposition`这个头部？换成IE浏览器测试了一下，可以了，说明nginx配置没有问题！那么谷歌不可能不遵循HTTP标准的，难道是浏览器缓存？使用隐身模式打开新的窗口，再次测试了一下，终于OK了。前后经历2个多小时，肉流满面啊。。
