---
title: nginx URL rewrite自动增加请求参数问题
layout: post
---


我们需要将一个URL在服务端进行重定向。比如请求: http://study.arganzheng.me/goto?targetUrl=http%3A%2F%2Freading.arganzheng.me%2Fbook%2Fshare&xxx=yyy，将重定向到targetUrl去：http://reading.arganzheng.me/book/share 去。

那么使用nginx的URL rewrite是很明显的解决方案：

 	location /goto {
        rewrite ^(.*)$  $arg_targetUrl redirect;
    }

但是测试发现：http://study.arganzheng.me/goto?targetUrl=http%3A%2F%2Freading.arganzheng.me%2Fbook%2Fshare&xxx=yyy经过上面的重定向之后变成了：http://reading.arganzheng.me/book/share?targetUrl=http%3A%2F%2Freading.arganzheng.me%2Fbook%2Fshare&xxx=yyy。就是请求字符串很讨厌的仍然被带上了。

谷歌了一下，发现很多人也遇到这个问题，解决方案很简单，只要在重定向的URL后面加上问号?作为结束符就可以了：

 	location /goto {
        rewrite ^(.*)$  $arg_targetUrl? redirect;
    }

[An easy way to make nginx redirect without query parameters](https://avitu.wordpress.com/2010/09/14/an-easy-way-to-make-nginx-remove-query-parameters-from-a-request/)

`nginx -s reload` 一下，果然可以了。

当然，主干流程完成之后，最好还要验证一下是不是GET请求，是不是有targetUrl参数。