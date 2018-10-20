---
title: 配置Nginx支持CORS的一个『坑』
layout: post
tags: [nginx]
catalog: true
---

前几天遇到前端跨域问题，之前也整理过这方面的问题：[JS跨域问题及解决方案](http://arganzheng.life/javascript-cross-domain-problem-and-solution.html)。第一种方案 Cross Origin Resource Sharing(CORS) 是目前主流的解决方案。


```
http {

    ...

    upstream scheduler {
        server 10.21.8.66:8080;
    }

    server {
        listen       80;
        server_name  10.21.8.66;

        access_log  logs/host.access.log  main;

        location / {
            root   html;
            index  index.html index.htm;
        }

        location /scheduler/ {
            if ($request_method = 'OPTIONS') {
                add_header 'Access-Control-Allow-Origin' '*';
                add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
                #
                # Custom headers and headers various browsers *should* be OK with but aren't
                #
                add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range';
                #
                # Tell client that this pre-flight info is valid for 20 days
                #
                add_header 'Access-Control-Max-Age' 1728000;
                add_header 'Content-Type' 'text/plain; charset=utf-8';
                add_header 'Content-Length' 0;
                return 204;
            }
            if ($request_method = 'POST') {
                add_header 'Access-Control-Allow-Origin' '*';
                add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
                add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range';
                add_header 'Access-Control-Expose-Headers' 'Content-Length,Content-Range';
            }
            if ($request_method = 'GET') {
                add_header 'Access-Control-Allow-Origin' '*';
                add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
                add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range';
                add_header 'Access-Control-Expose-Headers' 'Content-Length,Content-Range'; 
            }

            proxy_pass http://scheduler/;
        }

        error_page  404              /404.html;

        error_page   500 502 503 504  /50x.html;
        location = /50x.html {
            root   html;
        }

    }
}
```


但是在配置 nginx 的时候却发现总是不生效，表现是 Origin 操作已经成功返回`Access-Control-Allow-Origin`相关的 HEADER ，但是之后的请求还是没有通过：

![nginx-CROS-pitfall](/img/in-post/nginx-cros.bmp)

```
Failed to load http://10.21.8.66/scheduler/api/scheduler/start: No 'Access-Control-Allow-Origin' header is present on the requested resource. Origin 'http://localhost:8088' is therefore not allowed access. The response had HTTP status code 400.
```

搞了半天，后面发现原来是少了一个配置选项（[Module ngx_http_headers_module](http://nginx.org/en/docs/http/ngx_http_headers_module.html)）：

```
Syntax: add_header name value [always];
Default:    —
Context:    http, server, location, if in location
```

> Adds the specified field to a response header provided that the response code equals 200, 201 (1.3.10), 204, 206, 301, 302, 303, 304, 307 (1.1.16, 1.0.13), or 308 (1.13.0). The value can contain variables.
> 
> There could be several add_header directives. These directives are inherited from the previous level if and only if there are no add_header directives defined on the current level.
>
> If the always parameter is specified (1.7.5), the header field will be added regardless of the response code.

意思很直白，就是通过 nginx `add_header` 添加的 header，只有在这些状态码才会生效：200, 201, 204, 206, 301, 302, 303, 304, 307, or 308。其他的状态码必须当配置了`always`才会返回。

加上这个配置项就好了。


参考文档
-------

1. [跨域资源共享 CORS 详解](http://www.ruanyifeng.com/blog/2016/04/cors.html)
2. [Allowing cross origin requests (CORS) on Nginx for 404 responses](https://serverfault.com/questions/393532/allowing-cross-origin-requests-cors-on-nginx-for-404-responses/700670)
3. [Nginx CORS跨域](https://segmentfault.com/a/1190000013007649)
