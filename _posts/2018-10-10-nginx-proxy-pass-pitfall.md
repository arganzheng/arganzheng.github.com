---
title: nginx proxy_pass 的一个『坑』
layout: post
catalog: true
---

前几天在配置线上的 nginx 的时候，遇到一个诡异的问题，还搞了蛮久，记录一下。


首先配置入口：nginx.conf 

```shell
#user  nobody;
worker_processes  4;

error_log  /home/work/nginx/logs/error.log;

pid        /home/work/nginx/logs/nginx.pid;

#Specifies the value for maximum file descriptors that can be opened by this process.
worker_rlimit_nofile 51200;

events {
    use epoll;
    worker_connections  51200;
}


http {
    include       mime.types;
    default_type  application/octet-stream;

    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

    access_log  /home/work/nginx/logs/access.log  main;

    sendfile        on;
    tcp_nopush     on;

    keepalive_timeout  65;

    gzip  on;
   
    include vhost/*.conf; 
}
```

然后配置相应的 vhost/ai-platform.arganzheng.life.conf：

```shell
upstream portal {
    server 10.21.12.88:8080;
    # server 10.21.12.89:8080;
}

upstream blog {
    server 10.21.7.1:8000;
}

server {

    listen       80;
    server_name  ai-platform.arganzheng.life;

    access_log  /home/work/nginx/logs/ai-platform.arganzheng.life.access.log  main;

    root /home/work/apps/frontend/portal;

    location /api/ {
        proxy_pass http://portal;
    }
    

    location /blog/ {
        proxy_pass http://blog/;
    }

    location / {
        index  index.html index.htm;
    }

}
```

这里的陷阱在于 `proxy_pass` 后面有没有带上 `/` ，差别很大。官方文档上有说明，但是可能没有注意，就会出问题的时候莫名其妙。


> A request URI is passed to the server as follows:
> 
> If the proxy_pass directive is specified with a URI, then when a request is passed to the server, the part of a normalized request URI matching the location is replaced by a URI specified in the directive:
>
>     location /name/ {
>         proxy_pass http://127.0.0.1/remote/;
>     }
>
> If proxy_pass is specified without a URI, the request URI is passed to the server in the same form as sent by a client when the original request is processed, or the full normalized request URI is passed when processing the changed URI:
>
>     location /some/path/ {
>         proxy_pass http://127.0.0.1;
>     }
>
> Before version 1.1.12, if proxy_pass is specified without a URI, the original request URI might be passed instead of the changed URI in some cases.


举个例子，上面的配置，如果一个请求 `http://ai-platform.arganzheng.life/api/hello`   

如果配置是：

```
location /api/ {
    proxy_pass http://portal;
}
```

那么转发给 upstream 的URL将会是: `http://10.21.12.88:8080/api/hello`，如果配置的是: 

```
location /api/ {
    proxy_pass http://portal/;
}
```

那么转发给 upstream 的URL将会是: `http://10.21.12.88:8080/hello`。


参考文档
-------

1. [Module ngx_http_proxy_module](http://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_pass)







