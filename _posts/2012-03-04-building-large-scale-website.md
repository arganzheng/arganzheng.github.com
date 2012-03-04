---
layout: post
title: 构建可伸缩的大型网站
---

## 思想：
* 网络就是计算机
* 一切都是数据
* 富客户端和哑终端
* 提供接口而不是实现

## 开发语言

## 代码版本控制: svn, git

## 自动化构建工具：ant, maven

## 单元测试工具：junit, testNG
可幂性：数据准备、依赖mock、
RESTful接口如何UT？

## UI自动化测试工具：Selenium
WHY: 针对UC，最长路径的功能性测试
浏览器兼容性测试


## 集成测试工具：hudson
代码提交之后，自动进行集成测试。结合git的branch模型，可以做到直接在主干上进行。

## 机房

## 服务器硬件

## 服务器软件: 
    webServer: apache, lightd, nginx
    appServer: tomcat, jboss, jetty, nodejs

## 架构与设计

## 数据库

## 缓存

## 队列与排队

## 消息通知与异步处理

## 文件存储（图片）

## 服务化: dubbo；RESTful开发API; fileServer

## 单点登录，统一认证（session管理，解决跨域问题）

## 数据同步（多站点数据同步）: EROSA+EROMANGA+otter

## 监控

## 搜索

## 数据分析

## 性能优化
### 前端性能优化
#### 减少http请求（独角兽）
#### 并发请求（多个图片域名）
#### 雅虎的10条前端优化
#### 减少传递的数据量（js/css压缩）
#### 静态文件缓存: 浏览器缓存，图片版本号；CDN缓存；squid+apache或者nginx反向代理。


### 后端优化
#### 负载均衡：F5；LVS...
#### 数据库: SSD；避免大字段（Detail拆分）； 数据库分库；数据库分片。
#### 应用层Cache：redis，memcached, viewCache。
#### JVM调优


