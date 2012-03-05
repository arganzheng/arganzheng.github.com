---
layout: post
title: 构建可伸缩的大型网站
---

[TOC]

* 语言与框架
    * php: Zend
    * python: django
    * ruby: rail
    * java: springMVC

* 开发环境与服务器环境
    * linux

* 机房

* 服务器硬件

* 服务器软件: 
    * webServer: apache, lightd, nginx
    * appServer: tomcat, jboss, jetty, nodejs

* 代码版本控制
    * svn
    * git

* 自动化构建工具
    * ant: 用来封装shell脚本（定义目标和执行目标）是个不错的选择；可以考虑使用Fabric。
    * maven: java开发的不二选择

* 单元测试工具
    * junit
    * testNG: 能更好的集成测试

UT的可幂性：数据依赖（数据准备）、接口依赖（mock）
RESTful接口如何UT？

* UI自动化测试工具
    * WHY: 
        1. 针对UC，最长路径的功能性测试
        2. 浏览器兼容性测试
    * Selenium

* 集成测试工具
    * hudson
代码提交之后，自动进行集成测试。结合git的branch模型，可以做到直接在主干上进行。

* 架构与设计

* 数据库

* NoSQL

* 缓存

* 队列与排队

* 消息通知与异步处理

* 文件存储（图片）

* 服务化
    * ESB: 太重量级，尽量不要使用
    * dubbo
    * RESTful开放API
    * FileServer: 关键是思想，暴露接口而不是资源（存储）。在这点做的最好的是Amazon。

* 单点登录，统一认证
    * session管理：session or cookies
    * 解决跨域问题

* 数据同步（多站点数据同步）
    * EROSA+EROMANGA+otter

* 监控

* 搜索
    * Lucene
    * elasticsearch
    * 实时build

* 数据分析
    * Map-Reduce
    * Hive
    * HBase

* 性能优化
    * 前端性能优化
        * 减少http请求（合并多个js/css，合并小图标）
        * 并发请求（使用多个图片域名）
        * 减少传递的数据量（js/css压缩，使用单独的图片域名，避免cookies的传递）
        * 静态文件缓存: 浏览器缓存，图片版本号；CDN缓存；squid+apache或者nginx反向代理。
        * 雅虎的10条前端优化


* 后端优化
    * 负载均衡：F5；LVS...
    * 数据库: RAID; SSD；避免大字段（Detail拆分）； 数据库分库；数据库分片。
    * 应用层Cache：redis，memcached, viewCache。
    * JVM调优

* 开发流程优化


### 参考文章：
* [转百万级访问网站前期的技术准备](http://roclinux.cn/?p=2120)
