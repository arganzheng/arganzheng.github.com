---
layout: post
title: Spring使用@value annotation注入property变量和环境变量
---


    @Value("#{welife.rpc.sendCoupon.api}")
    private String              sendCouponApi;
    
    @Value("#{systemProperties['env']}")
    private String env;

Spring 3.0之后支持$语法，即`@Value("${welife.rpc.sendCoupon.api}")`

注意：前提是要注入properties的bean是由Spring托管。
