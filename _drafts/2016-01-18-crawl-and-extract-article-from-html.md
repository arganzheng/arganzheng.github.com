---
title: 网页正文抓取和转码
layout: post
catalog: true
---


开源的正文抽取算法
----------------

### 1. [cx-extractor](http://code.google.com/p/cx-extractor) 

哈工大的《基于行块分布函数的通用网页正文抽取》。基于行块分布函数的通用网页正文抽取：线性时间、不建DOM树、与HTML标签无关。对国内几大门户网站测试准确率95%以上，对1000个网页抽取耗时21.29秒。

### 2. [readability](https://www.readability.com/)

大名鼎鼎的arc90实验室的Readability，该算法已经商业化实现了firefox,chrome插件，及flipboard，并且已经集成进了safari浏览器。已经开放了API：https://www.readability.com/developers/api

### 3. [snacktory](https://github.com/karussell/snacktory)

http://paper.li/
http://tweetedtimes.com/


参考文章
------


1. [网页正文提取技术分析](http://www.iitshare.com/web-text-extraction-technical-analysis.html)
2. [cx-extractor](http://code.google.com/p/cx-extractor) 
3. [网页正文及内容图片提取算法](http://blog.rainy.im/2015/09/02/web-content-and-main-image-extractor/)
4. [我为开源做贡献，网页正文提取——Html2Article](http://www.cnblogs.com/jasondan/p/3497757.html)
5. [readability](https://www.readability.com/)
6. [Snacktory – Yet another Readability clone. This time in Java.](https://dzone.com/articles/snacktory-%E2%80%93-yet-another) Readability的java clone。Github地址：https://github.com/karussell/snacktory 
7. [boilerpipe: Boilerplate Removal and Fulltext Extraction from HTML pages](https://code.google.com/p/boilerpipe/) 论文地址：https://code.google.com/p/boilerpipe/wiki/WSDM2010Paper
8. [VIPS: a Vision-based Page Segmentation Algorithm](http://research.microsoft.com/apps/pubs/default.aspx?id=70027)