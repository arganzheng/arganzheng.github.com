---
title: Linux命令学习之——paste命令
layout: post
---


基本上没有使用过paste命令，不过今天系统的回顾和学习了linux的常用命令。发现paste命令还是很不错的。

例如有下面两个数据文件：

	forrest@forrest-laptop:~/study/shell$ cat sname.txt 
	Sr.No	Name
	11	Vivek
	12	Renuka
	13	Prakash
	14	Ashish
	15	Rani

	forrest@forrest-laptop:~/study/shell$ cat smark.txt 
	Sr.No	Mark
	11	67
	12	55
	13	96
	14	36
	15	67

这两个文件其实非常类似于数据库的数据，根据Sr.No作为外键关联起来，我们想得到这样的数据：

	Sr.No	Name	Mark
	11	Vivek	67
	12	Renuka	55
	13	Prakash	96
	14	Ashish	36
	15	Rani	        67

如何得到呢？

用cut和paste可以轻松达到这个目的：

	forrest@forrest-laptop:~/study/shell$ paste sname.txt smark.txt | cut -f1,2,4
	Sr.No	Name	Mark
	11	Vivek	67
	12	Renuka	55
	13	Prakash	96
	14	Ashish	36
	15	Rani       	67

**tips** 其实对于上面的需求，Linux下有个更简单方便的命令专门做这件事——join命令 - join lines of two files on a common field

	forrest@forrest-laptop:~/study/shell$ join sname.txt smark.txt
	Sr.No Name Mark
	11 Vivek 67
	12 Renuka 55
	13 Prakash 96
	14 Ashish 36
	15 Rani 67


参考文章
--------

1. [Putting lines together using paste utility](http://www.freeos.com/guides/lsst/ch05sec03.html)
2. [《paste命令》-linux命令五分钟系列之二十](http://roclinux.cn/?p=1334)