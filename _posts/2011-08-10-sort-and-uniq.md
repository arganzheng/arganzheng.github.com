---
title: sort和uniq tips
layout: post
---

sort和uniq一般搭配使用，因为uniq必须依赖于sort后的结果。

sort是对每一行从头开始比较排序，如果你要比较的只是一行的某一部分，如何处理呢？

解决方案是将要比较的那一部分（field）抽取出来，放在每一行的最前头。这样就可以根据这个fields比较了。

例如：有如下输入：EmployeeName   PayRate/h  WorkingTime

	Beth    4.00     0
	Hatter    8.00     8
	Forrest    2.00     6
	Magi    6.00     5

要根据每天的薪酬（PayRate * WorkingTime）高低排序。可以这么做：

	forrest@forrest-laptop:~/Desktop$ awk '{ printf("%6.2f %s\n", $2 * $3, $0) }' emp.txt | sort

	 0.00 Beth    4.00     0
	12.00 Forrest    2.00     6
	30.00 Magi    6.00     5
	64.00 Hatter    8.00     8

需要注意的是，sort的-k选项说明，很容易踩雷：

	forrest@ubuntu:~/Desktop$ cat sort_test_data.txt 
	0,10000057,1
	1,100001141,2
	2,100001141,1
	3,100001141,1
	4,100001141,1
	5,100001426,1
	6,100001426,1
	9,10000357,1
	7,100003003,1
	8,100003056,1

	forrest@ubuntu:~/Desktop$ sort -n -t , -k 1 sort_test_data.txt 
	0,10000057,1
	9,10000357,1
	1,100001141,2
	2,100001141,1
	3,100001141,1
	4,100001141,1
	5,100001426,1
	6,100001426,1
	7,100003003,1
	8,100003056,1

这里我指定了-n -t , -k 1，理论上是按照第一个字段数值排序，但是结果显示并不是如此。原因如下：

	forrest@ubuntu:~/Desktop$ man sort
	 -k, --key=POS1[,POS2]
	              start a key at POS1 (origin 1), end it at POS2 (default end of line)

关键在这句话，如果没有指定POS2，默认是直接比较到行尾。所以，如果你想要比较第一个字段，那么必须这么指定：

	forrest@ubuntu:~/Desktop$ sort -n -t , -k 1,1 sort_test_data.txt 
	0,10000057,1
	1,100001141,2
	2,100001141,1
	3,100001141,1
	4,100001141,1
	5,100001426,1
	6,100001426,1
	7,100003003,1
	8,100003056,1
	9,10000357,1

注意是-k是[POS1, POS2]，闭区间的，所以：

	forrest@ubuntu:~/Desktop$ sort -n -t , -k 1,2 sort_test_data.txt 
	0,10000057,1
	9,10000357,1
	1,100001141,2
	2,100001141,1
	3,100001141,1
	4,100001141,1
	5,100001426,1
	6,100001426,1
	7,100003003,1
	8,100003056,1

是比较第一到第二字段。

在 [《sort命令的k选项大讨论》-linux命令五分钟系列之二十七](http://roclinux.cn/?p=1472) 其实也有说明，我一开始没有仔细看：
对于员工工资进行排序，我们也使用了-k 3,3，这是最准确的表述，表示我们“只”对本域进行排序，因为如果你省略了后面的3，就变成了我们“对第3个域开始到最后一个域位置的内容进行排序”了。

uniq命令一般用于过滤重复行，但是如果我们需要指定有多少行重复，那么-c选项可以告诉我们这点。比如一个单元测试失败。


参考文章
--------

1. [《sort帮你排序》-linux命令五分钟系列之二十六](http://roclinux.cn/?p=1350) linux大棚 http://roclinux.cn/ 相当靠谱的一个Linux学习网站 
2. [《sort命令的k选项大讨论》-linux命令五分钟系列之二十七](http://roclinux.cn/?p=1472) 写的相当不错，有理论有实例，值得好好一看
