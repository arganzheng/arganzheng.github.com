---
title: Linux命令学习之——cut命令
layout: post
---


顾名思义，cut命令就是剪切命令，类似于很多的linux命令，cut也是一个行编辑命令。它用于extract sections from each line of input——usually from a file.

> Extraction of line segments can typically be done by bytes (-b), characters (-c), or fields (-f) separated by a delimiter (-d — the tab character by default). A range must be provided in each case which consists of one of N, N-M, N- (N to the end of the line), or -M (beginning of the line to M), where N and M are counted from 1 (there is no zeroth value). Since version 6, an error is thrown if you include a zeroth value. Prior to this the value was ignored and assumed to be 1.

cut支持如下几种格式的切分和抽取：

1. 按照字节抽取(bytes -b)——在处理二进制文件的时候，可用这个
2. 按照字符抽取(characters -c)——在处理文本的时候，多用这个
3. 按照分割符抽取域(fields -f，分割符可以用-d指定，默认是tab)

——如果使用这个的时候，一般可以改成使用awk，但是其实如果仅仅使用这个的话，那么其实应该使用cut，否则就是用牛刀杀鸡了。

**注意** cut只擅长处理“以一个字符间隔”的文本内容

cut在使用-f切分域的时候，不会合并多个切分符号。例如，默认的切分符是tab，但是仅仅是一个tab。因为cut只允许分隔符是一个字符。
这也正是cut的缺点，所以基本上不会使用cut命令处理分隔符是whilespace（tab或者空格）的情况，因为首先tab和空格很难区分，其次可能有多个tab或者空格（特别是空格，基本上都是多个）。这种情况下，使用awk最合适了。

事实上如果你指定多个分隔符，cut会报错：

	forrest@forrest-laptop:~/study/shell$ cut -d '  ' -f1 sname.txt 
	cut: the delimiter must be a single character
	Try `cut --help' for more information.

**tips** 如何知道一个文件是使用空格还是tab作分隔符？

	forrest@forrest-laptop:~/study/shell$ cut -d ' ' -f1 sname.txt 
	Sr.No		Name
	11	Vivek
	12	Renuka
	13	Prakash
	14	Ashish
	15	Rani

	forrest@forrest-laptop:~/study/shell$ sed -n l sname.txt 
	Sr.No\t\tName$
	11\tVivek$
	12\tRenuka$
	13\tPrakash$
	14\tAshish$
	15\tRani$

用sed -n l可以看到sname.txt的分隔（转移字符会打印出来）情况。
这里第一行使用了两个tab作为分隔，试验一下：

	forrest@forrest-laptop:~/study/shell$ cut -f2 sname.txt 

	Vivek
	Renuka
	Prakash
	Ashish
	Rani
	forrest@forrest-laptop:~/study/shell$ cut -f3 sname.txt 
	Name





	forrest@forrest-laptop:~/study/shell$ awk '{ print $2}' sname.txt 
	Name
	Vivek
	Renuka
	Prakash
	Ashish
	Rani

但是用awk就不会有这个问题。

**实战** 只提交有意义的修改

	$ svn st | grep "^[AMD]"  | cut -c9- | xargs svn ci -m "xxx"

参考文章
--------

1. [《cut命令》-linux命令五分钟系列之十九](http://roclinux.cn/?p=1328)