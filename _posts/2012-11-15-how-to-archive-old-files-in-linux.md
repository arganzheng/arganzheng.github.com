---
title: Linux下如何备份旧文件
layout: post
---

将30天前的文件打包压缩备份，然后删除原文件

今天王岩发了一个shell脚本让大家review一下：

	mmddnew=`date "+%Y%m%d"`
	basedir="/data/tomcatlog/api/reconciliationFiles/"
	echo ${mddnew}
	cd ${basedir}
	echo "压缩"
	find ./ -maxdepth 1 -type f -mtime +30 | xargs tar czvf ./${mmddnew}.tar
	echo "移动压缩包"
	mv ./${mmddnew}.tar ./bak/
	echo "删除旧的文件"
	find ./ -maxdepth 1 -type f -mtime +30 -exec echo {} \; -exec rm {} \;

这个脚本做的事情就是 将某个目录下更新时间在30天以前的所有文件（不包括文件夹）打包压缩，备份到另一个目录，然后删除这些文件。

这里面有一个比较严重的问题，就是xargs其实是对命令行参数有个数限制的。如果这个文件夹中文件数很多的话（或者文件名很长的话），会出现这样一个问题。就是xargs会截断你的参数，分多次调用tar命令，而tar的-c选项又会导致覆盖问题。

谷歌一下，发现还真有人遇到这样的问题：[关于xargs参数被截断, tar文件被覆盖的问题](http://my.oschina.net/leejun2005/blog/77807)。

他的解决方案是先创建一个空的tar包，然后使用追加的方式打包。

于是王严又改成这个样子了：

	mmddnew=`date "+%Y%m%d"`
	basedir="/data/tomcatlog/api/reconciliationFiles/"
	echo ${mddnew}
	cd ${basedir}
	echo "压缩"
	find ./ -maxdepth 1 -type f -mtime +30 | xargs tar uf ./${mmddnew}.tar
	gzip ./${mmddnew}.tar
	echo "移动压缩包"
	mv ./${mmddnew}.tar.gz ./bak/
	echo "删除旧的文件"
	find ./ -maxdepth 1 -type f -mtime +30 | xargs rm

先用追加形式打包，再压缩。

但是还是有一个问题：就是文件名称如果包含空格之类的东东，那么打包出来的jar包有点问题。

使用print0选项可以解决这个问题。具体参见 [xargs超长参数的处理](http://hi.baidu.com/90system/item/14ed563bc45c371c9cc65ecb) 和 [find中的-print0和xargs中-0的奥妙](http://blog.163.com/laser_meng@126/blog/static/16972784420117102638257/)。

于是代码又变成这样了：

	mmddnew=$(date +%Y%m%d)
	basedir="/data/tomcatlog/api/reconciliationFiles/"
	cd ${basedir}
	echo "压缩"
	find . -maxdepth 1 -type f -mtime +30 -print0 | xargs -0 tar uf ${mmddnew}.tar
	gzip ${mmddnew}.tar
	echo "移动压缩包"
	mv ${mmddnew}.tar.gz ./bak
	echo "删除旧的文件"
	find . -maxdepth 1 -type f -mtime +30 -print0 | xargs -0 rm -f

最后一个小优化，就是上面的脚本find了两次，一次打包，一次删除。是不是可以合并成一次呢？如果文件打包成功即删除。答案是可以的，使用简单的for循环就可以搞定：

	for i in `find . -maxdepth 1 -type f -mtime +30`; do tar uf  ${mmddnew}.tar "$i" && rm "$i" ; done

但是有个小小的缺陷：for in 是对字符串按blanks（空格或者换行）切份的，如果文件名中有空格或者换行，则不行。

`tar -u`在linux下如果不存在会自动创建，但是在mac下不行，所以还是先创建一个空的tar包好。




