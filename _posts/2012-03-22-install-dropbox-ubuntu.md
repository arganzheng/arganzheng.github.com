---
layout: post
title: ubuntu下安装Dropbox
---

本来在ubuntu下安装个软件是一件简单的事情，不值得写成文章。不过由于有“墙”的存在，所以还是记录一下吧，造福后人。

###### 1. 安装dropbox

这个到官网下载相应的deb包安装就可以了。

###### 2. 安装dropbox-deamon

本来安装完dropbox后，执行：

    dropbox start -i

就可以自动安装的，但是由于“墙”的原因，所以会安装不成功，可以手动下载，具体的可以参考这篇文章： [Dropbox for ubuntu无法安装][1]

可以使用这个URL下载：注意要使用https。

    cd 
    wget https://www.getdropbox.com/download?plat=lnx.x87
    tar xzvf dropbox-lnx.x86-1.2.52.tar.gz
    dropbox start
    
这样就可以了。

[1]:http://hi.baidu.com/sunyang_kaka/blog/item/89436545dbbaa451510ffeb6.html "Dropbox for ubuntu无法安装"

