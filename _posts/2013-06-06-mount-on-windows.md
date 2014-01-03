---
title: windows下mount网络硬盘
layout: post
---

使用net use命令

    net use 盘符: \\远程机器\共享名\路径 /user:...

例子：

    net use z: \\tencent.com\tfs\部门目录\ECC腾讯电商控股公司\电商研发部\ /user:arganzheng@tencent.com