---
title: 移除SVN锁
layout: post
---

现在还有用SVN悲观锁的。。赶紧移除吧 

    svn propdel svn:needs-lock -R -q 要移除锁设置的工程目录
    
**TIPS** 在这之前，你可能需要先抢占锁：

    Tortoise==>Check For Modifications==>sell all==>Break Lock==>Commit
