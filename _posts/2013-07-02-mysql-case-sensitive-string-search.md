---
layout: post
title: MySQL字符串比较大小写问题
---


刚刚才知道MySQL的varchar类型比较默认是忽略大小写的，还忽略最后的空格。[C.5.5.1. Case Sensitivity in String Searches](http://dev.mysql.com/doc/refman/5.0/en/case-sensitivity.html)

    mysql> SELECT 'a' = 'A';
        -> 1

解决方案是使用BINARY操作符[10.1.7.7. The BINARY Operator](http://dev.mysql.com/doc/refman/5.0/en/charset-binary-op.html)

    mysql> SELECT BINARY 'a' = 'A';
            -> 0
    mysql> SELECT 'a' = 'a ';
            -> 1
    mysql> SELECT BINARY 'a' = 'a ';
            -> 0