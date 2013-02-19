---
layout: post
title: sed实战 
---


### 实战1. 删除空行（包括由空格组成的空行）

首先要隆重介绍我们的测试文件，因为这是非常特殊的文件：

    forrest@ubuntu:~/Public$ cat -v wuhui.txt 
    1^M
    ^M
    2^M
      ^M
    3^M
       	^M
    4^M
            	 ^M
    5^M
                    ^M
    6^M
                      		     ^M
                      ^M
                      
我用cat -v选项，显示不可见字符。^M是换行的意思。不过这是不正常的。正常的文件换行符用-v也是看不到的。

这里如下四种空行：

1. 单独一个换行符
2. N个空格＋换行
3. N个TAB＋换行
4. N个（空格＆TAB）＋换行

对于这个文件，如果使用：
    
    forrest@ubuntu:~/Public$ sed /^$/d wuhui.txt
    1
    
    2
      
    3
       	
    4
            	 
    5
                    
    6
                      		     
                  
    forrest@ubuntu:~/Public$ 

是没有效果的。因为由于特殊字符的存在，空行并不是真的空行（^$）。Google了一下，可以使用这个命令来匹配：

    forrest@ubuntu:~/Public$ sed /^[[:space:]]*$/d wuhui.txt
    1
    2
    3
    4
    5
    6
    forrest@ubuntu:~/Public$ 
    
`[[:space:]]`表示空格或者tab的集合，这里有点意外的是居然匹配了^M这个不可见的换行符号。另外，注意到`[[:space:]]`后面跟着一个*，表示匹配0个或多个。

`[[:space:]]`可以用\s表示，如下：
    
    forrest@ubuntu:~/Public$ sed '/^\s*$/d' wuhui.txt
    1
    2
    3
    4
    5
    6
    forrest@ubuntu:~/Public$
    
但是使用转义字符，一定要对命令添加引号：
    
    forrest@ubuntu:~/Public$ sed /^\s*$/d wuhui.txt
    1
    
    2
      
    3
       	
    4
            	 
    5
                    
    6
                      		     
                      
    forrest@ubuntu:~/Public$ 
    
    
### 实战2. 使用正值表达式获取信息——使用括号捕获匹配信息。

假设我们要从svnurl中获取保存本地的目录名，例如：`http://svn.alibaba-inc.com/repos/ali_sourcing/apps/web/intl-myalibaba/tags/20110621_r_release4==>$basedir/ali_sourcing/apps/web/intl-myalibaba/`，`http://svn.alibaba-inc.com/repos/ali_sourcing/apps/web/intl-myalibaba/branches/20110620_61781_5==>$basedir/ali_sourcing/apps/web/intl-myalibaba/`。
    
两种思路：一种是保留我们需要的部分，替换我们不需要的部分：

按照这种思路，我们将`
http://svn.alibaba-inc.com/repos/ali_sourcing/apps/web/intl-myalibaba/tags/20110621_r_release4`
==>`$basedir/ali_sourcing/apps/web/intl-myalibaba/tags/20110621_r_release4`
==>`$basedir/ali_sourcing/apps/web/intl-myalibaba/`

于是写出这样的sed语句：

    echo $svnurl | sed "s#http://svn.alibaba-inc.com/repos/#$basedir#" | sed 's#\(branches\|tags\)/.*##' 
    
但是这种做法有个问题，就是要匹配的文本一直在变化，后面的sed语句是在前面的基础上执行，对于有变量替换的语句，结局很难预料。比如上面的第二个sed替换，将tags或者branches后面的信息去除，如果第一个sed中增加的$basedir含有这两个关键词(tags和branches)，那么就会有问题。

使用RE的一个重要的原则就是匹配文本尽量不变化，即使要变化也必须在可以预知的情况下。这就引出了第二种思路：抽取我们需要的部分，利用它来创建（而不是直接修改原匹配文本）我们需要的信息。
按照这种思路：
`http://svn.alibaba-inc.com/repos/ali_sourcing/apps/web/intl-myalibaba/tags/20110621_r_release4`
==>`ali_sourcing/apps/web/intl-myalibaba/`
==>`$basedir/ali_sourcing/apps/web/intl-myalibaba/`

于是写出了这条sed语句：

    echo $svnurl | sed "s#http://svn.alibaba-inc.com/repos/\(.*\)\(?:branches\|tags\).*#$basedir\1#"

其中(?:)表示组合但是不匹配。
 
写成shell函数就是这个样子：
    
    get_path_from_svnurl()
    {
        local svnurl=$1
        local basedir=$2

        # sed中有变量，不能用单引号
        # 思路1：
        # echo $svnurl | sed "s#http://svn.alibaba-inc.com/repos/#$basedir#" | sed 's#\(branches\|tags\)/.*##' 
        # 思路2：
        echo $svnurl | sed "s#http://svn.alibaba-inc.com/repos/\(.*\)\(?:branches\|tags\).*#$basedir\1#"
    }
    
