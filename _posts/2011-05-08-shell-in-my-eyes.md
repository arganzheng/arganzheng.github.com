---
layout: post
title: shell语言之我见
---

2011-05-08 星期天 热

shell是一门非常简单而原始的语言，容易入手，它最常见的作用是将一连串的unix命令（cd, ls , find, etc.）串起来。所以只要你对这些命令比较熟，你只要熟悉shell的一点点语法和特性就可以马上写出一个小型甚至中型的shell程序了。

最近用shell写了好几个脚本处理一些svn相关操作（拉N个分支，合并N个分之，提交N个分之，等），也算是对shell有比较深入的认识，这里谈谈个人的看法。

我认为shell是一门面向文本的语言。这主要取决于shell的不支持用户自定义类型（struct和class），而它本身的数据类型又少的可怜（只支持两种类型：int和string，有些shell支持其他类型，如bash支持array，ksh支持float等）。因此，它所有的操作，基本也是针对这两种数据类型的。 这导致你调用一个函数的时候，这个函数无法返回一个高级的足够的数据结构给你，如一个list，或者map，或者tuple，换句话说，即使它可以返回Collections，Collection中的元素如果只能是简单的int和string，估计也不能很好的满足你。那么我们只能这么处理——像pile一样处理，一个函数输入文本，调用函数分析文本，从中获取它需要的信息。这样，函数之间调用看起来像是pile通讯。比如像在高级语言中返回一个List<Pair> pairs; 在shell中可以返回如下文本：

    1, one
    2, two
    3, three

调用函数分析这个文本，提取出相应的值（这里shell的for xx in `funcall` 有点让你感觉返回list的感觉，其实是shell将返回的文本进行spit的结果，总算是提供了一个相对比较便利的做法）。可想而知，非常麻烦。所以，建议始终将shell保留在没有自定义数据结构（包括list和map这样的collection）需要的场景，也就是说是unix命令的简单粘合而已。其他情况下，使用perl或者python这样的语言会更方便一些，他们也可以很方便的调用shell。


#### 关于函数和返回值

法一：使用 return 这个命令，把函数中某个数值返回，其实是使用Exist Status作为返回值，使用$?接收返回值

法二：在函数中使用 echo 输出想要返回的结果，使用`function call`接收函数输出结果。

这两者往往是结合在一起的，如果你不想显示在stdout中显示函数的echo值，可以采用function > /dev/null将函数的echo输出到黑洞中。
> Output can be in the form of stdout or a return code value or both.


#### 完备与足够

记得在学习形式语言与自动机理论时，有讨论到4类形式语言和4种自动机：

0. 0型文法——图灵机
1. 正则文法——有穷自动机
2. 上下文无关文法——下推自动机
3. 上下文有关文法——线性有界自动机

一门语言，只要提供分支判断，支持for循环（或者递归调用），那么它就算是一门完备的语言（图灵机），但是完备不一定是足够的，太低级的语言，不利于提高效率。

在shell中，即使他是一门面向文本的语言，但是他对字符串的支持也仅仅局限在利用各个现有unix小工具的基础上。在shell中，你会经常看到这样的语句：

    pure_filename=`echo $file  |awk -F / '{print $NF}' |sed -e "s#^\.##" `

使用awk和sed来处理字符串，而不是像在java或者python这样的含string类型的语言中，直接使用string类型的方法：startwith, substr, index, strip, compare, etc. 

基本上，shell的字符串处理是建立在正则表达式的基础上的，所以对正则表达式要非常了解。

#### 使用正值表达式获取信息——使用括号捕获匹配信息。

一个例子：

    ## 从svnurl中获取保存本地的目录名
    ## 如：http://svn.alibaba-inc.com/repos/ali_sourcing/apps/web/intl-myalibaba/tags/20110621_r_release4==>$basedir/ali_sourcing/apps/web/intl-myalibaba/
    ## http://svn.alibaba-inc.com/repos/ali_sourcing/apps/web/intl-myalibaba/branches/20110620_61781_5==>$basedir/ali_sourcing/apps/web/intl-myalibaba/
    get_path_from_svnurl()
    {
        local svnurl=$1
        local basedir=$2

        # sed中有变量，不能用单引号
        echo $svnurl | sed "s#http://svn.alibaba-inc.com/repos/\(.*\)\(?:branches\|tags\).*#$basedir\1#"
    }

两种思路：一种是保留我们需要的部分，替换我们不需要的部分：

按照这种思路，我们将

    http://svn.alibaba-inc.com/repos/ali_sourcing/apps/web/intl-myalibaba/tags/20110621_r_release4
    ==>$basedir/ali_sourcing/apps/web/intl-myalibaba/tags/20110621_r_release4
    ==>$basedir/ali_sourcing/apps/web/intl-myalibaba/

于是写出这样的sed语句：

    echo $svnurl | sed "s#http://svn.alibaba-inc.com/repos/#$basedir#" | sed 's#\(branches\|tags\)/.*##' 
但是这种做法有个问题，就是要匹配的文本一直在变化，后面的sed语句是在前面的基础上执行，对于有变量替换的语句，结局很难预料。比如上面的第二个sed替换，将tags或者branches后买的信息去除，如果第一个sed中增加的$basedir含有这两个关键词(tags和branches)，那么就会有问题。

使用RE的一个重要的原则就是匹配文本尽量不变化，即使要变化也必须在可以预知的情况下。

这就引出了第二种思路：
抽取我们需要的部分，利用它来创建（而不是直接修改原匹配文本）我们需要的信息。安装这种思路：

    http://svn.alibaba-inc.com/repos/ali_sourcing/apps/web/intl-myalibaba/tags/20110621_r_release4
    ==>ali_sourcing/apps/web/intl-myalibaba/
    ==>$basedir/ali_sourcing/apps/web/intl-myalibaba/

于是写出了这条sed语句：
    echo $svnurl | sed "s#http://svn.alibaba-inc.com/repos/\(.*\)\(?:branches\|tags\).*#$basedir\1#"

其中(?:)表示组合但是不匹配。

