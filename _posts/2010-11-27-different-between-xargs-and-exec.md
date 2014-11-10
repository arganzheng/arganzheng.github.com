---
title: -exec和xargs的区别
layout: post
---


当你在命令行执行：

    $find . -name 'core' -type f -exec rm {} \;

时，find -exec 命令会对每个匹配的文件执行一个单独的rm操作（execute a separate rm for each one）, 正如你手动敲入下面命令：

    rm ./bin/core
    rm ./source/shopping_cart/core
    rm ./backups/core
    ...

但是使用这种方式，如果有100个文件匹配了，那么就需要启100个进程，一个进程处理一个rm命令。一般来说，其越多进程，意味着越耗性能。我们可以换个思路，我们将要删除文件当作参数传递给rm不就可以了吗？也就是说：

    rm ./bin/core
    rm ./source/shopping_cart/core
    rm ./backups/core
    ...

改成：

    rm ./bin/core ./source/shopping_cart/core ./backups/core

但是前提是后面的命令必须支持多参数。像有些命令，比如unzip，就不支持输入多个jar包，所以必须用-exec。

xargs，顾名思义，是对参数进行处理的命令。它的任务就是将输入行转换成下一个命令的参数列表。因此上面的find -exec命令可以改写成：

    find . -name 'core' -type f -print | xargs rm

> With this approach, xargs bundles together as many filename arguments as possible for submission to each invocation of rm that's needed, in compliance with the OS's maximum allowed size for an argument list. This means xargs is guaranteed not only to handle all the arguments, but also to use the smallest possible number of processes in doing so. For example, if each command can handle 100 arguments, and there are 110 filenames to process, there will be two invocations of the command, respectively handling 100 and 10 arguments.

**TIPS**	处理文件/目录名中的空格

上面的例子有一个问题，如果文件或是目录名含有空格，则可能会有些问题，这是因为xargs默认会按照空白自负来划分输入。一个简单的解决办法就是告诉find使用NUL(\0)来分割结果（通过向find提供-print0选项），并且告诉xargs也使用Nul来分隔输入（--null或者-0 for short）。

    find . -name 'core' -type f -print0 | xargs -0 rm

其中操作系统允许的最大参数长度由如下命令得到：

    forrest@ubuntu:~$ getconf ARG_MAX
    2097152

这意味着xargs保证不会因为参数过多而挂掉。所以目前看来唯一需要保证的就是后面的命令支持多参数。比如前面说过的unzip，就不支持多参数，如果你使用xargs xxx.jar

    forrest@ubuntu:~/work/intl-standalone/searchaddbuild/deploy/WORLDS-INF/lib$ unzip -l alibaba-intl-biz-account-api-1.0-Dev.jar
    Archive:  alibaba-intl-biz-account-api-1.0-Dev.jar
      Length      Date    Time    Name
    ---------  ---------- -----   ----
            0  2010-11-24 19:43   META-INF/
          147  2010-11-24 19:42   META-INF/MANIFEST.MF
            0  2010-11-24 19:42   com/
            0  2010-11-24 19:42   com/alibaba/
            0  2010-11-24 19:42   com/alibaba/intl/
            0  2010-11-24 19:42   com/alibaba/intl/biz/
            0  2010-11-24 19:42   com/alibaba/intl/biz/company/
           。。。    
          931  2010-11-24 19:42   com/alibaba/intl/biz/member/api/exception/IllegalRegistInfoException.class
         1055  2010-11-24 19:42   com/alibaba/intl/biz/member/api/AccountCoreInfoRemoteServiceContainer.class
         2030  2010-11-24 19:42   com/alibaba/intl/biz/AccountCenterServicesLocator.class
          467  2010-11-24 19:42   META-INF/INDEX.LIST
    ---------                     -------
        43764                     51 files

但是如果你用xargs unzip，则会得到如下输出：

    forrest@ubuntu:~/work/intl-standalone/searchaddbuild/deploy/WORLDS-INF/lib$ ls | xargs unzip -l 
    Archive:  activation.jar
      Length      Date    Time    Name
    ---------  ---------- -----   ----
    ---------                     -------
            0                     0 files
    forrest@ubuntu:~/work/intl-standalone/searchaddbuild/deploy/WORLDS-INF/lib$ find . -name "*.jar" -type f | xargs unzip -l
    Archive:  ./poi-scratchpad-3.0.jar
      Length      Date    Time    Name
    ---------  ---------- -----   ----
    ---------                     -------
            0                     0 files

而使用-exec就没有问题：

    forrest@ubuntu:~/work/intl-standalone/searchaddbuild/deploy/WORLDS-INF/lib$ ls -exec unzip -l {} \;
    ls: invalid option -- 'e'
    Try `ls --help' for more information.
    forrest@ubuntu:~/work/intl-standalone/searchaddbuild/deploy/WORLDS-INF/lib$ find . -name "*.jar" -type f -exec unzip -l {} \;
         3041  2008-12-16 14:58   freemarker/core/AddConcatExpression$ConcatenatedHashEx.class
         1102  2008-12-16 14:58   freemarker/core/AddConcatExpression$ConcatenatedSequence.class
         3937  2008-12-16 14:58   freemarker/core/AddConcatExpression.class
         1500  2008-12-16 14:58   freemarker/core/AndExpression.class
         2463  2008-12-16 14:58   freemarker/core/ArithmeticEngine$BigDecimalEngine.class
         8050  2008-12-16 14:58   freemarker/core/ArithmeticEngine$ConservativeEngine.class
         。。。

**NOTES**	ls -exec是有问题的，因为ls会将-e作为它的一个选项解释，即：ls -e

**TIPS** 使用xargs的-n和-L选项来控制输入参数数量

如果你要执行的命令只接受1个或是2个参数，比如使用diff命令来对2个文件进行比较，那么xargs的-n选项就会非常有用，它可以指定一次向目标命令提供几个参数，如果参数数量多于你制定的数量，则命令将会被重复调用，直到所有输入都已经得到执行。注意，最后一次调用的参数有可能会少于指定的参数数量。下面让我们来看一个简单的例子：

    $ echo {0..9} | xargs -n 2
    0 1
    2 3
    4 5
    6 7
    8 9

同样的，你也可以使用-L参数制定每次只对某几行的输入进行操作，比如-L1将每次从输入中取一行作为参数传递给待执行的命令，当然，你可以将1改为任意行，但1是最常用的，下面这条命令将演示如何得到每个git commit的代码变化：

    git log -format="%H %P" | xargs -L1 git diff

这样，我们使用xargs的-n或者-L选项，可以达到跟-exec一样的作用：

    forrest@ubuntu:~/work_back/intl-standalone/searchaddbuild/deploy/WORLDS-INF/lib$ find -name "*.jar" | xargs -L1 unzip -l | grep napoli.properties
    404 2010-11-16 17:11 META-INF/autoconf/biz-napoli.properties.vm
    666 2010-11-27 01:49 biz/napoli.properties
    forrest@ubuntu:~/work_back/intl-standalone/searchaddbuild/deploy/WORLDS-INF/lib$

**TIPS**	xargs也用{}表示当前处理的参数：

    forrest@ubuntu:~/work_back/intl-standalone/searchaddbuild/deploy/WORLDS-INF/lib$ ls | xargs -t -I {} mv {} {}.old
    mv activation.jar activation.jar.old 
    mv activemq-core-5.2.0.jar activemq-core-5.2.0.jar.old 
    。。。

这一命令序列通过在每个名字结尾添加 .old 来重命名在当前目录里的所有文件。-I 标志告诉 xargs 命令插入有{}（花括号）出现的ls目录列表的每一行。

实战
----

### 1. SVN提交代码，如果你用-exec提交每个文件，必然被BS。所以最好是用xargs：

    $svn st | grep '^[AMD]' | cut -c9- | xargs svn ci -m "merge: test using xarge"

这样只会有一次提交记录。

### 2. 将lib下面非jar包删除

    forrest@ubuntu:~/work_back/intl-standalone/searchaddbuild/deploy/WORLDS-INF/lib$ ls | sed '/.jar/ d' | xargs rm -rf

### 3. 查找某个文件是否在jar包中

    forrest@ubuntu:~/work_back/intl-standalone/searchaddbuild/deploy/WORLDS-INF/lib$ find -name "*.jar" -exec unzip -l {} \; | grep napoli.properties
          404  2010-11-16 17:11   META-INF/autoconf/biz-napoli.properties.vm
          666  2010-11-27 01:49   biz/napoli.properties

但是注意到这个结果只能告诉你有这个文件，但是没有告诉你是那个jar包。如果你想知道是哪个jar包，可以用如下命令（这个暴强的命令来自于海锋，我等膜拜）：

    forrest@ubuntu:~/work_back/intl-standalone/searchaddbuild/deploy/WORLDS-INF/lib$ find -name "*.jar" -exec sh -c 'unzip -l $1 | xargs printf "$1   %s\n"' {} {} \; | grep napoli.properties
    ./alibaba-intl-commons-napoli-1.0-Dev.jar   META-INF/autoconf/biz-napoli.properties.vm
    ./alibaba-intl-commons-napoli-1.0-Dev.jar   biz/napoli.properties
	
聪妈提供了一个更简单的命令：

    forrest@ubuntu:~/work/intl-standalone/searchaddbuild/deploy/WORLDS-INF/lib$ find . -name "*.jar"  | xargs grep  napoli.properties
    Binary file ./alibaba-intl-commons-napoli-1.0-Dev.jar matches

第三种方式——使用单反引号(``)作命令替换command substitution

达到的效果与xargs非常类似，但是xargs有对命令参数作超长检查，而这个不会。所以不建议在这里使用。但是使用``从上一个命令中获取输入结果是非常有用的。

    forrest@ubuntu:~/work_back/intl-standalone/searchaddbuild_trunk/deploy/WORLDS-INF/lib$ unzip -l `find . -name "*.jar"`
    Archive:  ./poi-scratchpad-3.0.jar
      Length      Date    Time    Name
    ---------  ---------- -----   ----
    ---------                     -------
            0                     0 files
    forrest@ubuntu:~/work_back/intl-standalone/searchaddbuild_trunk/deploy/WORLDS-INF/lib$ for i in `find . -name "*.jar"`; do unzip -l $i | grep napoli.properties; done
          404  2010-11-16 17:11   META-INF/autoconf/biz-napoli.properties.vm
          705  2010-11-26 20:21   biz/napoli.properties
