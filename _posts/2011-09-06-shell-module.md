---
layout: post
title: shell如何模块化和复用——shell深入学习
---

基本上所有的编程语言都支持模块化，以达到功能复用的效果。比如java和python的import xxx。C/C++的include。那么shell支持模块化吗？

shell本质上并不支持模块化，但是有些机制可以使它达到类似的效果。

### 一些背景知识

首先要了解有两种方式可以执行一个shell脚本：

##### 1. 一种是新产生一个shell，然后执行相应的shell脚本

方法是在scripts文件开头加入以下语句

    #!/bin/bash

一般的script文件(.sh)即是这种用法。这种方法先启用新的sub-shell（新的子进程）,然后在其下执行命令。

也可以直接在命令行中指定shell类型，如：

    $ sh script文件


##### 2. 一种是在当前shell下执行，不再启用其他shell

方法是使用source命令，不再产生新的shell，而在当前shell下执行一切命令。

也有两种语法：

    $ source script文件

或者直接用点号：

    $ . script文件

一个非常形象的类比是：shell的source就是python的import（或者c中的include）。

**注意**：shell不会判断一个shell脚本是不是被导入多次，每次source（或者点号）scriptFile，都会在当前shell中执行scriptFile。这点和C的include是一样的。但是正如C可以使用条件包含避免重复导入头文件，shell也有类似的机制。这个我们在下面会讲到。

了解了背后的原理和机制，现在让我们回过头来看看client shell中如何导入module shell。

### 实战

一个例子胜过千言万语，我们将采用循序渐进的方式逐渐修复到最完美的方案。

先解决最简单的导入情况——module shell和client shell在一个文件夹下：

子模块定义：

    forrest@forrest-laptop:~/study/shell$ cat amodule.sh 
    #!/bin/bash

    global_var="This is a global variable define in amodule."

    say_hello(){
        global_var_define_in_function="This is a global variable define in a function in amodule." 
        local local_var="This is a local variable define in a function in amodule." 
        echo "Hello, $1. This is a function define in amodule."        
    }

使用脚本定义：

    forrest@forrest-laptop:~/study/shell$ cat main.sh 
    #!/bin/bash

    . ./amodule.sh

    say_hello "Forrest Gump"
    echo "global_var=$global_var"
    echo "global_var_define_in_function=$global_var_define_in_function"
    echo "local_var=$local_var"

执行结果：

    forrest@forrest-laptop:~/study/shell$ ./main.sh 
    Hello, Forrest Gump. This is a function define in amodule.
    global_var=This is a global variable define in amodule.
    global_var_define_in_function=This is a global variable define in a function in amodule.
    local_var=

可以看到，在main.sh中通过source amodule.sh，main.sh可以直接使用amodule.sh中定义的全局变量和函数。而局部变量是拿不到的。

但是在比较大型的shell项目中，module shell和client shell往往不在一个目录下，而且module shell之间还会相互依赖（导入），你会发现使用相对路径会出现找不到module shell的情况。看下面这个比较复杂而真实的例子：

    forrest@forrest-laptop:~/study/shell/svn_tools$ tree 
    .
    ├── modules
    │   ├── log.sh
    │   ├── svn_core.sh
    │   └── utils.sh
    └── svncobranches.sh

    1 directory, 4 files

依赖（导入）关系是：

svncobranches.sh导入modules/svn_core.sh，svn_core.sh导入utils.sh和log.sh，log.sh导入utils.sh。当然，svncobranches.sh也可能依赖modules/log.sh和modules/utils.sh，但是通过svn_core.sh已经间接导入了。

一般来说是这样的导入方式：

    forrest@forrest-laptop:~/study/shell/svn_tools$ cat svncobranches.sh 
    #!/bin/bash

    . modules/svn_core.sh

    forrest@forrest-laptop:~/study/shell/svn_tools$ cat modules/svn-core.sh 
    . utils.sh
    . log.sh

    forrest@forrest-laptop:~/study/shell/svn_tools$ cat modules/log.sh
    . utils.sh 

但是因为我们是运行svncobranches.sh，当svncobranches.sh在source svn_core.sh时，svn_core.sh的相对路径不是在modules目录，而不是在svn_tools目录。这样svn_core.sh导入utils.sh和log.sh就会出错：

    forrest@forrest-laptop:~/study/shell/svn_tools$ sh svncobranches.sh 
    .: 1: utils.sh: not found

而如果我们不是在svn_tools目录下运行svncobranches.sh，那么svncobranches.sh导入. modules/svn_core.sh就已经报错了：

    forrest@forrest-laptop:~$ sh ~/study/shell/svn_tools/svncobranches.sh 
    .: 5: Can't open modules/svn_core.sh

这个问题更严重了，因为我们不能强迫用户一定在client shell所在目录下运行我们的client shell脚本。如何处理呢？

注意到shell的source是这么一个查找机制：被source的module shell script，shell会在$PATH环境变量中搜索。根据这个信息，将我们的modules目录放在环境变量$PATH中，就可以不用指定相对或者绝对路径了。

但是关键在于第一个source，就是你的client shell导入modules shell，必须找到你当前的运行目录。这可以通过dirname得到：

    forrest@forrest-laptop:~/study/shell/svn_tools$ cat svncobranches.sh 
    #!/bin/bash

    # determine base directory; preserve where you're running from
    basedir=$(dirname $0)
    #echo $basedir
    export PATH=$PATH:$basedir/modules

    . svn_core.sh

    svncobranch http://svn.alibaba-inc.com/repos/ali_sourcing/share/biz/product/branches/20110817_73701_2 /home/forrest/work/intl-biz/product

    forrest@forrest-laptop:~/study/shell/svn_tools$ cat modules/svn-core.sh 
    . utils.sh
    . log.sh

    ### svn上将指定的SVN URL co到本地指定的目录，这里只是演示，打个log而已。
    # svncobranch svnurl path 
    svncobranch()
    {
        local svnurl=$1
        local path=$2

        log "svncobranch($svnurl, $path)"
    }

    forrest@forrest-laptop:~/study/shell/svn_tool/modules$ cat log.sh 
    # log(meg, loglevel)
    log()
    {
        datetime=`date +"%y-%m-%d %H:%M:%S"`
        message=$1
        if [ -z "$2" ]
        then
            loglevel="INFO"
        else
            loglevel=$2
        fi

        if [ ! -d log/ ]; then
            mkdir log
        fi

        logfname=`removepostfix $0`
        echo "$datetime [$0] $loglevel :: $message" | tee -a "log/$logfname.log"
    }

    log_error()
    {
            log "$1" "ERROR"
    }

    log_info()
    {
            log "$1" "INFO"
    }
    log_debug()
    {
            log "$1" "DEGUG"
    }

    log_warn()
    {
            log "$1" "WARN"
    }

    forrest@forrest-laptop:~/study/shell/svn_tools/modules$ cat utils.sh
    # trim(str)
    # remove blank space in both side
    trim()
    {
        echo $*
    }

    remove the postfix
    # svncobranches.sh ==> svncobranches
    removepostfix()
    {
        filename=$(basename "$1")              
        echo "$filename" | sed 's/.sh$//'
    }

    forrest@forrest-laptop:~$ sh ~/study/shell/svn_tool/svncobranches.sh 
    11-09-18 10:50:06 [/home/forrest/study/shell/svn_tools/svncobranches.sh] INFO :: svncobranch(http://svn.alibaba-inc.com/repos/ali_sourcing/share/biz/product/branches/20110817_73701_2, /home/forrest/work/intl-biz/product)

可以看到，我们的模块化策略其实是变成这样了：由client shell将整个modules目录放入到PATH环境变量中，那么在该shell进程中所有的导入都不需要指定相对或者绝对路径。好像它们都在同一个目录下一样。

上面的程序在一般情况下运行良好，不过有个小问题，就是如果用户对你的shell建立了一个软链接，那么需要follow symbol link：
    forrest@forrest-laptop:~$  ln -s /home/forrest/study/shell/svn_tools/svncobranches.sh svncobranches.sh
    forrest@forrest-laptop:~$ sh svncobranches.sh
    .
    .: 10: svn_core.sh: not found

使用readlink -f $0可以得到真实的文件路径：

    forrest@forrest-laptop:~/study/shell/svn_tools$ cat svncobranches.sh 
    #!/bin/bash

    # determine base directory; preserve where you're running from
    #echo "Path to $(basename $0) is $(readlink -f $0)"
    realpath=$(readlink -f "$0")
    basedir=$(dirname "$realpath")
    #echo "basedir=$basedir"

    export PATH=$PATH:$basedir/modules

    . svn_core.sh

    svncobranch http://svn.alibaba-inc.com/repos/ali_sourcing/share/biz/product/branches/20110817_73701_2 /home/forrest/work/intl-biz/product

    forrest@forrest-laptop:~$ sh svncobranches.sh

    Path to test.sh is /home/forrest/study/shell/svncobranches.sh
    basedir=/home/forrest/study/shell
    11-09-18 11:07:25 [svncobranches.sh] INFO :: svncobranch(http://svn.alibaba-inc.com/repos/ali_sourcing/share/biz/product/branches/20110817_73701_2,/home/forrest/work/intl-biz/product)

最后一个问题，就是我们在前面一开始提到的重复导入问题：如何避免一个module shell被导入多次呢？

首先看一下如果发生这种事情，会出现什么状况。

我们知道log.sh依赖于utils.sh，假如我们让utils.sh也依赖于log.sh（这个很正常，log本来就是很基础的服务。）

    forrest@forrest-laptop:~/study/shell/svn_tools/modules$ cat utils.sh
    . log.sh

    # trim(str)
    # remove blank space in both side
    trim()
    {
        echo $*
    }

    # remove the postfix
    # svncobranches.sh ==> svncobranches
    removepostfix()
    {
        filename=$(basename "$1")              
        echo "$filename" | sed 's/.sh$//'
    }

执行结果是根本执行不了，直接包错了：

    forrest@forrest-laptop:~/study/shell/svn_tools$ sh svncobranches.sh 
    .: 1: 3: Too many open files

这是因为shell在source的时候陷入了死循环了，因为log.sh和utils.sh互相依赖导致的。

怎么解决呢？其实这个问题在C中包含头文件也有类似的情况（不过C主要是避免重复定义），而C使用了#ifndef宏来条件包含。他其实也是定义了一个环境变量（preprocessor variable ，http://www.fredosaurus.com/notes-cpp/preprocessor/ifdef.html ）

    #ifndef MYHEADER_H
    #define MYHEADER_H
    . . . // This will be seen by the compiler only once 
    #endif /* MYHEADER_H */

我们可以采用类似的方式——export就相当于#define宏。

    if [ "$log" ]; then
        return
    fi
    export log="log.sh"

    . utils.sh

    ...

最终所有代码如下：

client shell定义如下：

    forrest@forrest-laptop:~/study/shell/svn_tools$ cat svncobranches.sh 
    #!/bin/bash

    # determine base directory; preserve where you're running from
    realpath=$(readlink -f "$0")
    export basedir=$(dirname "$realpath") #export basedir, so that module shell can use it. log.sh. e.g.
    export filename=$(basename "$realpath") #export filename, so that module shell can use it. log.sh. e.g.

    export PATH=$PATH:$basedir/modules

    . svn_core.sh

    svncobranch http://svn.alibaba-inc.com/repos/ali_sourcing/share/biz/product/branches/20110817_73701_2 /home/forrest/work/intl-biz/product
    
    forrest@forrest-laptop:~/study/shell/svn_tools$ cd modules/
    forrest@forrest-laptop:~/study/shell/svn_tools/modules$ cat svn_core.sh 
    if [ "$svn_core" ]; then
            return
    fi

    export svn_core="svn_core.sh"

    . utils.sh
    . log.sh

    ### svn上将指定的SVN URL co到本地指定的目录
    # svncobranch svnurl path 
    svncobranch()
    {
        local svnurl=$1
        local path=$2
        
        log "svncobranch($svnurl, $path)"
    }

    forrest@forrest-laptop:~/study/shell/svn_tools/modules$ cat log.sh 
    if [ "$log" ]; then
            return
    fi

    export log="log.sh"

    . utils.sh

    # log(meg, loglevel)
    log()
    {
        datetime=`date +"%y-%m-%d %H:%M:%S"`
        message=$1
        if [ -z "$2" ]
        then
            loglevel="INFO"
        else
            loglevel=$2
        fi
    
        outdir="$basedir/log"
        if [ ! -d "$outdir" ]; then
            mkdir "$outdir"
        fi
    
        
        logname=`removepostfix $filename`  
        echo "$datetime [$0] $loglevel :: $message" | tee -a "$outdir/$logname.log"
    }

    log_error()
    {
            log "$1" "ERROR"
    }

    log_info()
    {
            log "$1" "INFO"
    }

    log_debug()
    {
            log "$1" "DEGUG"
    }

    log_warn()
    {
            log "$1" "WARN"
    }

    forrest@forrest-laptop:~/study/shell/svn_tools/modules$ cat utils.sh 
    if [ "$utils" ]; then
    return
    fi

    export utils="utils.sh"

    . log.sh

    # trim(str)
    # remove blank space in both side
    trim()
    {
        echo $*
    }

    # remove the postfix
    # svncobranches.sh ==> svncobranches
    removepostfix()
    {
        filename=$(basename "$1") 
        echo "$filename" | sed 's/.sh$//'
    }

上面代码在笔者机器上全部测试过，应该没有问题。

---EOF---


