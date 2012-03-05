---
layout: post
title: shell模块的另一种组织方式
---

在前面一篇文章里[shell如何模块化和复用——shell深入学习][shell module]，我介绍了一种shell的模块化做法，但是在实际的代码中我估计很少人会这么处理，相反，有另一种组织方式（或者称为模块化方式吧）在一些开源代码中可以见到，感觉更符合shell的语言特性。

[shell module]: shell-module.html

由于shell的source语句（. 语句）本质上是在当前的shell进程中执行script脚本文件，所以如果这个脚本文件中不是单纯的定义子函数被导入上调用（也就是我们所谓的子模块或者库），那么他就类似于VM中的一个子control，仅仅是为了避免重复代码，而且往往需要导入多次（因为导入就是为了执行该子shell）。而对于作为子模块的子shell文件，我们其实是希望他只被导入一次的（因为我们是要导入他定义的子函数）。这种方式其实在很多开源软件中很常见，因为本身shell其实就不是一个模块话的语言，虽然可以通过某些机制让他支持，但是成本高而且没有太大必要。相比之下，后者具有更大的可操作性。比如最简单，而且也是最的常见的做法，就是将一个函数的函数体抽出来，放在一个单独的shell脚本中，如果调用该脚本的话，那么就提交将参数设置好，然后source一下该脚本，这样就相当于调用了该子函数（有个缺点就是参数和返回值是通过约定好的变量进行）。

举个具体的实际例子，nginx的configure脚本就是这么处理的：在总入口configure shell脚本中，大量的调用（导入）在auto目录下的子shell函数（脚本）：

    forrest@ubuntu:~/Install/nginx-1.0.12$ tree auto/
    auto/
    ├── cc
    │   ├── acc
    │   ├── bcc
    │   ├── ccc
    │   ├── conf
    │   ├── gcc
    │   ├── icc
    │   ├── msvc
    │   ├── name
    │   ├── owc
    │   └── sunc
    ├── define
    ├── endianess
    ├── feature
    ├── have
    ├── have_headers
    ├── headers
    ├── include
    ├── init
    ├── install
    ├── lib
    │   ├── conf
    │   ├── geoip
    │   │   └── conf
    │   ├── google-perftools
    │   │   └── conf
    │   ├── libatomic
    │   │   ├── conf
    │   │   └── make
    │   ├── libgd
    │   │   └── conf
    │   ├── libxslt
    │   │   └── conf
    │   ├── make
    │   ├── md5
    │   │   ├── conf
    │   │   ├── make
    │   │   ├── makefile.bcc
    │   │   ├── makefile.msvc
    │   │   └── makefile.owc
    │   ├── openssl
    │   │   ├── conf
    │   │   ├── make
    │   │   ├── makefile.bcc
    │   │   └── makefile.msvc
    │   ├── pcre
    │   │   ├── conf
    │   │   ├── make
    │   │   ├── makefile.bcc
    │   │   ├── makefile.msvc
    │   │   └── makefile.owc
    │   ├── perl
    │   │   ├── conf
    │   │   └── make
    │   ├── sha1
    │   │   ├── conf
    │   │   ├── make
    │   │   ├── makefile.bcc
    │   │   ├── makefile.msvc
    │   │   └── makefile.owc
    │   ├── test
    │   └── zlib
    │       ├── conf
    │       ├── make
    │       ├── makefile.bcc
    │       ├── makefile.msvc
    │       ├── makefile.owc
    │       └── patch.zlib.h
    ├── make
    ├── modules
    ├── nohave
    ├── options
    ├── os
    │   ├── conf
    │   ├── darwin
    │   ├── freebsd
    │   ├── linux
    │   ├── solaris
    │   └── win32
    ├── sources
    ├── stubs
    ├── summary
    ├── types
    │   ├── sizeof
    │   ├── typedef
    │   ├── uintptr_t
    │   └── value
    └── unix

如其中的. auto/feature在多个shell中都被导入过（而且有些还不只一次，这取决于调用的次数）：
有95个地方调用到了：

    forrest@ubuntu:~/study/nginx-1.0.12$ find . | xargs grep -Hn '. auto/feature' 
    ./auto/unix:42:. auto/feature
    ./auto/unix:61:. auto/feature
    ./auto/unix:78:    . auto/feature
    ./auto/unix:95:        . auto/feature
    ./auto/unix:125:        . auto/feature
    ./auto/unix:157:. auto/feature
    ./auto/unix:168:    . auto/feature
    ./auto/unix:183:. auto/feature
    ./auto/unix:193:. auto/feature
    ./auto/unix:203:. auto/feature
    ./auto/unix:217:. auto/feature
    ./auto/unix:228:. auto/feature
    ./auto/unix:241:. auto/feature
    ./auto/unix:253:. auto/feature
    ./auto/unix:263:. auto/feature
    ./auto/unix:270:    . auto/feature
    ./auto/unix:285:. auto/feature
    ./auto/unix:292:    . auto/feature
    ./auto/unix:307:. auto/feature
    ./auto/unix:317:. auto/feature
    ./auto/unix:329:. auto/feature
    ./auto/unix:339:. auto/feature
    ./auto/unix:352:    . auto/feature
    ./auto/unix:371:        . auto/feature
    ./auto/unix:463:    . auto/feature
    ./auto/unix:474:. auto/feature
    ./auto/unix:484:. auto/feature
    ./auto/unix:494:. auto/feature
    ./auto/unix:505:. auto/feature
    ./auto/unix:519:    . auto/feature
    ./auto/unix:547:    . auto/feature
    ./auto/unix:558:. auto/feature
    ./auto/unix:568:. auto/feature
    ./auto/unix:578:. auto/feature
    ./auto/unix:591:. auto/feature
    ./auto/unix:606:. auto/feature
    ./auto/unix:620:. auto/feature
    ./auto/unix:632:. auto/feature
    ./auto/unix:640:    . auto/feature
    ./auto/unix:653:    . auto/feature
    ./auto/unix:668:. auto/feature
    ./auto/unix:679:. auto/feature
    ./auto/unix:689:. auto/feature
    ./auto/unix:699:. auto/feature
    ./auto/unix:709:. auto/feature
    ./auto/lib/openssl/conf:52:        . auto/feature
    ./auto/lib/geoip/conf:13:    . auto/feature
    ./auto/lib/geoip/conf:28:    . auto/feature
    ./auto/lib/geoip/conf:45:    . auto/feature
    ./auto/lib/geoip/conf:62:    . auto/feature
    ./auto/lib/libgd/conf:13:    . auto/feature
    ./auto/lib/libgd/conf:29:    . auto/feature
    ./auto/lib/libgd/conf:46:    . auto/feature
    ./auto/lib/libgd/conf:63:    . auto/feature
    ./auto/lib/sha1/conf:52:        . auto/feature
    ./auto/lib/sha1/conf:63:            . auto/feature
    ./auto/lib/md5/conf:62:            . auto/feature
    ./auto/lib/md5/conf:72:            . auto/feature
    ./auto/lib/md5/conf:86:            . auto/feature
    ./auto/lib/google-perftools/conf:13:    . auto/feature
    ./auto/lib/google-perftools/conf:28:    . auto/feature
    ./auto/lib/libatomic/conf:30:    . auto/feature
    ./auto/lib/pcre/conf:98:        . auto/feature
    ./auto/lib/pcre/conf:113:            . auto/feature
    ./auto/lib/pcre/conf:124:            . auto/feature
    ./auto/lib/pcre/conf:140:            . auto/feature
    ./auto/lib/pcre/conf:156:            . auto/feature
    ./auto/lib/zlib/conf:54:        . auto/feature
    ./auto/lib/libxslt/conf:22:    . auto/feature
    ./auto/lib/libxslt/conf:38:    . auto/feature
    ./auto/lib/libxslt/conf:55:    . auto/feature
    ./auto/lib/libxslt/conf:72:    . auto/feature
    ./auto/lib/libxslt/conf:101:    . auto/feature
    ./auto/lib/libxslt/conf:116:    . auto/feature
    ./auto/lib/libxslt/conf:133:    . auto/feature
    ./auto/lib/libxslt/conf:150:    . auto/feature
    ./auto/os/linux:61:. auto/feature
    ./auto/os/linux:85:. auto/feature
    ./auto/os/linux:106:. auto/feature
    ./auto/os/linux:120:. auto/feature
    ./auto/os/linux:133:. auto/feature
    ./auto/os/linux:146:. auto/feature
    ./auto/os/solaris:39:. auto/feature
    ./auto/os/solaris:55:. auto/feature
    ./auto/os/darwin:54:. auto/feature
    ./auto/os/darwin:81:. auto/feature
    ./auto/os/darwin:100:. auto/feature
    ./auto/os/darwin:116:. auto/feature
    ./auto/cc/gcc:30:. auto/feature
    ./auto/cc/name:15:    . auto/feature
    ./auto/cc/conf:120:        . auto/feature
    ./auto/cc/conf:144:    . auto/feature
    ./auto/cc/conf:160:        . auto/feature
    ./auto/cc/conf:174:    . auto/feature
    ./auto/cc/conf:184:#    . auto/feature
    forrest@ubuntu:~/study/nginx-1.0.12$ find . | xargs grep '. auto/feature' | wc -l
    95

有如下17个文件使用到了：

    forrest@ubuntu:~/study/nginx-1.0.12$ find . | xargs grep -l '. auto/feature' 
    ./auto/unix
    ./auto/lib/openssl/conf
    ./auto/lib/geoip/conf
    ./auto/lib/libgd/conf
    ./auto/lib/sha1/conf
    ./auto/lib/md5/conf
    ./auto/lib/google-perftools/conf
    ./auto/lib/libatomic/conf
    ./auto/lib/pcre/conf
    ./auto/lib/zlib/conf
    ./auto/lib/libxslt/conf
    ./auto/os/linux
    ./auto/os/solaris
    ./auto/os/darwin
    ./auto/cc/gcc
    ./auto/cc/name
    ./auto/cc/conf
    forrest@ubuntu:~/study/nginx-1.0.12$ find . | xargs grep -l '. auto/feature' | wc -l
    17

那么auto/feature里面到底定义了什么了：

    forrest@ubuntu:~/Install/nginx-1.0.12$ cat auto/feature
    # Copyright (C) Igor Sysoev
    # Copyright (C) Nginx, Inc.


    echo $ngx_n "checking for $ngx_feature ...$ngx_c"

    cat << END >> $NGX_AUTOCONF_ERR

    ----------------------------------------
    checking for $ngx_feature

    END

    ngx_found=no

    if test -n "$ngx_feature_name"; then
        ngx_have_feature=`echo $ngx_feature_name \
                    | tr abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ`
    fi

    if test -n "$ngx_feature_path"; then
        for ngx_temp in $ngx_feature_path; do
            ngx_feature_inc_path="$ngx_feature_inc_path -I $ngx_temp"
        done
    fi

    cat << END > $NGX_AUTOTEST.c

    #include <sys/types.h>
    $NGX_INCLUDE_UNISTD_H
    $ngx_feature_incs

    int main() {
        $ngx_feature_test;
        return 0;
    }

    END


    ngx_test="$CC $CC_TEST_FLAGS $CC_AUX_FLAGS $ngx_feature_inc_path \
            -o $NGX_AUTOTEST $NGX_AUTOTEST.c $NGX_TEST_LD_OPT $ngx_feature_libs"

    ngx_feature_inc_path=

    eval "/bin/sh -c \"$ngx_test\" >> $NGX_AUTOCONF_ERR 2>&1"


    if [ -x $NGX_AUTOTEST ]; then

        case "$ngx_feature_run" in

            yes)
                # /bin/sh is used to intercept "Killed" or "Abort trap" messages
                if /bin/sh -c $NGX_AUTOTEST >> $NGX_AUTOCONF_ERR 2>&1; then
                    echo " found"
                    ngx_found=yes

                    if test -n "$ngx_feature_name"; then
                        have=$ngx_have_feature . auto/have
                    fi

                else
                    echo " found but is not working"
                fi
            ;;

            value)
                # /bin/sh is used to intercept "Killed" or "Abort trap" messages
                if /bin/sh -c $NGX_AUTOTEST >> $NGX_AUTOCONF_ERR 2>&1; then
                    echo " found"
                    ngx_found=yes

                    cat << END >> $NGX_AUTO_CONFIG_H

                    #ifndef $ngx_feature_name
                    #define $ngx_feature_name  `$NGX_AUTOTEST`
                    #endif

                    END
                else
                    echo " found but is not working"
                fi
            ;;

            bug)
                # /bin/sh is used to intercept "Killed" or "Abort trap" messages
                if /bin/sh -c $NGX_AUTOTEST >> $NGX_AUTOCONF_ERR 2>&1; then
                    echo " not found"

                else
                    echo " found"
                    ngx_found=yes

                    if test -n "$ngx_feature_name"; then
                        have=$ngx_have_feature . auto/have
                    fi
                fi
            ;;

            *)
                echo " found"
                ngx_found=yes

                if test -n "$ngx_feature_name"; then
                    have=$ngx_have_feature . auto/have
                fi
            ;;

        esac

    else
        echo " not found"

        echo "----------"    >> $NGX_AUTOCONF_ERR
        cat $NGX_AUTOTEST.c  >> $NGX_AUTOCONF_ERR
        echo "----------"    >> $NGX_AUTOCONF_ERR
        echo $ngx_test       >> $NGX_AUTOCONF_ERR
        echo "----------"    >> $NGX_AUTOCONF_ERR
    fi

    rm $NGX_AUTOTEST*

我们看到，它其实是根据传递过来的变量，生成一个测试$NGX_AUTOTEST.c文件，然后调用CC编译器编译看是否通过（设置ngx_found变量），如果设置了$ngx_feature_run为yes，那么还会执行编译后的测试文件。如果通过说明当前操作系统支持这个feature，否则就是不支持。

同样，auto/have也被调用（导入）了多次。

---EOF---

