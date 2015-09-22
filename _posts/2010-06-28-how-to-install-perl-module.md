---
title: 如何安装perl模块
layout: post
---

2010－06－28 星期一 暴雨

运行perl脚本时，经常会发现如下类似的错误：

    forrest@forrest-desktop:~/study/perl/log4perl$ ./logToScreen.pl 
    Can't locate Log/Log4perl.pm in @INC (@INC contains: /etc/perl /usr/local/lib/perl/5.10.0 /usr/local/share/perl/5.10.0 /usr/lib/perl5 /usr/share/perl5 /usr/lib/perl/5.10 /usr/share/perl/5.10 /usr/local/lib/site_perl .) at ./logToScreen.pl line 3.
    BEGIN failed--compilation aborted at ./logToScreen.pl line 3.

这个是因为

    use Log::Log4perl;

Log::Log4perl模块没有安装。

    forrest@ubuntu:~/study/perl$ ./memcached.pl 
    Can't locate Cache/Memcached.pm in @INC (@INC contains: /etc/perl /usr/local/lib/perl/5.10.1 /usr/local/share/perl/5.10.1 /usr/lib/perl5 /usr/share/perl5 /usr/lib/perl/5.10 /usr/share/perl/5.10 /usr/local/lib/site_perl .) at ./memcached.pl line 5.
    BEGIN failed--compilation aborted at ./memcached.pl line 5.

这个是因为

    use Cache::Memcached;

Cache::Memcached模块没有安装。

### 解决方法

网上搜索了一下：发现有一篇文章写的非常好：[Perl(Pete's notes)](http://www.cisl.ucar.edu/nets/intro/staff/siemsen/tools/perl.html#log4perl)

> ### Installing Perl modules with CPAN.pm (best way)
>
> Use the CPAN.pm module. To read about it, do "perldoc CPAN", or in XEmacs use the Perldoc pull-down when you're editing a Perl file.
> 
> The first time you use CPAN.pm, it will ask a long series of questions, the answers for which can be found below. Don't answer them until you've installed ncftp on the local machine.
> 
> If you've already installed CPAN and just want to use it, do like
>   (as root)
>   (sudo) perl -MCPAN -e shell
>   install Log::Log4perl
>   install HTML::TokeParser::Simple
>   h
>   q
>
> The above will install Log4perl in /usr/lib/perl5/site_perl/5.6.1/Log/Log4perl.

需要注意的是必须使用root权限才能安装成功。

---

补记：2010-07-21 星期三 晴朗
--------------------------

今天翻看了一下《Learn Perl, 5th》，第十一章是Perl Modules，一开始就是介绍怎么安装perl模块的。感觉总结的非常好。

### 1. Fining Modules:

Modules come in two types: those that come with Perl and that you should have available to you, and those that you can get from CPAN to install yourself. 

To find modules that don’t come with Perl, start at either CPAN Search (http://search.cpan.org) or Kobes’ Search (http://kobesearch.cpan.org/).* You can browse through the categories or search directly.

**tips: 如何检查一个perl模块是否已经安装了？**

可以使用perldoc moduleName检查。
不过首先要现安装perl-doc

    $sudo apt-get install perl-doc
    $ perldoc CGI
    Try it with a module that does not exist and you’ll get an error message.
    $ perldoc Llamas
    $ No documentation found for "Llamas".

最佳实践：使用MCPAN的m moduleName命令

    cpan[1]> m DBI
    CPAN: Storable loaded ok (v2.20)
    Going to read '/home/forrest/.cpan/Metadata'
      Database was generated on Wed, 21 Jul 2010 07:35:04 GMT
    Module id = DBI
        DESCRIPTION  Generic Database Interface (see DBD modules)
        CPAN_USERID  TIMB (Tim Bunce <Tim.Bunce@pobox.com>)
        CPAN_VERSION 1.612
        CPAN_FILE    T/TI/TIMB/DBI-1.612.tar.gz
        UPLOAD_DATE  2010-07-16
        DSLIP_STATUS MmcOp (mature,mailing-list,C,object-oriented,Standard-Perl)
        MANPAGE      DBI - Database independent interface for Perl
        INST_FILE    /usr/local/lib/perl/5.10.1/DBI.pm
        INST_VERSION 1.612


    cpan[2]> m DBD::Oracle
    Module id = DBD::Oracle
        DESCRIPTION  Oracle Driver for DBI
        CPAN_USERID  DBIML (DBI Mailing Lists <dbi-users@perl.org>)
        CPAN_VERSION 1.24
        CPAN_FILE    P/PY/PYTHIAN/DBD-Oracle-1.24b.tar.gz
        DSLIP_STATUS MmcO? (mature,mailing-list,C,object-oriented,)
        INST_FILE    (not installed)

### 2. Installing Modules

三种方法：

法一：下载安装包手动安装

you can download the distribution, unpack it, and run a series of commands from the shell. Check for a README or INSTALL file that gives you more information. If the module uses MakeMaker,† the sequence will be something like this:
    $ perl Makefile.PL
    $ make install
If you can’t install modules in the system-wide directories, you can specify another directory with a PREFIX argument to Makefile.PL:
    $ perl Makefile.PL PREFIX=/Users/fred/lib
Some Perl module authors use another module, Module::Build, to build and install their creations. That sequence will be something like this:
    $ perl Build.PL
    $ ./Build install
缺点：无法自动安装依赖的包（Some modules depend on other modules though, and they won’t work unless you install yet more modules.）

法二：使用Perl自带的模块——CPAN.pm模块

$ perl -MCPAN -e shell

就是我们前面介绍的方法，这里就不赘述了。

法三：使用Perl自带的一个perl脚本——cpan脚本

$ cpan Module::CoreList LWP CGI::Prototype

**NOTES** 

为了使用法二和法三，你需要先按照perl-cpan。

    yum install perl-CPAN

**TIPS**

使用cpan安装默认是安装最新版本的，如果要指定版本，需要指定全路径，格式一般是 <作者名>/包名，如：`CAPTTOFU/DBD-mysql-3.0008.tar.gz`。


