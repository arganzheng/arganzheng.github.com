---
title: 使用pythonbrew和virtualenv构建Python沙箱环境
layout: post
---


一个非常好的python工具——virtualenv
--------------------------------

一直没有重视virtualenv，知道最近在MySQL备机上跑数据迁移脚本，发现要写一些python脚本，但是内建的python只有最根本的packages，import经常报错。想要手动安装，发现没有权限写入`/usr/lib/pythonX.X/site-packages`目录。而且，没有`easy_install`或者`pip`（安装这些同样需要root权限），安装python模块也挺费劲的。
Google了一下，发现有一了一个不错的工具，可以满足我的需求——建立一个独立的自由的python环境。

virtualenv本质上是一个python packages，所以可以使用easy_install或者pip安装，但是服务器上一般不会安装easy_install或者pip的，即使有，使用这些包管理工具默认也是安装到`/usr/local/lib/python2.6/dist-packages`这样的特殊权限目录。幸亏还有另外一种方式，就是直接运行virtualenv脚本安装方式。只要将jar包下载下来解压就可以直接执行了（其实知需要下载virtualenv.py就可以了）:

    [admin@ds-product-db1b /home/admin/temp]
    $wget http://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.6.1.tar.gz#md5=1a475df2219457b6b4febb9fe595d915
    ...
    2011-06-20 23:41:54 (78.6 KB/s) - `virtualenv-1.6.1.tar.gz' saved [1779120/1779120]
    [admin@ds-product-db1b /home/admin/temp]
    $tar zxvf virtualenv-1.6.1.tar.gz -C /home/admin/
    ...
    [admin@ds-product-db1b /home/admin]
    $cd virtualenv-1.6.1/
    [admin@ds-product-db1b /home/admin]
    $ln -s virtualenv-1.6.1/ virtualenv
    [admin@ds-product-db1b /home/admin]
    $cd virtualenv
    [admin@ds-product-db1b /home/admin/virtualenv]
    $ll
    total 160
    -rw-r--r-- 1 admin admin   318 Mar 20 12:46 AUTHORS.txt
    drwxr-xr-x 2 admin admin  4096 Jun 20 23:42 docs
    -rw-r--r-- 1 admin admin  1175 Mar 16 09:55 LICENSE.txt
    -rw-r--r-- 1 admin admin   280 Mar 16 09:56 MANIFEST.in
    -rw-r--r-- 1 admin admin 32949 Apr 30 15:44 PKG-INFO
    drwxr-xr-x 2 admin admin  4096 Jun 20 23:42 scripts
    -rw-r--r-- 1 admin admin    59 Apr 30 15:44 setup.cfg
    -rw-r--r-- 1 admin admin  2157 Apr 30 14:34 setup.py
    drwxr-xr-x 2 admin admin  4096 Jun 20 23:42 virtualenv.egg-info
    -rw-r--r-- 1 admin admin 84575 Apr 30 15:42 virtualenv.py
    drwxr-xr-x 2 admin admin  4096 Apr 30 15:44 virtualenv_support

    [admin@ds-product-db1b /home/admin/virtualenv]
    $python virtualenv.py ENV
    New python executable in ENV/bin/python
    Installing setuptools.............done.
    Installing pip...............done.
    [admin@ds-product-db1b /home/admin/virtualenv]
    $cd ENV/
    [admin@ds-product-db1b /home/admin/virtualenv/ENV]
    $ls
    bin  include  lib  lib64
    [admin@ds-product-db1b /home/admin/virtualenv/ENV]
    $which python
    /usr/bin/python

    [admin@ds-product-db1b /home/admin/virtualenv/ENV]
    $source bin/activate
    (ENV)
    [admin@ds-product-db1b /home/admin/virtualenv/ENV]
    $which python
    ~/virtualenv-1.6.1/ENV/bin/python
    (ENV)

这样就创建并且将当前的session设置为ENV python环境。
简单来说，上面就这几个命令完成：

    easy_install virtualenv
    virtualenv myvirtenv
    cd myvirtenv
    source bin/active

另外，我们看到在ENV/bin目录下顺带了easy_install和pip，我们可以用他们来安装python模块，它们会将这些模块安装在你们的虚拟环境中。这就不需要而外的权限了。

    [admin@ds-product-db1b /home/admin/virtualenv/ENV]
    $easy_install ipython
    Searching for ipython
    ...

酷吧！关键是实用啊！


What virtualenv cannot do!
--------------------------

不过看到其实virtualenv的安装还是依赖于python的，所以要求目标机器已经安装了共用的python解释器，一般的linux环境都应该是有的，所以这个不是问题，关键的问题是，virtualenv其实是一个python的运行虚拟环境（独立环境），而不是一个python本身的版本管理器（这点与rvm是最大的不同，rvm不只是ruby version manager，而且它附带的gem命令会安装ruby包到当前选择的ruby版本相应的库中）。为了解决这个问题，另一个工具应运而生了——它就是pythonblew！作者也承认，是受perlblew和rvm的启发而编写的。
他的安装和使用与rvm几乎同出一辙：

    (ENV)forrest@forrest-laptop:~$ curl -kL http://xrl.us/pythonbrewinstall | bash
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    103  2799  103  2799    0     0    797      0  0:00:03  0:00:03 --:--:--  9786
    Downloading http://pypi.python.org/packages/source/p/pythonbrew/pythonbrew-1.1.tar.gz
    ######################################################################## 100.0%
    Extracting /home/forrest/.pythonbrew/dists/pythonbrew-1.1.tar.gz
    Installing pythonbrew into /home/forrest/.pythonbrew

    Well-done! Congratulations!

    The pythonbrew is installed as:
        
      /home/forrest/.pythonbrew

    Please add the following line to the end of your ~/.bashrc

      source /home/forrest/.pythonbrew/etc/bashrc

    After that, exit this shell, start a new one, and install some fresh
    pythons:

      pythonbrew install 2.7.2
      pythonbrew install 3.2

    For further instructions, run:

      pythonbrew help

    The default help messages will popup and tell you what to do!

    Enjoy pythonbrew at /home/forrest/.pythonbrew!!

安装完成后有很多提示，按照做就可以了。接下来你就可以用他来安装你需要的python版本了：

    forrest@forrest-laptop:~$ pythonbrew list -k
    # Pythons
    Python-1.5.2
    Python-1.6.1
    Python-2.0.1
    Python-2.1.3
    Python-2.2.3
    Python-2.3.7
    Python-2.4.6
    Python-2.5.6
    Python-2.6.7
    Python-2.7.2
    Python-3.0.1
    Python-3.1.4
    Python-3.2.1
    forrest@forrest-laptop:~$ pythonbrew install 3.2
    Downloading Python-3.2.tgz as /home/forrest/.pythonbrew/dists/Python-3.2.tgz
    ######################################################################## 100.0%
    Extracting Python-3.2.tgz into /home/forrest/.pythonbrew/build/Python-3.2

    This could take a while. You can run the following command on another shell to track the status:
      tail -f /home/forrest/.pythonbrew/log/build.log

    Patching Python-3.2
    Installing Python-3.2 into /home/forrest/.pythonbrew/pythons/Python-3.2

然后你可以用这个版本来安装相应的virtualenv，从而为这个版本构建一个虚拟运行环境。pythonbrew本身有命令安装virtualenv，其实就是一个virtualenvWrapper:

    Create isolated python environments (uses virtualenv):

    pythonbrew venv init
    pythonbrew venv create proj
    pythonbrew venv list
    pythonbrew venv use proj
    pythonbrew venv delete proj

有了独立的自由的python环境，剩下来的事情只要你能想象到，没有python做不到的:)

如果有gcc的话，源码安装python还是蛮简单的。如果没有root权限，可以使用--prefix安装在home目录下：
>
  wget http://www.python.org/ftp/python/2.7.4/Python-2.7.4.tar.bz2
  tar -xjf Python-2.7.4.tar.bz2  
  cd Python-2.7.4  
>
Now read the README file to figure out how to install, or do the following with no guarantees from me that it will be exactly what you need.
>
  ./configure  
  make  
  sudo make install 

### tip: virtualenvwrapper 

> **Wrappers**
The wrappers provided by virtualenvwrapper (that I know of) are:
>
mkvirtualenv (create a new virtualenv)
rmvirtualenv (remove an existing virtualenv)
workon (change the current virtualenv)
add2virtualenv (add external packages in a .pth file to current virtualenv)
cdsitepackages (cd into the site-packages directory of current virtualenv)
cdvirtualenv (cd into the root of the current virtualenv)
deactivate (deactivate virtualenv, which calls several hooks)

说明：--no-site-packages Option

>If you build with virtualenv --no-site-packages ENV it will not inherit any packages from /usr/lib/python2.5/site-packages (or wherever your global site-packages directory is). This can be used if you don't have control over site-packages and don't want to depend on the packages there, or you just want more isolation from the global system.

virtualenv默认会将/usr/lib/pythonX.X/site-packages下的模块copy到你的虚拟python环境。使用这个选项可以避免这个行为。

思考：virtualenv其实要解决的问题是多工程依赖版本不一致问题，在Java中这是通过maven来解决的（maven的repository允许多个版本共存），但是在python中，每个python install的lib(site-packages)下只能有一个版本的模块存在（这个类似于debian的packages管理）。为了解决这个问题，才引出了多python install，因而可以各个install使用自己的lib(site-packages)——可以理解为多repository吧。这个解决方案不能说不好，但是感觉不是很优雅，毕竟需要安装多个python实例。而且要source bin/activtive来修改path值。感觉还是maven优雅一些，将版本用到极致。不过也正是这样，当我们对公个repository没有权限时，virtualenv提供的似有repository倒是给了我们机会。


**说明** 其实python的包安装挺恶心的。。如果你不能使用pip或者easy_install，源码安装你就痛苦死了。各种依赖问题。关于如何安装python package，并且各种安装方式的区别这篇文章写的很不错 So You Want to Install a Python Package 。强烈推荐阅读。


参考文章
-------

1. Tools of the Modern Python Hacker: Virtualenv, Fabric and Pip [http://www.clemesha.org/blog/modern-python-hacker-tools-virtualenv-fabric-pip]
2. Presentation: pip and virtualenv [http://mathematism.com/2009/07/30/presentation-pip-and-virtualenv/]
3. virtualenvwrapper [http://www.doughellmann.com/articles/pythonmagazine/completely-different/2008-05-virtualenvwrapper/index.html]
4. How to install and manage different versions of Python in Linux [http://www.howopensource.com/2011/05/how-to-install-and-manage-different-versions-of-python-in-linux/]
5. Using Pythonbrew and Virtualenv(with pip) for creating sandboxed Python development environments. [http://suvashthapaliya.com/blog/2012/01/sandboxed-python-virtual-environments/]



