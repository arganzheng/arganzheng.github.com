---
title: 卓有成效的程序员——Mac篇
layout: post
catalog: true
tags: [productivity]
---


作为一个资深程序猿，在linux、windows和mac下都进行过相当长的一段开发时间，是时候总结一下mac下的一些高效工作Tips了。


离线文档
-------

有时候有些文档需要翻墙或者访问很慢，你可以将整个文档离线下载下来，在本地阅读。

登录能翻墙的服务器，使用wget的mirror功能下载整个网站：

	wget -m --no-check-certificate --convert-links https://www.graphengine.io/

**NOTES**

* 因为我们要下载的网站是https的，所以要增加`--no-check-certificate`参数，否则会报错。
* convert-links可以把网页的链接转成相对路径，如`/img/overview.png`会被转成`img/overview.png'，这样就不会出现资源加载不到了。

然后压缩和下载：

	tar -czf graphengine.tar.gz www.graphengine.io/
	sz -be graphengine.tar.gz

本地解压：
	
	tar -xzvf graphengine.tar.gz

就可以本地离线翻阅了。


Mac下的apt-get —— Homebrew & Homebrew-Cask
------------------------------------------

第一个要按照的工具是[Homebrew](http://brew.sh/)。

还有它的好兄弟[Homebrew-Cask](https://caskroom.github.io)：

	$ brew tap caskroom/cask

用他们可以很方便的安装其他的工具和软件。


终端工具——iTerm2
---------------

用mac开发的话，终端和命令行是每天都要面对的东东，一个好用的终端和shell必不可少。

终端这里推荐[iTerm2](http://www.iterm2.com)，除了配色好看之外，还具有一些不错的特性：

1. 分屏(Split Panes)
	* command + d : 左右分屏
	* command + shift + d : 上下分屏
	* command + shift + 方向键 : 切换屏幕
2. 自动补全(Autocomplete)
	* command + ; : 自动补全历史记录
3. 调出历史命令(Poste History)：
	* command + shift + h : 调出剪贴板历史，忘记某个命令的时候非常好用
4. 查找和拷贝(Find and Paste)
	* command + f : 查找，然后用 tab 和 shift + tab 可以向右和向左补全，补全之后的内容会被自动复制
	* 还可以用 alt + enter 将查找结果直接拷贝到终端

窗口管理（这几个命令是通用的）：

1. command + n : 新建窗口
2. command + t : 新建标签页
3. command + w : 关闭当前页
4. command + 方向键/数字 : 切换标签页
5. command + enter : 切换全屏
6. command + r 或者 control + l : 清屏

行内光标快速移动（跟Emace保持一致）：

1. control + a : 移动到行首
2. control + e : 移动到行尾
3. control + k : 删除从光标到行尾的内容


终极shell—— zsh + on my zsh
---------------------------

mac系统自带了zsh:

	arganzhengs-MacBook-Pro:~ argan$ cat /etc/shells
	# List of acceptable shells for chpass(1).
	# Ftpd will not allow users to connect who are not using
	# one of these shells.

	/bin/bash
	/bin/csh
	/bin/ksh
	/bin/sh
	/bin/tcsh
	/bin/zsh

但是一般不是最新版:

	arganzhengs-MacBook-Pro:~ argan$ zsh --version
	zsh 5.0.8 (x86_64-apple-darwin15.0)

可通过 Homebrew 来安装最新版：

	$ brew install zsh

mac默认是使用bash:
	$ echo $SHELL
	/bin/bash

可以使用如下命令将当前用户的默认shell改成zsh:

	$  chsh -s /usr/local/bin/zsh
	Changing shell for argan.
	Password for argan:
	chsh: /usr/local/bin/zsh: non-standard shell

出错了，因为 /usr/local/bin/zsh 不在 /etc/shells 文件中，要把它加进去，或者直接使用/bin/zsh就可以了。

zsh功能强大是公认的，但是由于配置过于复杂，初期无人问津，直到有一天，国外有个穷极无聊的程序员开发出了一个能够让你快速上手的zsh项目，叫做[oh my zsh](https://github.com/robbyrussell/oh-my-zsh)，一下子讲zsh的配置简化了很多。

安装就一条简单的命令搞定：

	$ sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
	Cloning Oh My Zsh...
	Cloning into '/Users/argan/.oh-my-zsh'...
	remote: Counting objects: 823, done.
	remote: Compressing objects: 100% (693/693), done.
	remote: Total 823 (delta 20), reused 683 (delta 9), pack-reused 0
	Receiving objects: 100% (823/823), 559.90 KiB | 184.00 KiB/s, done.
	Resolving deltas: 100% (20/20), done.
	Checking connectivity... done.
	Looking for an existing zsh config...
	Found ~/.zshrc. Backing up to ~/.zshrc.pre-oh-my-zsh
	Using the Oh My Zsh template file and adding it to ~/.zshrc
	Time to change your default shell to zsh!
	Changing shell for argan.
	Password for argan:
	chsh: no changes made
	         __                                     __
	  ____  / /_     ____ ___  __  __   ____  _____/ /_
	 / __ \/ __ \   / __ `__ \/ / / /  /_  / / ___/ __ \
	/ /_/ / / / /  / / / / / / /_/ /    / /_(__  ) / / /
	\____/_/ /_/  /_/ /_/ /_/\__, /    /___/____/_/ /_/
	                        /____/                       ....is now installed!


	Please look over the ~/.zshrc file to select plugins, themes, and options.

	p.s. Follow us at https://twitter.com/ohmyzsh.

	p.p.s. Get stickers and t-shirts at http://shop.planetargon.com.

	➜  arganzheng.github.com git:(master) ✗

按照成功还会自动给你切换默认的shell，真是贴心。

zsh有很多不错的功能，比如: kill <tab> 会自动列出进程，或者 kill nginx。

不过 oh my zsh 项目更强大的地方在于它提供了完善的插件体系。相关的文件在~/.oh-my-zsh/plugins目录下，默认提供了100多种，大家可以根据自己的实际学习和工作环境采用，想了解每个插件的功能，只要打开相关目录下的 zsh 文件看一下就知道了。

插件也是在 .zshrc 里配置，找到plugins关键字，你就可以加载自己的插件了。系统默认只是启用了 git，你可以在后面追加内容，例如启用插件 git, autojumps osx, 需要在配置文件中加入如下内容：

	plugins=(git autojump osx)

* osx 插件
	* man-preview 通过 preview 程序查看一个命令的手册
	* quick-look 快速预览文件
	* pfd 返回当前 Finder 打开的文件夹的路径
	* cdf 切换到当前 Finder 所在的目录
* autojump 插件
	* j 目录名（支持模糊匹配和自动补全）: 快速跳转到目录
	* d : 列出当前会话中访问过的目录列表，输入列表前的序号可以直接跳转
	* ..（跳转到父目录）
	* ... （跳转到父目录的父目录）
	* 直接输入目录名即可跳转（省略了cd命令）

**说明** autojump插件需要先安装才能使用：

	➜  ~ brew install autojump
	==> Downloading https://homebrew.bintray.com/bottles/autojump-22.3.0.el_capitan.bottle.tar.gz
	######################################################################## 100.0%
	==> Pouring autojump-22.3.0.el_capitan.bottle.tar.gz
	==> Caveats
	Add the following line to your ~/.bash_profile or ~/.zshrc file (and remember
	to source the file to update your current session):
	  [[ -s $(brew --prefix)/etc/profile.d/autojump.sh ]] && . $(brew --prefix)/etc/profile.d/autojump.sh

	If you use the Fish shell then add the following line to your ~/.config/fish/config.fish:
	  [ -f /usr/local/share/autojump/autojump.fish ]; and . /usr/local/share/autojump/autojump.fish

	zsh completion has been installed to:
	  /usr/local/share/zsh/site-functions
	==> Summary
	🍺  /usr/local/Cellar/autojump/22.3.0: 17 files, 163.8K

记得按照安装后的提示，把下面的配置信息加入~/.zshrc中：

	[[ -s $(brew --prefix)/etc/profile.d/autojump.sh ]] && . $(brew --prefix)/etc/profile.d/autojump.sh


其他的常用软件
------------

1. Alfred: 绝对神器，类似于启动器，方便启动各种App。
2. Spectacle: 窗口管理工具，在Windows上可以非常方便的将窗口居左居右居上居下，而这个工具就是通过快捷键在Mac上实现类似的功能，简单使用。
3. Qbserve: 时间跟踪记录软件
4. Github或者[sourcetree](https://www.sourcetreeapp.com/): 免费的Git图像化管理工具



