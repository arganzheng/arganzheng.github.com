---
title: vim-插件篇
layout: post
catalog: true
tags: [vim, plugin, linux]
---


插件管理
-------

你第一个要安装的插件是管理插件的插件——vundle。

vim 自身希望通过在 .vim/ 目录中预定义子目录管理所有插件（比如，子目录 doc/ 存放插件帮助文档、plugin/ 存放通用插件脚本），vim 的各插件打包文档中通常也包含上述两个（甚至更多）子目录，用户将插件打包文档中的对应子目录拷贝至 .vim/ 目录即可完成插件的安装。一般情况下这种方式没问题，但我等重度插件用户，.vim/ 将变得混乱不堪，至少存在如下几个问题：

1. 插件名字冲突。所有插件的帮助文档都在 doc/ 子目录、插件脚本都在 plugin/ 子目录，同个名字空间下必然引发名字冲突；
2. 插件卸载易误。你需要先知道 doc/ 和 plugin/ 子目录下哪些文件是属于该插件的，再逐一删除，容易多删/漏删。

我们希望每个插件在 .vim/ 下都有各自独立子目录，这样需要升级、卸载插件时，直接找到对应插件目录变更即可；另外，我们希望所有插件清单能在某个配置文件中集中罗列，通过某种机制实现批量自动安装/更新/升级所有插件。vundle（https://github.com/VundleVim/Vundle.vim ）为此而生，它让管理插件变得更清晰、智能。

vundle 会接管 .vim/ 下的所有原生目录，所以先清空该目录，再通过如下命令安装 vundle：

	git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim

接下来在 .vimrc 增加相关配置信息：

	" vundle 环境设置
	filetype off
	set rtp+=~/.vim/bundle/Vundle.vim
	" vundle 管理的插件列表必须位于 vundle#begin() 和 vundle#end() 之间
	call vundle#begin()
	Plugin 'VundleVim/Vundle.vim'
	...
	" 插件列表结束
	call vundle#end()
	filetype plugin indent on

其中，每项

	Plugin 'VundleVim/Vundle.vim'

对应一个插件（这与 go 语言管理不同代码库的机制类似），后续若有新增插件，只需追加至该列表中即可。vundle 支持源码托管在 `https://github.com/` 的插件，同时 vim 官网 `http://www.vim.org/` 上的所有插件均在 `https://github.com/vim-scripts/` 有镜像，所以，基本上主流插件都可以纳入 vundle 管理。例如 VundleVim/Vundle.vim，它在 .vimrc 中配置信息为 VundleVim/Vundle.vim，vundle 很容易构造出其真实下载地址 `https://github.com/VundleVim/Vundle.vim.git·，然后借助 git 工具进行下载及安装。

此后，需要安装插件，先找到其在 github.com 的地址，再将配置信息其加入 .vimrc 中的call vundle#begin() 和 call vundle#end() 之间，最后进入 vim 执行

	:PluginInstall

便可通知 vundle 自动安装该插件及其帮助文档。

要卸载插件，先在 .vimrc 中注释或者删除对应插件配置信息，然后在 vim 中执行

	:PluginClean

即可删除对应插件。插件更新频率较高，差不多每隔一个月你应该看看哪些插件有推出新版本，批量更新，只需执行

	:PluginUpdate
	
即可。

通过 vundle 管理插件后，切勿通过发行套件自带的软件管理工具安装任何插件，不然 .vim/ 又要混乱了。


**TIPS** vundle之前是pathogen，不过现在已经out了，建议迁移到vundle来。


文件导航/搜索
-----------

### 目录导航 nerdtree


### 标签导航 tagbar


### 搜索文件 ctrlp.vim


### 搜索代码 ctrlsf.vim


快速移动
-------

### 行/位置/搜索 vim-easymotion

### 可视化标签 vim-signature


自动补全与代码片段
----------------

###YCM 毫秒级补全/ python / c系等

### ultisnips
　
### vim-snippets

括号补全

delimimate
xml/html标签补全

closetag


插件3: 快速编码
-------------

快速注释:

nerdcommenter
快速编辑

vim-surround
vim-repeat
去空格

vim-trailing-whitespace
代码对齐

vim-easy-align


这里的有用因人而异。对于笔者来说，主要是作为一个C/C++的IDE来使用。

* [NERDtree](https://github.com/scrooloose/nerdtree): 工程目录浏览
* [vim-cpp-enhanced-highlight](https://github.com/octol/vim-cpp-enhanced-highlight): C++代码高亮
* [vim-fswitch](https://github.com/derekwyatt/vim-fswitch): *.cpp 和 *.h 间切换
	* nmap <silent> <Leader>sw :FSHere<cr>
* [NERD Commenter](https://github.com/scrooloose/nerdcommenter): 快速开关注释
	* <leader>cc: 注释当前选中文本，如果选中的是整行则在每行首添加 //，如果选中一行的部分内容则在选中部分前后添加分别 /、/；
	* <leader>cu: 取消选中文本块的注释。
* ctrlp: 快速文件模糊查找 Fuzzy file, buffer, mru, tag, etc finder.
* tagbar: 代码outline(taglist的升级版)
* [matchIt](http://www.wklken.me/posts/2015/06/07/vim-plugin-matchit.html)
* FuzzyFinder（依赖于L9 library，另一个vim插件）。很多人推荐（Command-t，但是后者依赖ruby，恶心的是要求vim依赖ruby。。而且要用该vim编译所有的ruby版本来编译Command-t插件。。）
* Snipmate
* YCM: 史上最强大的智能补全
* omni-completion with tags(cscope) and TagList(tagbar)
* conque_term
* VOoM
* a.vim


