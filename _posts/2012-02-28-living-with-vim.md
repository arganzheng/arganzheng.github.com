---
layout: post
title: 工欲善其事，必先利其器——从零打造你的vim
---


内功心法
-------

### 预练此功，必先自宫。

#### 1、源码安装编辑器 vim

发行套件的软件源中预编译的vim很多时候版本太老旧，但是有些插件，比如YCM插件，对VIM版本有很严格的要求，所以如果你的版本实在太低，还是建议先升级一下。可以采用源码安装：

	git clone git@github.com:vim/vim.git
	cd vim/
	./configure --with-features=huge --enable-pythoninterp --enable-rubyinterp --enable-luainterp --enable-perlinterp --with-python-config-dir=/usr/lib/python2.7/config/ --enable-gui=gtk2 --enable-cscope --prefix=/usr
	make
	make install

其中，--enable-pythoninterp、--enable-rubyinterp、--enable-perlinterp、--enable-luainterp 等分别表示支持 ruby、python、perl、lua 编写的插件，--enable-gui=gtk2 表示生成采用 GNOME2 风格的 gvim，--enable-cscope 支持 cscope，--with-python-config-dir=/usr/lib/python2.7/config/ 指定 python 路径（先自行安装 python 的头文件 python-devel），这几个特性非常重要，影响后面各类插件的使用。注意，你得预先安装相关依赖库的头文件，python-devel、python3-devel、ruby-devel、lua-devel、libX11-devel、gtk-devel、gtk2-devel、gtk3-devel、ncurses-devel，如果缺失，源码构建过程虽不会报错，但最终生成的 vim 很可能缺失某些功能。构建完成后在 vim 中执行

	:echo has('python')

若输出 1 则表示构建出的 vim 已支持 python，反之，0 则不支持。

#### 2、从零开始配置vim

Put the pass behind you before you can move on. so, 让我们抛弃过去，从0开始。首先备份一下你.vim目录和.vimrc文件：

	$ cd
	$ mv .vim vim_old
	$ mv .vimrc vim_old/vimrc_old
	$ mkdir .vim
	$ vim .vimrc

加入如下最几本的配置：

	$ set nocompatible " 不要支持vi模式
	$ syntax on " 支持语法高亮
	$ filetype plugin indent on " 加载插件和支持缩进

这样，一个最基本的vim已经配置完毕。

### 重新映射关键的vim常用键

这个其实非常重要，但是很少有人会告诉你你应该这么做，而且应该怎么做最好。大部分都停留在各种快捷键的说明上，但是事实上这些快捷键未必好按。越是常做的事情，越值得优化。现在就开始吧：

#### Ctrl键

Ctrl键一般在键盘的左下角，一般于Fn键相邻。这个键相对键盘的home row来说还是相对远了点，所以有vim的使用者嘲笑emacs是“用到小手指抽筋的编辑器”，可见这个键其实并不是很好按。而且最让我崩溃的是，不同的笔记本对Fn键和Ctrl键的位置经常不一样，导致经常按错。但是其实vim很多的快捷键也需要用到Ctrl键，所以为了避免五十步笑一百步，我们最好将它映射在一个好按的地方——Caps Lock就是这样的一个最佳选择。这个键就像人体的阑尾一样，基本不需要，而它又处于home row位置。我的建议是在系统级别将他作掉。方法如下：

法一： Using a directive in /etc/X11/xorg.conf:

In your Section "InputDevice" section for the keyboard, add the line:

	$ Option "XkbOptions" "ctrl:nocaps"

法二： Using an xmodmap:

Create a file in your homedir called .Xmodmap with the contents:

	$ remove Lock = Caps_Lock
	$ remove Control = Control_L
	$ keysym Caps_Lock = Control_L
	$ add Control = Control_L

#### Esc键

这个键就更扯淡了，经常要用，但是却远在天边。它跟Ctrl在键盘上的布局一般是天南地北，但是相对于home row比Ctrl还要远的多，属于想抽筋都没办法的键。因此更有必要映射这个键了。但是映射到哪里好呢？
首先需要了解一下Esc键的用处，他的作用是从其他模式（如insert mode、visual mode）返回到命令模式（nomal mode），因此在非命令模式下的不常用键就是候选者。

一般有两种选择：

法一：映射成"jj"，因为在英语中基本没有单词是以jj开头的（好吧，我承认我邪恶了一下:)）。

法二：其实vim本身就意识到这个问题了，所以它默认将Esc键映射为`Ctrl-[`（这个组合在nomal模式下被tag给用了，但是在其他的模式下并没有什么作用）。由于我们前面已经将Ctrl键映射了，所以，`Ctrl-[`就非常好按了（当然比起jj还是要复杂一点，但是不需要做任何事情哦）。我的建议是两者都做（其实法二是内建的，不需要我们做任何事情）。

#### `<Leader>`键：

其实`<Leader>`键并不是固定的键盘上的一个键，它是vim的一个标识键，默认是转义符`'\'`，这个键也比较远，所以最好重新定义它。这个几乎没有争议——`','`是大家公认的替代者。

在你的.vimrc中增加这么一行配置项：

	" Set the value used for <leader> in mappings
	let g:mapleader = ","


但是个人觉得';'也是个不错的选择，因为他比','更近，而且几乎也没有其他命令抢占它。最重要的是，vimperator就是用';'，这样两者可以统一起来。不过这个大家自己看着办吧，都成。

','键也不是很好按，但是相对来说比'\'要好得多。让`<Leader>`键好按之后就可以好好利用它了（就像我们前面映射Ctrl键一样），可以用它来开启/关闭（toggle）不同的插件。这样的效果类似于vim的模式切换一样，相同的键就可以复用了。

> Keybindings are great, but there's so many already in use that it's hard not to overwrite existing ones. Using the <Leader> key to define keybindings has been great. As examples, I mapped <Leader>m to toggle tasks, and <Leader><Tab> to invoke snipMate.


#### 映射ex模式切换键(可选)：

在正常模式下跳转到ex模式需要按':'，可以映射为';'，这样就一只手搞定了。不过这个提升不多，大家也看着办吧:)

	" Similarly, : takes two keystrokes, ; takes one; map the latter to the former
	 " in normal mode to get to the commandline faster
	 nnoremap ; :


配置好vimrc
----------

在用最基本的vimrc配置之后，在确定自己已经熟悉vim原生的快捷键之后，我们考虑优化一下它的配置文件。经过多方参考和实践，我最终的vimrc配置如下（没有插件的配置，插件配置见下面）：

see "arganzheng/myvim-settings":https://github.com/arganzheng/myvim-settings/blob/master/vimrc

经过我的实践，这些配置非常好用，可以大大提高你的vim体验。


插件篇
-----

插件就像武功招式，如果太早学习，而忽略了最基本的vim命令，其实是本末倒置的。只有将基本的操作都熟悉之后，才考虑利用一些插件来提供你的工作效率。

下面是一些笔者认为比较好用的插件及简单的安装说明。使用说明网上有很多，这里就不赘述了，但是一些比较好的配置或者小技巧会稍微点一下。


### 插件管理

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


### 界面美化

系统默认的VIM有时候会比较简陋，可以安装主题风格美化一下。个人尝试了几个配色方案之后，个人比较喜欢下面这三个主题：

	"" 主题风格
	Plugin 'tomasr/molokai' " 多彩 molokai
	Plugin 'acarapetis/vim-colors-github' " Github风格(只有light模式)
	Plugin 'vim-scripts/peaksea'

	"" 配色方案
	set background=dark
	"colorscheme molokai
	colorscheme peaksea
	"colorscheme github

### 其他有用的插件

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


一些非常有用的TIPS
----------------

下面是笔者在工作中发现的一些非常实用的TIPS，希望对你也有帮助。

### 1、代码折叠

笔者一直都不觉得代码折叠是很重要的东东，直到我遇到了凤巢。每个cpp文件都是上万行代码的，很多函数都是几千行代码。。这时候才发现代码折叠还是非常有用的。

vim 自身支持多种折叠：手动建立折叠（manual）、基于缩进进行折叠（indent）、基于语法进行折叠（syntax）、未更改文本构成折叠（diff）等等，其中，indent、syntax 比较适合编程，按需选用。增加如下配置信息：

	" 基于缩进或语法进行代码折叠
	"set foldmethod=indent
	set foldmethod=syntax
	" 启动 vim 时关闭折叠代码
	set nofoldenable

操作：za，打开或关闭当前折叠；zM，关闭所有折叠；zR，打开所有折叠。

### 2、匹配符号间跳转

vim的 % , 会自动跳转到匹配的()[]{}<>等。再按照matchit插件，可以增强这个功能，支持标签之间的跳转，如<html>和</html>。

### 3、全匹配搜索当前词

vim的 * ，会自动精确搜索当前词。

参考文章与推荐阅读
---------------

1. [zmievski](http://zmievski.org/2007/02/vim-for-php-programmers-slides-and-resources)
2. [Coming Home to Vim](http://stevelosh.com/blog/2010/09/coming-home-to-vim)
3. [Kevin McCarthy](http://www.8t8.us/)
4. [mwop](http://ggmwop.net/blog/249-Vim-Toolbox,-2010-Edition)
5. [Mir Nazim](http://mirnazim.org/writings/vim-plugins-i-use/)
6. [所需即所获：像 IDE 一样使用 vim](https://github.com/yangyangwithgnu/use_vim_as_ide): 非常长，不过很详尽，也很新。
7. [一些VIM的个性化配置](http://www.wklken.me/posts/2016/02/03/some-vim-configs.html): 推荐，比较符合我的爱好。
8. 耗子的系列文章：
	1. [简明 Vim 练级攻略](http://coolshell.cn/articles/5426.html)
	2. [无插件Vim编程技巧](http://coolshell.cn/articles/11312.html)
	3. [Vim的分屏功能](http://coolshell.cn/articles/1679.html)
	4. [将vim变得简单:如何在vim中得到你最喜爱的IDE特性](http://coolshell.cn/articles/894.html)
