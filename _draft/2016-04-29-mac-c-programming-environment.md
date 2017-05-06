---
title: mac下C编程环境
layout: post
catalog: true
---

最近转岗到BDG部门，参与图片广告变现项目。基本复用了凤巢的代码，在上面进行修改和优化。凤巢的代码是C++编写的，所以只能转C++了。遇到的第一个问题就是搭建一个高效的C++编程环境。其实主要是寻找一个顺手好用的IDE。

不像Java，基本上就是在这两个IDE中选择：Eclipse或者IDEA，都是支持多个平台，而且都非常好用。C++的IDE相对混乱一些，各个平台都有不同的IDE支持，而且免费的还不怎么好用，特别如果需要能够编译和Debug，那么基本都是收费IDE，比如Visual Studio，而且只支持Windows平台。

笔者的工作环境是Mac，而且是屌丝，所以就想找一下有没有Mac下好用的C++ IDE。谷歌了一下，发现有下面几个选择：

* XCode
	* Mac自带的IDE，支持很多种语言
	* 有点大块头。。启动非常慢。。运行有点卡。。
* Vim/Emacs + ctags + [cscope] + gdb + 各种插件
	* 老派程序员的推荐，腾讯程序员的标配。一开始学习曲线比较高，不过一旦熟悉之后就会如鱼得水。
	* ctags: 跳到函数或者变量定义 (Open Declaration)
	* cscope: 查看函数调用堆栈 (Open Call Hierarchy)
	* gdb: 调试用的
	* 推荐VIM插件如下: 
		* NERDTree : toggle between files
		* YouCompleteMe: autocomplete support for C/C++
		* Syntastic: syntax checking
		* Git-gutter: git support
		* clewn:debug support
* Eclipse with CDT
	* 不是很好用，各种红叉叉。。
* Sublime Text
	* 一个Python写的非常好用的文本编辑器
	* 可以花点功夫打造一下变成一个简陋的IDE: [Sublime Text 3 化身为高大上的C/C++ IDE](https://xuanwo.org/2014/06/05/sublime-text-3-IDE/)
* [Atom](https://atom.io/)
	* Github团队开源的一个文本编辑器，跟Sublime Text很像。
	* 同样可以花点功夫打造成一个简陋的IDE: [Atom plugins for C++ development](https://blogs.aerys.in/jeanmarc-leroux/2015/07/31/atom-plugins-for-c-development/)
* [Visual Studio Code](https://code.visualstudio.com/)
	* 微软开源的一个简化版Visual Studio，支持多个平台。
	* 参考了Sublime Text的Command Palette模式，可以不用记住太多的快捷键。
	* 速度非常快，也非常好用，不过内建支持语言是JavaScript, TypeScript, Node.js。其他语言需要安装相应的扩展插件。很遗憾，[C++的扩展](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)目前只是Release版本，支持并不是很好。不过基本够用。是一个非常有前途的IDE。
	* 国人已经有在使用了：[学会用好 Visual Studio Code](http://nshen.net/article/2015-11-20/vscode/)


最后的选择，打算装一下逼，试试: Vim + ctags + [cscope] + gdb + 各种插件，毕竟公司大部分老牌程序员的选择，也是腾讯程序员的选择，而且其实笔者以前也折腾过Vim，只是没有使用它来编写C++代码而已。

1. [工欲善其事，必先利其器——从零打造你的vim](http://arganzheng.life/living-with-vim.html)
2. [所需即所获：像 IDE 一样使用 vim](https://github.com/yangyangwithgnu/use_vim_as_ide)

照着作者的vimrc配置文件配置，然后进入 vim 执行 :PluginInstall。看到插件一个个的被安装，感觉确实蛮自动化的。不过有两个插件没有成功安装：

	arganzhengs-MacBook-Pro:~ argan$ vim .vimrc
	UltiSnips requires Vim >= 7.4
	YouCompleteMe unavailable: requires Vim 7.3.598+

而我的mac自带的vim版本信息如下：

	arganzhengs-MacBook-Pro:bin argan$ vim --version
	VIM - Vi IMproved 7.3 (2010 Aug 15, compiled Oct 28 2015 19:46:19)
	Compiled by root@apple.com
	Normal version without GUI.  Features included (+) or not (-):
	-arabic +autocmd -balloon_eval -browse +builtin_terms +byte_offset +cindent
	-clientserver -clipboard +cmdline_compl +cmdline_hist +cmdline_info +comments
	-conceal +cryptv +cscope +cursorbind +cursorshape +dialog_con +diff +digraphs
	-dnd -ebcdic -emacs_tags +eval +ex_extra +extra_search -farsi +file_in_path
	+find_in_path +float +folding -footer +fork() -gettext -hangul_input +iconv
	+insert_expand +jumplist -keymap -langmap +libcall +linebreak +lispindent
	+listcmds +localmap -lua +menu +mksession +modify_fname +mouse -mouseshape
	-mouse_dec -mouse_gpm -mouse_jsbterm -mouse_netterm -mouse_sysmouse
	+mouse_xterm +multi_byte +multi_lang -mzscheme +netbeans_intg -osfiletype
	+path_extra -perl +persistent_undo +postscript +printer -profile +python/dyn
	-python3 +quickfix +reltime -rightleft +ruby/dyn +scrollbind +signs
	+smartindent -sniff +startuptime +statusline -sun_workshop +syntax +tag_binary
	+tag_old_static -tag_any_white -tcl +terminfo +termresponse +textobjects +title
	 -toolbar +user_commands +vertsplit +virtualedit +visual +visualextra +viminfo
	+vreplace +wildignore +wildmenu +windows +writebackup -X11 -xfontset -xim -xsmp
	 -xterm_clipboard -xterm_save
	   system vimrc file: "$VIM/vimrc"
	     user vimrc file: "$HOME/.vimrc"
	      user exrc file: "$HOME/.exrc"
	  fall-back for $VIM: "/usr/share/vim"
	Compilation: gcc -c -I. -D_FORTIFY_SOURCE=0 -Iproto -DHAVE_CONFIG_H -arch i386 -arch x86_64 -g -Os -pipe
	Linking: gcc -arch i386 -arch x86_64 -o vim -lncurses

妹。。就差一点点就符合要求了。。难怪use vim as ide的作者第一件事就是教我们源码安装vim。。

	$ git clone https://github.com/vim/vim.git
	$ cd vim/
	$ ./configure --with-features=huge --enable-pythoninterp --enable-rubyinterp --enable-luainterp --enable-perlinterp --with-python-config-dir=/usr/lib/python2.7/config/ --enable-gui=gtk2 --enable-cscope --prefix=/usr
	$ make
	$ sudo -i # change to root user
	# make install

**TIPS** Mac OS的Rootless机制

在这里笔者遇到让我大吃一惊的事情：make install居然报错了，说是没有权限操作！！！谷歌了一下，发现是Mac OS从El Capitan版本开始加入了[Rootless机制](http://www.jianshu.com/p/22b89f19afd6)，root就是一个普通用户。。解决方案是重启系统，按照Command+r，进入恢复模式菜单，选择Utils=>Terminal，进入终端，输入命令：

	# csrutil disable
	Successfully disable System Integrity Protection, Please restart the machine for the changes to take effect.
	# reboot

重启之后进入，sudo和root就妥妥的了。现在版本是最新的了：

	arganzhengs-MacBook-Pro:~ argan$ vim --version
	VIM - Vi IMproved 7.4 (2013 Aug 10, compiled Apr 29 2016 18:28:20)
	MacOS X (unix) version
	Included patches: 1-1797
	Compiled by argan@arganzhengs-MacBook-Pro.local
	Huge version without GUI.  Features included (+) or not (-):
	+acl             +farsi           +mouse_netterm   +tag_binary
	+arabic          +file_in_path    +mouse_sgr       +tag_old_static
	+autocmd         +find_in_path    -mouse_sysmouse  -tag_any_white
	-balloon_eval    +float           +mouse_urxvt     -tcl
	-browse          +folding         +mouse_xterm     +terminfo
	++builtin_terms  -footer          +multi_byte      +termresponse
	+byte_offset     +fork()          +multi_lang      +termtruecolor
	+channel         -gettext         -mzscheme        +textobjects
	+cindent         -hangul_input    +netbeans_intg   +timers
	-clientserver    +iconv           +packages        +title
	+clipboard       +insert_expand   +path_extra      -toolbar
	+cmdline_compl   +job             +perl            +user_commands
	+cmdline_hist    +jumplist        +persistent_undo +vertsplit
	+cmdline_info    +keymap          +postscript      +virtualedit
	+comments        +langmap         +printer         +visual
	+conceal         +libcall         +profile         +visualextra
	+cryptv          +linebreak       +python          +viminfo
	+cscope          +lispindent      -python3         +vreplace
	+cursorbind      +listcmds        +quickfix        +wildignore
	+cursorshape     +localmap        +reltime         +wildmenu
	+dialog_con      -lua             +rightleft       +windows
	+diff            +menu            +ruby            +writebackup
	+digraphs        +mksession       +scrollbind      -X11
	-dnd             +modify_fname    +signs           -xfontset
	-ebcdic          +mouse           +smartindent     -xim
	+emacs_tags      -mouseshape      +startuptime     -xsmp
	+eval            +mouse_dec       +statusline      -xterm_clipboard
	+ex_extra        -mouse_gpm       -sun_workshop    -xterm_save
	+extra_search    -mouse_jsbterm   +syntax          -xpm
	   system vimrc file: "$VIM/vimrc"
	     user vimrc file: "$HOME/.vimrc"
	 2nd user vimrc file: "~/.vim/vimrc"
	      user exrc file: "$HOME/.exrc"
	  fall-back for $VIM: "/usr/share/vim"
	Compilation: gcc -c -I. -Iproto -DHAVE_CONFIG_H   -DMACOS_X_UNIX  -g -O2 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=1
	Linking: gcc   -L. -L/usr/local/lib  -L/usr/local/lib -o vim        -lm -lncurses  -liconv -framework Cocoa   -fstack-protector  -L/System/Library/Perl/5.18/darwin-thread-multi-2level/CORE -lperl -framework Python   -lruby.2.0.0 -lobjc

但是启动还是报错：

	arganzhengs-MacBook-Pro:~ argan$ vim .vimrc
	ycm_core.[so|pyd|dll] not detected; you need to compile YCM before using it. Read the docs!

这是因为YCM是一个Client-Server架构的软件，与vim交互的插件部分仅仅是Client，做真正语义补全的Server部分是需要编译的，编译的结果就是上面错误日志说的`ycm_core.[so|pyd|dll]`共享库。

C++的智能补全是依赖于 libclang 共享库，所以我们需要先下载最新版的libclang。不用担心，YCM的install脚本会自动下载：

	arganzhengs-MacBook-Pro:~ argan$ cd .vim/bundle/YouCompleteMe/
	arganzhengs-MacBook-Pro:YouCompleteMe argan$ ./install.sh --clang-completer
	WARNING: this script is deprecated. Use the install.py script instead.
	Searching for python libraries...
	Searching for python with prefix: /System/Library/Frameworks/Python.framework/Versions/2.7 and lib python2.7:
	Using PYTHON_LIBRARY=/System/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib PYTHON_INCLUDE_DIR=/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7
	-- The C compiler identification is AppleClang 7.0.0.7000176
	-- The CXX compiler identification is AppleClang 7.0.0.7000176
	-- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
	-- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
	-- Detecting C compiler ABI info
	-- Detecting C compiler ABI info - done
	-- Detecting C compile features
	-- Detecting C compile features - done
	-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
	-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
	-- Detecting CXX compiler ABI info
	-- Detecting CXX compiler ABI info - done
	-- Detecting CXX compile features
	-- Detecting CXX compile features - done
	Your C++ compiler supports C++11, compiling in that mode.
	-- Found PythonLibs: /System/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib (found suitable version "2.7.10", minimum required is "2.6")
	Downloading Clang 3.8.0
	-- [download 0% complete]
	-- [download 1% complete]

	....

	-- [download 99% complete]
	-- [download 100% complete]
	Using libclang to provide semantic completion for C/C++/ObjC
	Using external libclang: /private/var/folders/h2/78s9jtg967jf33mx2rwy2fgr0000gn/T/ycm_build.5AQdek/clang+llvm-3.8.0-x86_64-apple-darwin/lib/libclang.dylib
	-- Found PythonInterp: /usr/bin/python2.7 (found version "2.7.10")
	-- Looking for pthread.h
	-- Looking for pthread.h - found
	-- Looking for pthread_create
	-- Looking for pthread_create - found
	-- Found Threads: TRUE
	-- Configuring done
	-- Generating done
	-- Build files have been written to: /private/var/folders/h2/78s9jtg967jf33mx2rwy2fgr0000gn/T/ycm_build.5AQdek
	Scanning dependencies of target BoostParts
	[  0%] Building CXX object BoostParts/CMakeFiles/BoostParts.dir/libs/atomic/src/lockpool.cpp.o
	[  0%] Building CXX object BoostParts/CMakeFiles/BoostParts.dir/libs/chrono/src/process_cpu_clocks.cpp.o

	....

	[100%] Linking CXX shared library /Users/argan/.vim/bundle/YouCompleteMe/third_party/ycmd/ycm_core.so
	[100%] Built target ycm_core


**NOTES && TIPS** 

1、YCM的安装脚本默认会下载libclang，如果想要让YCM使用系统的libclang，可以增加一个编译选项重新编译一下：

	./install.py --clang-completer --system-clang

2、不过 YCM 作者[强烈不建议使用system libclang](https://github.com/Valloric/YouCompleteMe#full-installation-guide)，以避免各种妖人问题。最好是自己从llvm.org官网下载预编译的二进制文件：

在 [official binaries from llvm.org](http://llvm.org/releases/download.html) 找到最新版 LLVM，Pre-built Binaries 下选择适合你发行套件的最新版预编译二进制文件，下载并解压至 ~/Downloads/clang+llvm。

	$ cd ~/Downloads/
	$ wget http://llvm.org/releases/3.8.0/clang+llvm-3.8.0-x86_64-apple-darwin.tar.xz
	$ tar zxvf clang+llvm-3.8.0-x86_64-apple-darwin.tar.xz


然后我们可以开始编译 YCM 共享库了。

	$ cd ~/Downloads/
	$ mkdir ycm_build
	$ cd ycm_build 
	$ zypper --no-refresh se python-devel python3-devel boost-devel llvm-clang-devel
	$ cmake -G "Unix Makefiles" -DUSE_SYSTEM_BOOST=ON -DPATH_TO_LLVM_ROOT=~/downloads/clang+llvm/ . ~/.vim/bundle/YouCompleteMe/third_party/ycmd/cpp
	$ cmake --build . --target ycm_core

在 ~/.vim/bundle/YouCompleteMe/third_party/ycmd 中将生成 ycm_client_support.so、ycm_core.so、libclang.so 等三个共享库文件；


CMake Error at /usr/local/Cellar/cmake/3.4.3/share/cmake/Modules/FindBoost.cmake:1247 (message):
  Unable to find the requested Boost libraries.

  Unable to find the Boost header files.  Please set BOOST_ROOT to the root
  directory containing Boost or BOOST_INCLUDEDIR to the directory containing
  Boost's headers.
Call Stack (most recent call first):
  ycm/CMakeLists.txt:196 (find_package)



OK，编译安装好了，让我们打开vim试试效果。结果报错了：

	oExtraConfDetected: No .ycm_extra_conf.py file detected, so no compile flags are available. Thus no semantic support for C/C++/ObjC/ObjC++. Go READ THE DOCS *NOW*, DON'T file a bug report.

还差最后一步，我们需要配置一下YCM。YCM 后端调用 libclang 进行语义分析，而 libclang 有很多参数选项（如，是否支持 C++11 的 -std=c++11、是否把警告视为错误的 -Werror），必须有个渠道让 YCM 能告知 libclang。YCM默认会在我们打开的文件的当前目录查找一个`.ycm_extra_conf.py`配置文件，如果找不到会依次向上级目录查找（递归）。一般来说我们的工程都是在home目录下，所以可以把它放在home目录下（跟.vim同级）。当然，你也可以配置一下 .vimrc 来改变YCM的默认加载目录：

	let g:ycm_global_ycm_extra_conf = "~/.vim/.ycm_extra_conf.py"

但是不同的工程使用的 libclang 参数选项可能根据需要有所不同（99%的情况需要修改.ycm_extra_conf.py的flags参数），所以一种简单有效的做法就是home目录的.ycm_extra_conf.py文件作为默认(fallback)的配置文件，对于需要定制化的工程，在该工程的根目录下新建一个 .ycm_extra_conf.py 作为其私有配置文件，在此文件中写入该工程的编译参数选项。


**TIPS** 如何查看YCM问题

如果你的YCM没有按照预期的效果运行，可以在vim中通过 `:messages`命令查看vim的日志:

	Messages maintainer: Bram Moolenaar <Bram@vim.org>
	"Work/BDG/code/bd-retras/src/DqRpcClient.cpp" [converted] 618L, 20126C
	The ycmd server SHUT DOWN (restart with ':YcmRestartServer'). Run ':YcmToggleLogs stderr' to check the logs.
	Restarting ycmd server...
	Printing YouCompleteMe debug information...
	-- Server crashed, no debug info from server
	-- Server running at: http://127.0.0.1:57485
	-- Server process ID: 4381
	-- Server logfiles:
	--   /var/folders/h2/78s9jtg967jf33mx2rwy2fgr0000gn/T/ycm_temp/server_57485_stdout.log
	--   /var/folders/h2/78s9jtg967jf33mx2rwy2fgr0000gn/T/ycm_temp/server_57485_stderr.log
	The ycmd server SHUT DOWN (restart with ':YcmRestartServer'). Run ':YcmToggleLogs stderr' to check the logs.

然后可以看到YCM的日志文件路径，再根据日志文件定位问题。


### ctags

ctags 默认并不会提取所有标签，运行

	ctags --list-kinds=c++

可看到 ctags 支持生成标签类型的全量列表：

	c  classes 
	d  macro definitions 
	e  enumerators (values inside an enumeration) 
	f  function definitions 
	g  enumeration names 
	l  local variables [off] 
	m  class, struct, and union members 
	n  namespaces 
	p  function prototypes [off] 
	s  structure names 
	t  typedefs 
	u  union names 
	v  variable definitions 
	x  external and forward variable declarations [off] 

其中，标为 off 的局部对象、函数声明、外部对象等类型默认不会生成标签，可以显式让其加上所有类型：

	cd /data/workplace/example/
	ctags -R --c++-kinds=+p+l+x+c+d+e+f+g+m+n+s+t+u+v --fields=+liaS --extra=+q --language-force=c++

**NOTES**

mac自带的是ctags (/usr/bin/ctags)，而tagbar要求的是exuberant crags，用homebrew装一个就可以:

	$ brew install ctags

默认按照在`/usr/local/bin/ctags`下，但是如果不指定全路径的话，默认还是会优先使用/usr/bin/ctags。所以我们可以简单的把`/usr/bin/ctags`给干掉：
	# mv ctags ctags.rm


参考文章
-------

1. [工欲善其事，必先利其器——从零打造你的vim](http://arganzheng.life/living-with-vim.html)
2. [所需即所获：像 IDE 一样使用 vim](https://github.com/yangyangwithgnu/use_vim_as_ide)
