---
title: tips for sublime text
layout: post
---


### Command Palette

类似于shell终端，提供了sublime text的所有支持命令。可以说是总入口，比如Package Control就是通过Command Palette调用出来的(Press ⌘⇧P to bring up the command palette, begin to type “install”, and select “Package Control: Install Package”. )。快捷键为Command-P(`⌘⇧P`)。

### Package Control

安装完Sublime text之后，先安装一个插件管理器方便插件的安装和卸载。[Sublime Package Control](http://wbond.net/sublime_packages/package_control/installation)。内网安装插件要设置代理：[Sublime Text 2 Package Control安装与设置代理方法](http://www.tanktan.com/blog/sublime2-proxy/)。

### Command line Tool(命令行打开)

建立软连接：

    ln -s "/Applications/Sublime Text 2.app/Contents/SharedSupport/bin/subl" /usr/bin/subl

然后就可以在命令行直接用sublime text打开文件和目录了：

    subl -n ~/blog ## 打开blog目录作为项目
    sub1 -n . ## 编辑当前目录
    
### 支持GBK

Sublime Text 2默认不支持GB2312和GBK，可以安装一个插件让它支持。[解决乱码，让Sublime Text 2支持GB2312和GBK](http://www.fuzhaopeng.com/2012/sublime-text-2-with-gb2312-gbk-support/)。


### Customizing Settings(自定义)

On OS X, go to “Sublime Text 2 » Preferences”.

### Go To File Palette(⌘p/⌘t)

⌘p或者⌘t打开Go To File面板。非常强大，支持的pattern有: 
    
1. fileName     ## Go To File
2. [fileName]@methodName    ## Go To method of file，if fileName为空，表示当前文件，等效于⌘r。
3. [fileName]:lineNumber    ## Go To line number of file, if fileName为空，表示当期文件，等效于Ctrl-g。

### Favorite Plugins

### 常用快捷键

1. 删除当前行 Ctrl-x

### Sublime Text As IDE

#### Sublime Text as Python IDE

1. 安装SublimeRope for python autocompletion
2. Python lint:SublimeLinter
3. Debugbing:SublimeREPL



## 参考文章

1. [Getting Started with Sublime Text 2](http://opensoul.org/blog/archives/2012/01/12/getting-started-with-sublime-text-2/)
2. [Setting up Sublime Text for Ruby development](http://zhuravel.biz/setting-up-sublime-text-for-ruby-development)
