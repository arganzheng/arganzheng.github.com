---
layout: post
title: github on windows
---

### 使用[github for windows](http://windows.github.com/)

### 设置代理

To configure GitHub for Windows to use your  corporate proxy, edit the .gitconfig file typically found at C:\Users\<yourusername>\.gitconfig or C:\Documents & Settings\<yourusername>\.gitconfig

    [user]
        name = arganzheng
    	email = arganzheng@gmail.com
    [http]
    proxy = http://web-proxy.oa.com:8080
    [https]
    proxy = http://web-proxy.oa.com:8080

不错的文章： [git简易指南](http://rogerdudler.github.io/git-guide/index.zh.html)