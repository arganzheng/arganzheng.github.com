---
title: markdown中图片如何指定大小
layout: post
tags: [markdown, kramdown, jekyll]
category: 技术
---


Markdown让你专注于内容而不是格式，但是有时候你确实想要控制一下显示效果，比如说图片。在markdown中，图片是通过这样的方式插入的：

```markdown
![test image size](/img/post-bg-2015.jpg)
```

显示效果如下：

![test image size](/img/post-bg-2015.jpg)

可以看到默认是最大化，如果我们想要控制图片的显示大小，怎么做呢？最简单直观的做法就是使用原始的HTML标签：

```html
<img src="/img/post-bg-2015.jpg" height="700px" width="400px" >
```

显示效果如下：

<img src="/img/post-bg-2015.jpg" height="700px" width="400px" >

但是这样子就不是纯粹的markdown文本了，是markdown跟html的混合体，当然图片这个html标签还算是简洁。

另一种方式就是通过外部的CSS控制，对post中的所有img做统一处理：


```css
img[alt=img_alt_you_want_to_control] {
  width: 70%;
  border: none;
  background: none;
}
```

然后对于你想要控制显示效果的img使用这个alt:

```markdown
![img_alt_you_want_to_control](/img/post-bg-2015.jpg)
```

不过这种方式对所有的img显示效果统一处理了，而且对alt有要求（当然你可以用其他的方式定位到文章中的img），不是很灵活。


谷歌了一下，发现其实有些markdown实现是支持指定图片属性的，如 [kramdown](https://kramdown.gettalong.org/syntax.html#images):


> Here is an inline `![smiley](smiley.png){:height="36px" width="36px"}`.

会渲染成：

```html
<img src="smiley.jpg" alt="smiley" height="36px" width="36px">
```

所以我们可以这么指定：

```markdown
![test image size](/img/post-bg-2015.jpg){:class="img-responsive"}
![test image size](/img/post-bg-2015.jpg){:height="50%" width="50%"}
![test image size](/img/post-bg-2015.jpg){:height="700px" width="400px"}
```

渲染结果如下：

```html
<img src="/img/post-bg-2015.jpg" alt="test image size" class="img-responsive">
<img src="/img/post-bg-2015.jpg" alt="test image size" height="50%" width="50%">
<img src="/img/post-bg-2015.jpg" alt="test image size" height="700px" width="400px">
```

显示效果如下：

![test image size](/img/post-bg-2015.jpg){:class="img-responsive"}
![test image size](/img/post-bg-2015.jpg){:height="50%" width="50%"}
![test image size](/img/post-bg-2015.jpg){:height="700px" width="400px"}


因为我们没有在CSS上定义 `img-responsive` class，所以这里第一个没有效果。

你需要在你的 jekyll 配置文件(_config)中告诉它使用 kramdown 渲染：

```
# Markdown settings
# replace redcarpet to kramdown,
# although redcarpet can auto highlight code, the lack of header-id make the catalog impossible, so I switch to kramdown
# document: http://jekyllrb.com/docs/configuration/#kramdown
markdown: kramdown
kramdown:
  input: GFM                            # use Github Flavored Markdown !important
```

有个好消息是Github默认就是使用kramdown，所以在github上写markdown文件也可以用这个语法。

---

**TIPS** 上面第二种方式，也可以使用隐藏的html标签做控制：

```
![test image size](/img/post-bg-2015.jpg)<!-- .element height="50%" width="50%" -->
![test image size](/img/post-bg-2015.jpg)<!-- .element style="border: 0; background: None; box-shadow: None" -->
```

不过个人感觉还是kramdown的语法简单明了。

Enjoy writing!

--EOF--


### 参考文章

1. [Inserting Images in Markdown Jekyll Posts](http://dev-notes.eu/2016/01/images-in-kramdown-jekyll/)