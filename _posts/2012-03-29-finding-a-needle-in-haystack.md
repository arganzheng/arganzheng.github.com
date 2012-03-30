---
layout: post
title: 海量图片存储思考
---

最近在学习分布式存储和数据分析相关的东西，特别是看了Facebook的这篇论文：[Finding a needle in Haystack: Facebook’s photo storage][haystack]，感觉特别与阿里巴巴现在的现状类似。阿里现在的图片存储正处于Facebook的原始阶段——CDN(中美有各自的CDN) ==> imageServer(apache+squid) ==> NFS共享存储。

说明：CDN的作用

CDN只适合于缓存popular static content(images, html, css, js, etc.)。对于long tail（长尾）文件，并不适合。这就是我们上次ASC从前台dump图片，将存储搞挂的原因。facebook使用一个类似于上传时间戳的东东作为cookies，来防止这类攻击。

[haystack]:http://static.usenix.org/event/osdi10/tech/full_papers/Beaver.pdf "Finding a needle in Haystack: Facebook’s photo storage"

海量小文件的主要问题在于metadata的开销，操作系统根据一个filePath读取相应的文件时，至少需要三次磁盘IO：
1. 根据filePath获取inode number
2. 根据inode number读取inode块
3. 根据inode块中的信息读取相应的data（对于小文件，data只需要一次索引就可以全部读取完毕）

说明：对于第一个步骤：finding an inode by reference to a given filename实际上是个很复杂的过程，大概过程如下：

使用nameidata{}作为lookup函数的参数和返回值：
    <fs.h>
    struct nameidata {
        struct dentry
        struct vfsmount
        struct qstr
        unsigned int
        ...
    }

The kernel uses the path_lookup function to find any path or filename.
   fs/namei.c
   int fastcall path_lookup(const char *name, unsigned int flags, struct nameidata *nd)

检查name返回是否以/开始，如果是说明是绝对路径，通过current->fs->root (the process root directory)得到。 否则，是相对路径，使用当前路径，通过current->fs->pwd (the process-current directory)获取。 
拿到起始目录的dentry结果(root or pwd)，从而得到该dentry对应的inode，那么根据这个inode我们可以得到它对应的子目录，通过比较得到相应的目录项，从而又得到子目录的inode，一层一层往下走就得到最后的文件了。

为了提高性能，内核使用了dentry缓存（一个hash map和LRU list）来减少不必要的IO。

操作系统其实是有缓存filesystem metadata的，但是如果文件特别多的话，缓存不断的换入换出，反而容易产生颠簸。而且这些filesystem meta信息加起来大概是150B，非常不利于cache在内存中。而这些信息其实对于应用来说并不关心，比如user, group, atime等。

Haystack的思路其实非常的简单和直观——因为海量小文件的主要性能瓶颈在于大量的metadata读取，那么关键是减少metadata的量。

因为每个文件对应一个metadata，那么可以将多个小文件合并成一个大文件，这样就可以大大减少filesystem metadata的数量了。 另外inode也随着减少了。

然而引出了一个问题就是如何定位大文件中的小文件，每个小文件还是需要一定的metadata来描述和定位的，这就引出了application metadata的概念。

1. Application metadata: 应用层面概念。存放用于构造URL的信息，浏览器可以直接获取。
2. Filesystem metadata: 操作系统中文件系统层面概念。存放让一个host可以直接retrive保存在该host硬盘上的图片。

相对于filesystem metadata来说，application metadata只关心必要的信息，像user, group, mode, atime, ctime, mtime等信息对图片来说并不是必要的，所以可以高度精简，这样所有的application metadata就有可能全部缓存在内存中，这样每次文件读取就只是真正的数据读取了。

合并大文件中每个小文件的获取一般是采用如下方式：

对应用来说它面对的是application metadata: <logic_volume_id, photo_id>
对store engine来说：它接受应用发过来的<logic_volume_id, photo_id>，通过一个mapping hash转成filesystem metadata：<physical_volume_id, offset, size>

然而将多个小文件合并成一个大文件（即不使用传统的文件系统）会带来如下几个问题：

1. 第一个也是个人感觉最重要的一个问题是很多静态文件服务器，如apache，nginx等，因为他们面向的是操作系统的文件系统，所以他们认为一个URL对应一个文件，现在一个URL可能只是对应一个文件的一个小部分（自定义文件系统）。这样就需要特殊的处理逻辑（利用fileSystem metadata获取）。这个facebook也没有好的解决方案，他们是采用RESTful的GET方式获取图片。在URL中传递必须的信息，但是由相应的代码逻辑进行处理。

2. 另外一个问题是这种文件一般是append only，因为删除或者更新中间的小文件会带来很大的麻烦（需要空间的移动）。简单的解决方案就是不允许更新和删除：
    1. 更新相当于重新上传文件（版本号加一，以避免客户端缓存）
    2. 删除简单的标记删除，即逻辑删除而不是物理删除。

