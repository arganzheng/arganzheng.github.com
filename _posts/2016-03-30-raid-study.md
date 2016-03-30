---
title: RAID学习
layout: post
---


最近申请到两台机器，RMS系统上显示的配置信息如下：

	CPU: INTEL Xeon E5-2620 V2 6C 2.1GHZ:2
	内存: HUAWEI PC3L-10600 16G:6
	硬盘: HUAWEI SATA 500G 7.2K:2
	Flash: SSD 480G MLC:5

可以看到有5块480G的SSD盘。我们知道使用合适的RAID可以提到读性能，增加数据的安全性。那么五块盘安装什么磁盘阵列合适呢？

大学的时候学习过，但是因为平时基本不用关心这个，也就记得一点点印象，现在刚好趁这个机会学习一下。


什么是RAID
---------

RAID全称为独立磁盘冗余阵列(Redundant Array of Independent Disks)，它的基本思想就是把多个相对便宜的硬盘组合起来，成为一个硬盘阵列组，使性能达到甚至超过一个价格昂贵、 容量巨大的硬盘。

RAID通常被用在服务器电脑上，使用完全相同的硬盘组成一个逻辑扇区，因此操作系统只会把它当做一个硬盘。 

RAID分为不同的等级，各个不同的等级均在数据可靠性及读写性能上做了不同的权衡。 在实际应用中，可以依据自己的实际需求选择不同的RAID方案。


常见的RAID方案
-------------

### RAID 0

RAID0称为条带化(Striping)存储，将数据分段存储于各个磁盘中，读写均可以并行处理。因此其读写速率为单个磁盘的N倍(N为组成RAID0的磁盘个数)，但是却没有数据冗余，单个磁盘的损坏会导致数据的不可修复。

![RAID 0](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/RAID_0.svg/325px-RAID_0.svg.png)

大多数striping的实现允许管理者通过调节两个关键的参数来定义数据分段及写入磁盘的方式，这两个参数对RAID0的性能有很重要的影响。

**STRIPE WIDTH**

stripe width是指可被并行写入的 stripe 的个数，即等于磁盘阵列中磁盘的个数。

**STRIPE SIZE**

也可称为block size(chunk size，stripe length，granularity)，指写入每个磁盘的数据块大小。以块分段的RAID通常可允许选择的块大小从 2KB 到 512KB不等，也有更高的，但一定要是2的指数倍。以字节分段的(比如RAID3)一般的stripe size为1字节或者 512字节，并且用户不能调整。 stripe size对性能的影响是很难简单估量的，最好在实际应用中依自己需求多多调整并观察其影响。通常来说，减少stripe size，文件会被分成更小的块，传输数据会更快，但是却需要更多的磁盘来保存，增加positioning performance，反之则相反。应该说，没有 一个理论上的最优的值。很多时候，也要考虑磁盘控制器的策略，比如有的磁盘控制器会等 等到一定数据量才开始往磁盘写入。

### RAID 1

镜像存储(mirroring)，没有数据校验。数据被同等地写入两个或多个磁盘中，可想而知，写入速度会比较慢，但读取速度会比较快。读取速度可以接近所有磁盘吞吐量的总和，写入速度受限于最慢的磁盘。 RAID1也是磁盘利用率最低的一个。如果用两个不同大小的磁盘建立RAID1，可以用空间较小的那一个，较大的磁盘多出来的部分可以作他用，不会浪费。

![RAID 1](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/RAID_1.svg/130px-RAID_1.svg.png)


### RAID 5

RAID Level 5是一种储存性能、数据安全和存储成本兼顾的存储解决方案。

它使用的是Disk Striping（硬盘分区）技术。RAID 5至少需要三块硬盘，RAID 5不是对存储的数据进行备份，而是把数据和相对应的奇偶校验信息存储到组成RAID5的各个磁盘上，并且奇偶校验信息和相对应的数据分别存储于不同的磁盘上。当RAID5的一个磁盘数据发生损坏后，可以利用剩下的数据和相应的奇偶校验信息去恢复被损坏的数据。

RAID 5可以理解为是RAID 0和RAID 1的折衷方案。RAID 5可以为系统提供数据安全保障，但保障程度要比镜像低而磁盘空间利用率要比镜像高。RAID 5具有和RAID 0相近似的数据读取速度，只是因为多了一个奇偶校验信息，写入数据的速度相对单独写入一块硬盘的速度略慢，若使用“回写缓存”可以让性能改善不少。同时由于多个数据对应一个奇偶校验信息，RAID 5的磁盘空间利用率要比RAID 1高，存储成本相对较便宜。


![RAID 5](https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/RAID_5.svg/220px-RAID_5.svg.png)

可用空间大小: Size = (N-1) x MIN(S1, S2, ... ,Sn)


### RAID 10

顾名思义，RAID 10是先镜射再分区数据，再将所有硬盘分为两组，视为是RAID 0的最低组合，然后将这两组各自视为RAID 1运作。

![RAID 10](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/RAID_10.svg/220px-RAID_10.svg.png)


**TIPS** RAID 01

RAID 01则是跟RAID 10的程序相反，是先分区再将数据镜射到两组硬盘。它将所有的硬盘分为两组，变成RAID 1的最低组合，而将两组硬盘各自视为RAID 0运作。

![RAID 01](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/RAID_01.svg/220px-RAID_01.svg.png)

当RAID 10有一个硬盘受损，其余硬盘会继续运作。RAID 01只要有一个硬盘受损，同组RAID 0的所有硬盘都会停止运作，只剩下其他组的硬盘运作，可靠性较低。如果以六个硬盘建RAID 01，镜射再用三个建RAID 0，那么坏一个硬盘便会有三个硬盘脱机。因此，RAID 10远较RAID 01常用，零售主板绝大部分支持RAID 0/1/5/10，但不支持RAID 01。

### RAID 50

RAID 5与RAID 0的组合，先作RAID 5，再作RAID 0，也就是对多组RAID 5彼此构成Stripe访问。由于RAID 50是以RAID 5为基础，而RAID 5至少需要3颗硬盘，因此要以多组RAID 5构成RAID 50，至少需要6颗硬盘。以RAID 50最小的6颗硬盘配置为例，先把6颗硬盘分为2组，每组3颗构成RAID 5，如此就得到两组RAID 5，然后再把两组RAID 5构成RAID 0。

RAID 50在底层的任一组或多组RAID 5中出现1颗硬盘损坏时，仍能维持运作，不过如果任一组RAID 5中出现2颗或2颗以上硬盘损毁，整组RAID 50就会失效。

RAID 50由于在上层把多组RAID 5构成Stripe，性能比起单纯的RAID 5高，容量利用率比RAID5要低。比如同样使用9颗硬盘，由各3颗RAID 5再组成RAID 0的RAID 50，每组RAID 5浪费一颗硬盘，利用率为(1-3/9)，RAID 5则为(1-1/9)。


![RAID 50](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/RAID_50.png/220px-RAID_50.png)


应用和实战
---------

### 实际应用情况

1. RAID2、3、4较少实际应用，因为RAID5已经涵盖了所需的功能，因此RAID2、3、4大多只在研究领域有实现，而实际应用上则以RAID5为主。
2. RAID4有应用在某些商用机器上，像是NetApp公司设计的NAS系统就是使用RAID4的设计概念。
3. RAID10和RAID5也是经常用来比较的两种方案，二者都在生产实践中得到了广泛的应用。RAID10安全性更高，但是空间利用率低。至于读写性能，与cache有很大关联，最好根据实际情况测试比较选择。
4. 一般来说，Raid5可以满足最优的读写性能和安全性，如果想更保险可以使用Raid50，更土豪使用Raid 10。

### 实战

怎样查看机器的RAID情况呢？在Linux下可以使用`MegaCli`命令查看。但是要先安装。

	# MegaCli -cfgdsply –aALL # 查看硬盘信息，会打印一大堆东东。。
	
使用下面参数可以查看RAID级别信息：

	# MegaCli  -LDInfo -Lall -aALL


	Adapter 0 -- Virtual Drive Information:
	Virtual Drive: 0 (Target Id: 0)
	Name                :
	RAID Level          : Primary-1, Secondary-0, RAID Level Qualifier-0
	Size                : 464.729 GB
	Is VD emulated      : No
	Mirror Data         : 464.729 GB
	State               : Optimal
	Strip Size          : 128 KB
	Number Of Drives    : 2
	Span Depth          : 1				## 深度若是1，说明实际只是RAID 1。2表示可以是RAID 10
	Default Cache Policy: WriteBack, ReadAhead, Direct, Write Cache OK if Bad BBU
	Current Cache Policy: WriteBack, ReadAhead, Direct, Write Cache OK if Bad BBU
	Default Access Policy: Read/Write
	Current Access Policy: Read/Write
	Disk Cache Policy   : Disk's Default
	Encryption Type     : None
	PI type: No PI

	Is VD Cached: No


	Virtual Drive: 1 (Target Id: 1)
	Name                :
	RAID Level          : Primary-5, Secondary-0, RAID Level Qualifier-3
	Size                : 1.742 TB
	Is VD emulated      : Yes
	Parity Size         : 446.102 GB
	State               : Optimal
	Strip Size          : 128 KB
	Number Of Drives    : 5
	Span Depth          : 1					## 深度若是1，说明实际只是RAID 5。2表示可以是RAID 50
	Default Cache Policy: WriteBack, ReadAheadNone, Direct, Write Cache OK if Bad BBU
	Current Cache Policy: WriteBack, ReadAheadNone, Direct, Write Cache OK if Bad BBU
	Default Access Policy: Read/Write
	Current Access Policy: Read/Write
	Disk Cache Policy   : Disk's Default
	Encryption Type     : None
	PI type: No PI

	Is VD Cached: No



	Exit Code: 0x00

可以看到我们这个机器有两个虚拟设备（RAID阵列），两块SATA盘组成了`RAID Level          : Primary-1, Secondary-0, RAID Level Qualifier-0`，因为Span Depth=1，所以表示使用了RAID 1；5块SSD盘组成了`RAID Level          : Primary-5, Secondary-0, RAID Level Qualifier-3`，表示使用了RAID 50。因为Span Depth=1，所以表示使用了RAID 5。其实看到五块盘就知道不可能是RAID 50了。

具体可以参考: [megacli 查看Raid卡和硬盘信息](http://tenderrain.blog.51cto.com/9202912/1639865)

MegaCli不只是可以用来查看RAID信息，还可以创建RAID，具体参见: [How to use the MegaCLI Utility with your RAID Controller on your PowerEdge Server in Linux](http://www.dell.com/support/article/us/en/19/623352/en)


参考文章
-------

1. [RAID技术介绍和总结](http://blog.jobbole.com/83808/)
2. [维基百科 RAID](https://zh.wikipedia.org/wiki/RAID)
3. [megacli 查看Raid卡和硬盘信息](http://tenderrain.blog.51cto.com/9202912/1639865)
