Aerospike运行报错:no version information available (required by bin/asd) 	
======================================================================

catalog: true
昨天在线上机器搭建Aerospike集群的时候遇到一个诡异的问题。线上的环境是 centos 6.3，本来可以直接用官网提供的[Redhat 6二进制版本](http://www.aerospike.com/docs/operations/install/linux/el6)，但是执行`sudo ./asinstall`的时候报错了：

	$ sudo ./asinstall # will install the .rpm packages
	Installing tools
	rpm -Uvh aerospike-tools-3.12.1-1.el6.x86_64.rpm
	error: Failed dependencies:
	    libcrypto.so.10(libcrypto.so.10)(64bit) is needed by aerospike-tools-3.12.1-1.el6.x86_64
	    libssl.so.10(libssl.so.10)(64bit) is needed by aerospike-tools-3.12.1-1.el6.x86_64
	    libz.so.1(ZLIB_1.2.0)(64bit) is needed by aerospike-tools-3.12.1-1.el6.x86_64
	Installing server
	rpm -Uvh aerospike-server-community-3.12.1-1.el6.x86_64.rpm
	error: Failed dependencies:
	    libcrypto.so.10(libcrypto.so.10)(64bit) is needed by aerospike-server-community-3.12.1-1.el6.x86_64
	    libz.so.1(ZLIB_1.2.0)(64bit) is needed by aerospike-server-community-3.12.1-1.el6.x86_64

然后改成[Install using Binary Package](http://www.aerospike.com/docs/operations/install/linux/other)方式，这个不需要安装，但是启动的时候报错了：

	# bin/aerospike start
	error: start failed due to an error.
	/home/work/aerospike-server/bin/asd: /lib64/libz.so.1: no version information available (required by /home/work/aerospike-server/bin/asd)
	/home/work/aerospike-server/bin/asd: /usr/lib64/libcrypto.so.1.0.0: no version information available (required by /home/work/aerospike-server/bin/asd)
	Apr 07 2017 04:03:02 GMT: WARNING (cf:misc): (hardware.c:626) no NUMA information found in /sys

看一下asd具体的依赖情况：

	# ldd -v bin/asd
	bin/asd: /lib64/libz.so.1: no version information available (required by bin/asd)
	bin/asd: /usr/lib64/libcrypto.so.1.0.0: no version information available (required by bin/asd)
		linux-vdso.so.1 =>  (0x00007fff10df8000)
		libcrypto.so.1.0.0 => /usr/lib64/libcrypto.so.1.0.0 (0x00007fe258a01000)
		libpthread.so.0 => /lib64/libpthread.so.0 (0x000000318b200000)
		librt.so.1 => /lib64/librt.so.1 (0x000000318be00000)
		libdl.so.2 => /lib64/libdl.so.2 (0x000000318aa00000)
		libz.so.1 => /lib64/libz.so.1 (0x000000318ba00000)
		libm.so.6 => /lib64/libm.so.6 (0x000000318b600000)
		libstdc++.so.6 => /usr/lib64/libstdc++.so.6 (0x000000318ea00000)
		libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x000000318da00000)
		libc.so.6 => /lib64/libc.so.6 (0x000000318ae00000)
		/lib64/ld-linux-x86-64.so.2 (0x000000318a600000)

		Version information:
		bin/asd:
			libz.so.1 (ZLIB_1.2.0) => not found
			librt.so.1 (GLIBC_2.2.5) => /lib64/librt.so.1
			libdl.so.2 (GLIBC_2.2.5) => /lib64/libdl.so.2
			libgcc_s.so.1 (GCC_3.3) => /lib64/libgcc_s.so.1
			libgcc_s.so.1 (GCC_3.0) => /lib64/libgcc_s.so.1
			libcrypto.so.1.0.0 (OPENSSL_1.0.0) => not found
			libm.so.6 (GLIBC_2.2.5) => /lib64/libm.so.6
			libc.so.6 (GLIBC_2.7) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.8) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.6) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.3.4) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.3.2) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.3) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.2.5) => /lib64/libc.so.6
			libstdc++.so.6 (GLIBCXX_3.4.9) => /usr/lib64/libstdc++.so.6
			libstdc++.so.6 (CXXABI_1.3) => /usr/lib64/libstdc++.so.6
			libstdc++.so.6 (GLIBCXX_3.4) => /usr/lib64/libstdc++.so.6
			libpthread.so.0 (GLIBC_2.3.3) => /lib64/libpthread.so.0
			libpthread.so.0 (GLIBC_2.2.5) => /lib64/libpthread.so.0
			libpthread.so.0 (GLIBC_2.3.2) => /lib64/libpthread.so.0
		/usr/lib64/libcrypto.so.1.0.0:
			libdl.so.2 (GLIBC_2.2.5) => /lib64/libdl.so.2
			libc.so.6 (GLIBC_2.4) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.3) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.7) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.3.4) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.2.5) => /lib64/libc.so.6
			...

看到这两个指向的路径如下：

	libcrypto.so.1.0.0 => /usr/lib64/libcrypto.so.1.0.0 (0x00007fe258a01000)
	libz.so.1 => /lib64/libz.so.1 (0x000000318ba00000)

但是在Version information确实是not found:

	libz.so.1 (ZLIB_1.2.0) => not found
	libcrypto.so.1.0.0 (OPENSSL_1.0.0) => not found

网上搜索一下，貌似是因为版本号不对:

	# locate libz.so
	/has/common/lib/lsi/libz.so.1
	/home/opt/gcc-4.8.2.bpkg-r2/gcc-4.8.2.bpkg-r2/lib64/libz.so
	/home/opt/gcc-4.8.2.bpkg-r2/gcc-4.8.2.bpkg-r2/lib64/libz.so.1
	/home/opt/gcc-4.8.2.bpkg-r2/gcc-4.8.2.bpkg-r2/lib64/libz.so.1.2.8
	/home/work/.jumbo/lib/libz.so
	/home/work/.jumbo/lib/libz.so.1
	/home/work/.jumbo/lib/libz.so.1.2.7
	/lib64/libz.so.1
	/lib64/libz.so.1.2.3
	/usr/lib64/libz.so

系统有多个libz.so：

	# ll /lib64/libz.so.1
	lrwxrwxrwx 1 root root 13 Dec 14 17:44 /lib64/libz.so.1 -> libz.so.1.2.3
	# ll /lib64/libz.so.1.2.3
	-rwxr-xr-x 1 root root 90952 Dec  8  2011 /lib64/libz.so.1.2.3
	# ll /home/work/.jumbo/lib/libz.so.1
	lrwxrwxrwx 1 work work 13 Dec 15 17:15 /home/work/.jumbo/lib/libz.so.1 -> libz.so.1.2.7
	# ll /home/work/.jumbo/lib/libz.so.1.2.7
	-rwxr-xr-x 1 work work 93678 Dec 15 17:15 /home/work/.jumbo/lib/libz.so.1.2.7

通过设置 LD_LIBRARY_PATH，让其优先查找/home/work/.jumbo/lib目录：

	# export LD_LIBRARY_PATH=/home/work/.jumbo/lib/
	# bin/aerospike start
	error: start failed due to an error.
	/home/work/aerospike-server/bin/asd: /home/work/.jumbo/lib/libcrypto.so.1.0.0: no version information available (required by /home/work/aerospike-server/bin/asd)
	Apr 07 2017 02:58:07 GMT: WARNING (cf:misc): (hardware.c:626) no NUMA information found in /sys

现在libz已经没有问题了，但是libcrypto还是有问题：

	# ldd -v bin/asd
	bin/asd: /home/work/.jumbo/lib/libcrypto.so.1.0.0: no version information available (required by bin/asd)
		linux-vdso.so.1 =>  (0x00007fffd03ff000)
		libcrypto.so.1.0.0 => /home/work/.jumbo/lib/libcrypto.so.1.0.0 (0x00007fe4856dc000)
		libpthread.so.0 => /lib64/libpthread.so.0 (0x000000318b200000)
		librt.so.1 => /lib64/librt.so.1 (0x000000318be00000)
		libdl.so.2 => /lib64/libdl.so.2 (0x000000318aa00000)
		libz.so.1 => /home/work/.jumbo/lib/libz.so.1 (0x00007fe4854b3000)
		libm.so.6 => /lib64/libm.so.6 (0x000000318b600000)
		libstdc++.so.6 => /usr/lib64/libstdc++.so.6 (0x000000318ea00000)
		libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x000000318da00000)
		libc.so.6 => /lib64/libc.so.6 (0x000000318ae00000)
		/lib64/ld-linux-x86-64.so.2 (0x000000318a600000)

		Version information:
		bin/asd:
			libz.so.1 (ZLIB_1.2.0) => /home/work/.jumbo/lib/libz.so.1
			librt.so.1 (GLIBC_2.2.5) => /lib64/librt.so.1
			libdl.so.2 (GLIBC_2.2.5) => /lib64/libdl.so.2
			libgcc_s.so.1 (GCC_3.3) => /lib64/libgcc_s.so.1
			libgcc_s.so.1 (GCC_3.0) => /lib64/libgcc_s.so.1
			libcrypto.so.1.0.0 (OPENSSL_1.0.0) => not found
			libm.so.6 (GLIBC_2.2.5) => /lib64/libm.so.6
			libc.so.6 (GLIBC_2.7) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.8) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.6) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.3.4) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.3.2) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.3) => /lib64/libc.so.6
			libc.so.6 (GLIBC_2.2.5) => /lib64/libc.so.6
			libstdc++.so.6 (GLIBCXX_3.4.9) => /usr/lib64/libstdc++.so.6
			libstdc++.so.6 (CXXABI_1.3) => /usr/lib64/libstdc++.so.6
			libstdc++.so.6 (GLIBCXX_3.4) => /usr/lib64/libstdc++.so.6
			libpthread.so.0 (GLIBC_2.3.3) => /lib64/libpthread.so.0
			libpthread.so.0 (GLIBC_2.2.5) => /lib64/libpthread.so.0
			libpthread.so.0 (GLIBC_2.3.2) => /lib64/libpthread.so.0
			...

注意到现在libz和libcrypto都指向/home/work/.jumbo/lib目录下的so文件了。

虽然这两个的名称都是 libcrypto.so.1.0.0，但是从大写就可以看出差别很大。。:

	# ll /home/work/.jumbo/lib/libcrypto.so.1.0.0
	-r-xr-xr-x 1 work work 2605702 Apr  6 17:35 /home/work/.jumbo/lib/libcrypto.so.1.0.0
	# ll /usr/lib64/libcrypto.so.1.0.0
	-rwxr-xr-x 1 root root 1662832 May 30  2012 /usr/lib64/libcrypto.so.1.0.0

这个可以通过readelf命令查看：

	# readelf -d /usr/lib64/libcrypto.so.1.0.0

	Dynamic section at offset 0x18b510 contains 24 entries:
	  Tag        Type                         Name/Value
	 0x0000000000000001 (NEEDED)             Shared library: [libdl.so.2]
	 0x0000000000000001 (NEEDED)             Shared library: [libz.so.1]
	 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
	 0x000000000000000e (SONAME)             Library soname: [libcrypto.so.10]
	 0x0000000000000010 (SYMBOLIC)           0x0
	 0x000000000000000c (INIT)               0x5c2e0
	 0x000000000000000d (FINI)               0x123dd8
	 0x000000006ffffef5 (GNU_HASH)           0x1f0
	 0x0000000000000005 (STRTAB)             0x1d288
	 0x0000000000000006 (SYMTAB)             0x6c08
	 0x000000000000000a (STRSZ)              70816 (bytes)
	 0x000000000000000b (SYMENT)             24 (bytes)
	 0x0000000000000003 (PLTGOT)             0x38bfe8
	 0x0000000000000002 (PLTRELSZ)           2616 (bytes)
	 0x0000000000000014 (PLTREL)             RELA
	 0x0000000000000017 (JMPREL)             0x5b8a8
	 0x0000000000000007 (RELA)               0x30588
	 0x0000000000000008 (RELASZ)             176928 (bytes)
	 0x0000000000000009 (RELAENT)            24 (bytes)
	 0x000000006ffffffe (VERNEED)            0x30508
	 0x000000006fffffff (VERNEEDNUM)         2
	 0x000000006ffffff0 (VERSYM)             0x2e728
	 0x000000006ffffff9 (RELACOUNT)          7361
	 0x0000000000000000 (NULL)               0x0

	# readelf -d /home/work/.jumbo/lib/libcrypto.so.1.0.0

	Dynamic section at offset 0x22aed0 contains 25 entries:
	  Tag        Type                         Name/Value
	 0x0000000000000001 (NEEDED)             Shared library: [libdl.so.2]
	 0x0000000000000001 (NEEDED)             Shared library: [libz.so.1]
	 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
	 0x000000000000000e (SONAME)             Library soname: [libcrypto.so.1.0.0]
	 0x0000000000000010 (SYMBOLIC)           0x0
	 0x000000000000000f (RPATH)              Library rpath: [/home/work/.jumbo/lib]
	 0x000000000000000c (INIT)               0x6a410
	 0x000000000000000d (FINI)               0x18e228
	 0x000000006ffffef5 (GNU_HASH)           0x1b8
	 0x0000000000000005 (STRTAB)             0x22420
	 0x0000000000000006 (SYMTAB)             0x92f8
	 0x000000000000000a (STRSZ)              80514 (bytes)
	 0x000000000000000b (SYMENT)             24 (bytes)
	 0x0000000000000003 (PLTGOT)             0x42bb10
	 0x0000000000000002 (PLTRELSZ)           2496 (bytes)
	 0x0000000000000014 (PLTREL)             RELA
	 0x0000000000000017 (JMPREL)             0x69a50
	 0x0000000000000007 (RELA)               0x38070
	 0x0000000000000008 (RELASZ)             203232 (bytes)
	 0x0000000000000009 (RELAENT)            24 (bytes)
	 0x000000006ffffffe (VERNEED)            0x38010
	 0x000000006fffffff (VERNEEDNUM)         2
	 0x000000006ffffff0 (VERSYM)             0x35ea2
	 0x000000006ffffff9 (RELACOUNT)          8457
	 0x0000000000000000 (NULL)               0x0

对于/usr/lib64/libcrypto.so.1.0.0：

	 0x000000000000000e (SONAME)             Library soname: [libcrypto.so.10]

而对于 /home/work/.jumbo/lib/libcrypto.so.1.0.0:

	 0x000000000000000e (SONAME)             Library soname: [libcrypto.so.1.0.0]

一个是10一个是1.0.0。。

那么Aerospike到底需要哪个版本呢？

	# readelf -a  bin/asd  | grep libcrypto.so -C 2
	Dynamic section at offset 0x2bf488 contains 33 entries:
	  Tag        Type                         Name/Value
	 0x0000000000000001 (NEEDED)             Shared library: [libcrypto.so.1.0.0]
	 0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
	 0x0000000000000001 (NEEDED)             Shared library: [librt.so.1]
	--
	  0x0070:   Name: GCC_3.3  Flags: none  Version: 21
	  0x0080:   Name: GCC_3.0  Flags: none  Version: 9
	  0x0090: Version: 1  File: libcrypto.so.1.0.0  Cnt: 1
	  0x00a0:   Name: OPENSSL_1.0.0  Flags: none  Version: 8
	  0x00b0: Version: 1  File: libm.so.6  Cnt: 1	

看起来它需要的是libcrypto.so.1.0.0，版本被OPENSSL_1.0.0依赖。

	# openssl version
	OpenSSL 1.0.0-fips 29 Mar 2010

看起来jumbo下的版本是正确的。。为什么不行呢。。


