---
title: RocksDB学习笔记
layout: post
catalog: true
---


简介
---


* Facebook
* code from leveldb(1.5)
* ideas from HBase
 exploit the full potential of high read/write rates offered by Flash or RAM subsystems
support efficient point lookups as well as range scans
performant for fast storage and for server workloads

RocksDB基于Google的LevelDB，但提高了扩展性可以运行在多核处理器上，可以有效使用快速存储，支持IO绑定、内存和一次写负荷。
In RocksDB 3.0, we added support for Column Families.


* LSM
* WAL

Get(key), Put(key), Delete(key) and Scan(key).



架构
---

RocksDB有三部分组成：

* memtable: is an in-memory data structure - new writes are inserted into the memtable and are optionally written to the logfile. 
* ssttable: SST stands for Sorted Sequence Table. They are persistent files storing data. In the file keys are usually organized in sorted order so that a key or iterating position can be identifies through a binary search.
* logfile: When the memtable fills up, it is flushed to a sstfile on storage and the corresponding logfile can be safely deleted. 



