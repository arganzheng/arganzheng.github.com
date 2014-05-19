---
title: 使用rsync进行文件同步
layout: post
---


以前一直用scp，但是rsync更强大。最常见的应用场景是需要将开发机的代码发布到测试机上编译和运行，确认没有问题之后再提交svn。以前的standalone发布程序做的其实就是这样的事情，里面的核心发布函数就是利用了rsync：

    Release(){
        echo_red `nchar = 25` "[$APP]:Press any key to release on $CONTAINER, or Ctrl+c to quit !" `nchar = 25`
        read _xx
        ssh $CONTAINER "mkdir -p $TARGETDEPLOYDIR"
        rsync -avz --delete $SOURCEDEPLOYDIR/ $CONTAINER:$TARGETDEPLOYDIR/
        ssh $CONTAINER chmod +x $TARGETDEPLOYDIR/bin/*
        number_of_script_under_deploy=`ssh $CONTAINER "ls $TARGETDEPLOYDIR|egrep *\.sh|wc -l"`
        if [ $number_of_script_under_deploy -gt 0 ];then
                ssh $CONTAINER chmod +x $TARGETDEPLOYDIR/*.sh
        fi
        if [ -n "$TARGETTEMPLATESDIR" ];then
                rsync -avz --delete $SOURCETEMPLATESDIR/ $CONTAINER:$TARGETTEMPLATESDIR/
        fi
        if [ -n "$TARGETRESOURCESDIR" ];then
                rsync -avz --delete $SOURCERESOURCESDIR/ $CONTAINER:$TARGETRESOURCESDIR/
        fi
        echo_red `nchar = 35` "[$APP]:Release finished on $CONTAINER "`nchar = 35`
    }

--detete参数表示先删除目标目录再同步过去。带有detele参数最好先加上--dry-run测试一下。另外，delete删除永远的是destDir，并且是destDir与srcDir不同的部分（这就是rsync名称的来源吧:)）事实上，rsync的机制就是比较srcDir和destDir的差异（两边都存在的文件通过比较修改时间）然后进行相应的同步。如果srcDir与destDir是一样的，那么执行rsync即使加上--delete参数也不会有实质性动作。这真是它高效的地方。
另外几个有用的选项是--include和--exclude。
Tips:如果仅仅想发上去运行，那么将deploy目录发布上去就可以了，然后上去autoconfig（或者直接vim target中的tar包）。

另外，ssh可以远程执行：ssh user@remote "rm /home/user/foo.txt"


补充：more about rsync
----------------------

### 1. Trailing Slashes Do Matter...Sometimes

 rsync对SourceDir最后是否带斜杠有不同的处理。例如：

    rsync -auv /home/work/STATIC/mbrowser/guanxing  /home/work/mnt/mfs/mbrowser 

将产生 备份到 /home/work/mnt/mfs/mbrowser/guanxing 目录

而

    rsync -auv /home/work/STATIC/mbrowser/guanxing/  /home/work/mnt/mfs/mbrowser 

将产生 备份到 /home/work/mnt/mfs/mbrowser 目录


DestDir 最后有没有斜杠没有影响。


### 2. Using the --delete flag

If a file was originally in both source/ and destination/ (from an earlier rsync, for example), and you delete it from source/, you probably want it to be deleted from destination/ on the next rsync. However, the default behavior is to leave the copy at destination/ in place. Assuming you want rsync to delete any file from destination/ that is not in source/, you'll need to use the --delete flag:

    rsync -a --delete source/ destination/


--delete-after表示先同步再删除。


参考文章
--------

1. [Easy Automated Snapshot-Style Backups with Linux and Rsync](http://www.mikerubel.org/computers/rsync_snapshots/)


