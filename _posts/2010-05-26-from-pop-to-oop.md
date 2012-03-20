---
layout: post
title: 从面向过程到面向对象——在C中如何实现面向对象编程
---

#### 引言

像我们这些80后的童鞋，一般都是从C入门，然后转到C++和Java这些面向对象语言。当习惯了OOP的思想后就会开始思考，OOP是怎么来的。为什么我们觉得从面向过程到面向对象是一种自然的进化呢？语言就是一种工具，是应需要产生的，所以必然有这样的需求，才有这样的结果。笔者在阅读开源C源码的过程发现了很多OOP的特征，因此推测OOP就是一种自然的进化。

回顾OOP的三大特性：封装，继承和多态。现在让我们分别研究怎样在C中实现这些OOP特性。

#### 1. 实现OOP中的封装性（不包括成员权限控制）

OOP中所谓的封装，其实就是将数据与操作数据的函数绑定在一起成为一个类。其中类中的数据称为数据成员，类中的函数称为方法。这个很简单，只需要在C结构中存放函数指针即可。一般两种存放方式。见下图所示：

![encapsulation](/media/images/encapsulation.jpg "encapsulation in C")

#### 2. 实现OOP中的继承（不包括类权限继承）

OOP中的继承，其实就是将各个类中通用的数据和方法放在基类中，当子类继承自该父类时，就自动拥有该父类定义数据和方法。

因此我们可以将子类中存放一个指向父类（通用类）的指针，然后在构建子类的时候先构建父类并将父类的地址存放在子类的这个指针中，这样子类就可以通过这个指针访问父类的数据和方法了。另一种方式是直接嵌入父类结构，这个更常见。具体例子如下所示：

父类：
    struct inode {
        struct hlist_node   i_hash;
        struct list_head    i_list;
        struct list_head    i_sb_list;
        struct list_head    i_dentry;
        unsigned long       i_ino;
        atomic_t        i_count;
        unsigned int        i_nlink;
        uid_t           i_uid;
        gid_t           i_gid;
        dev_t           i_rdev;
        u64         i_version;
        loff_t          i_size;
    #ifdef __NEED_I_SIZE_ORDERED
        seqcount_t      i_size_seqcount;
    #endif
        struct timespec     i_atime;
        struct timespec     i_mtime;
        struct timespec     i_ctime;
        unsigned int        i_blkbits;
        blkcnt_t        i_blocks;
        unsigned short          i_bytes;
        umode_t         i_mode;
        spinlock_t      i_lock; /* i_blocks, i_bytes, maybe i_size */
        struct mutex        i_mutex;
        struct rw_semaphore i_alloc_sem;
        const struct inode_operations   *i_op;
        const struct file_operations    *i_fop; /* former ->i_op->default_file_ops */
        struct super_block  *i_sb;
        struct file_lock    *i_flock;
        struct address_space    *i_mapping;
        struct address_space    i_data;
    #ifdef CONFIG_QUOTA
        struct dquot        *i_dquot[MAXQUOTAS];
    #endif
        struct list_head    i_devices;
        union {
            struct pipe_inode_info  *i_pipe;
            struct block_device *i_bdev;
            struct cdev     *i_cdev;
        };
        int         i_cindex;

        __u32           i_generation;

    #ifdef CONFIG_DNOTIFY
        unsigned long       i_dnotify_mask; /* Directory notify events */
        struct dnotify_struct   *i_dnotify; /* for directory notifications */
    #endif

    #ifdef CONFIG_INOTIFY
        struct list_head    inotify_watches; /* watches on this inode */
        struct mutex        inotify_mutex;  /* protects the watches list */
    #endif

        unsigned long       i_state;
        unsigned long       dirtied_when;   /* jiffies of first dirtying */

        unsigned int        i_flags;

        atomic_t        i_writecount;
    #ifdef CONFIG_SECURITY
        void            *i_security;
    #endif
        void            *i_private; /* fs or device private pointer */
    }; 

子类：

    /*
    * third extended file system inode data in memory
    */
    struct ext3_inode_info {
        __le32  i_data[15]; /* unconverted */
        __u32   i_flags;
    #ifdef EXT3_FRAGMENTS
        __u32   i_faddr;
        __u8    i_frag_no;
        __u8    i_frag_size;
    #endif
        ext3_fsblk_t    i_file_acl;
        __u32   i_dir_acl;
        __u32   i_dtime;

        /*
        * i_block_group is the number of the block group which contains
        * this file's inode.  Constant across the lifetime of the inode,
        * it is ued for making block allocation decisions - we try to
        * place a file's data blocks near its inode block, and new inodes
        * near to their parent directory's inode.
        */
        __u32   i_block_group;
        __u32   i_state;        /* Dynamic state flags for ext3 */

        /* block reservation info */
        struct ext3_block_alloc_info *i_block_alloc_info;

        __u32   i_dir_start_lookup;
    #ifdef CONFIG_EXT3_FS_XATTR
        /*
        * Extended attributes can be read independently of the main file
        * data. Taking i_mutex even when reading would cause contention
        * between readers of EAs and writers of regular file data, so
        * instead we synchronize on xattr_sem when reading or changing
        * EAs.
        */
        struct rw_semaphore xattr_sem;
    #endif
    #ifdef CONFIG_EXT3_FS_POSIX_ACL
        struct posix_acl    *i_acl;
        struct posix_acl    *i_default_acl;
    #endif

        struct list_head i_orphan;  /* unlinked but open inodes */

        loff_t  i_disksize;

        /* on-disk additional length */
        __u16 i_extra_isize;

        struct mutex truncate_mutex;

        // 父类数据成员
        struct inode vfs_inode;
    }; 

说明：在VFS中，或者说在C中并不能做到严格的面向对象编程，尤其是在处理通用类和具体类数据问题上。这是因为虽然我们可以通过在将子类的实现函数指针存放在基类的函数指针中，但是这些函数操作的对象却是基类而不是具体子类。为了使这些函数可以通过基类指针访问具体子类数据，VFS通过在基类中定义一个指向子类的指针（为了与子类解耦，采用void指针），让子类函数通过它访问自己的数据。具体例子如下：

> The s_fs_info field points to superblock information that belongs to a specific filesystem; for instance, as we'll see later in Chapter 18, if the superblock object refers to an Ext2 filesystem, the field points to an ext2_sb_info structure, which includes the disk allocation bit masks and other data of no concern to the VFS common file model.

对于EXT2 Superblock对象：
> the s_fs_info field of the VFS superblock points to a structure containing filesystem-specific data. In the case of Ext2, this field points to a structure of type ext2_sb_info{}.

    struct super_block {
        struct list_head        s_list;            /* list of all superblocks */
        dev_t                   s_dev;             /* identifier */
        unsigned long           s_blocksize;       /* block size in bytes */
        unsigned long           s_old_blocksize;   /* old block size in bytes */
        unsigned char           s_blocksize_bits;  /* block size in bits */
        unsigned char           s_dirt;            /* dirty flag */
        unsigned long long      s_maxbytes;        /* max file size */
        struct file_system_type s_type;            /* filesystem type */
        struct super_operations s_op;              /* superblock methods */
        struct dquot_operations *dq_op;            /* quota methods */
        struct quotactl_ops     *s_qcop;           /* quota control methods */
        struct export_operations *s_export_op;     /* export methods */
        unsigned long            s_flags;          /* mount flags */
        unsigned long            s_magic;          /* filesystem's magic number */
        struct dentry            *s_root;          /* directory mount point */
        struct rw_semaphore      s_umount;         /* unmount semaphore */
        struct semaphore         s_lock;           /* superblock semaphore */
        int                      s_count;          /* superblock ref count */
        int                      s_syncing;        /* filesystem syncing flag */
        int                      s_need_sync_fs;   /* not-yet-synced flag */
        atomic_t                 s_active;         /* active reference count */
        void                     *s_security;      /* security module */
        struct list_head         s_dirty;          /* list of dirty inodes */
        struct list_head         s_io;             /* list of writebacks */
        struct hlist_head        s_anon;           /* anonymous dentries */
        struct list_head         s_files;          /* list of assigned files */
        struct block_device      *s_bdev;          /* associated block device */
        struct list_head         s_instances;      /* instances of this fs */
        struct quota_info        s_dquot;          /* quota-specific options */
        char                     s_id[32];         /* text name */

        // 指向具体的文件系统（子类）
        void                     *s_fs_info;       /* filesystem-specific info*/

        struct semaphore         s_vfs_rename_sem; /* rename semaphore */
    };

另外，继承不仅包括继承数据成员，还包括继承基类的方法，像VFS中就有很多generic_XXX方法。关于多态性见下面讨论。

一个体现C中组合与继承很好的例子就是Linux的Block IO（我们会在下面做详细介绍）Linux Block Device Architecture （Author：Ravi Kiran UVS，非常精彩！！！http://www.geocities.com/ravikiran_uvs/articles/blkdevarch.html）。下面简单描述一下。

##### 父类：gendisk{}

This stores the information about a disk. The important fields are queue, part and fops used to store the request queue, partition information and the block device operations table respectively. The part field points to an array of pointers to hd_structs each representing a partition.

This device driver has to allocate the gendisk structure, load the partition table, allocate the request queue and fill the other fields in the gendisk structure.

    struct gendisk {
        request_queue_t             *queue;
        struct hd_struct                **part;
        struct block_device_operations   *fops;
        ...
    };

##### hd_struct{}

This stores the information about a partition on a disk.

    struct hd_struct {
        sector_t         start_sector;
        sector_t         nr_sects;
        int              partno;
        ...
    };

##### block_device{}

This is used to represent a block device in the kernel. This can represent the entire disk or a particular partition. When the structure represents a partition, the bd_contains field points to the device object which contains the partition. The bd_part field points to the partition structure of the device. In the structure representing the device, the field bd_disk points to the gendisk structure of the device.

This structure is created only when the device is opened. Note that the device driver still creates the gendisk structure, allocates the request queue and registers the structures with the kernel. But till the device is actually opened (either by reading through a device file or by mounting it), this structure will not be created.

The field bd_inode points to the inode in the bdev file system. The device will be accessed as a block device type file. That inode that represents that device file is stored in the bd_inodes list.

When the device file is opened for the first time, the kernel allocates the block_device structure and fills the structures. This is actually allocated by the function bdev_alloc_inode. This is called when the inode in the bdev file system has to be allocated. This allocates the space required for both the structures in a single buffer.

    struct block_device {
        dev_t                    bd_bdev;
        struct inode            *bd_inode;
        struct list_head         bd_inodes;
        struct block_device     *bd_contains;
        struct hd_struct        *bd_part;
        struct gendisk          *bd_disk;
        struct list_head         bd_list;
        struct backing_dev_info *bd_inode_backing_dev_info;
        ...
    };

![blockSubsystem](/media/images/blockSubsystem.jpg "blockSubsystem")

《UTLK, 3th》Figure 14-3. Linking the block device descriptors with the other structures of the block subsystem

说明：这种通过组合的方式实现继承其实在OOP中也很常见，整个设计模式的核心其实就在于组合。

#### 3. 实现OOP中的多态性

OOP中的多态的表现就是通过一个通用类指针（或引用）在运行时根据实际类型自动调用子类的方法。因此，如果我们能够动态确定要调用哪个函数，并将其地址设置在一个通用类中一个固定的指针字段上，那么我们就可以达到通过这个固定指针字段调用不同函数的效果。实际上这也是C++中多态性的实现方法（通过一个指针指向一个虚函数表，并在创建时候根据子类类型动态填充该虚函数表）。一个很好的比喻就是《阿甘正传》的最经典的一句话："Life is a box of chocolate, you never know what you're gonna to get!"。Chocolate就是那个固定的函数指针字段，而根据取出来的具体的巧克力将有不同的味道（方法，行为）。

注意：C中实现多态有个缺点，就是通过指向基类的指针调用子类的函数并不能处理子类的数据。而必须在基类中设置一个指针指向子类（一般为void *指针）。这在前面第二点"实现OOP中的继承"中就有谈到了。这里再举个例子；

    struct inode{
        /* 2.4内核中 */
        union {
            struct ext2_inode_info  ext2_i;
            struct ext3_inode_info  ext3_i;
            ...
                void        *generic_ip;
        } u;
        /* 2.6 内核中 */
        void    *i_private; /* fs or device private pointer */
    }

    //2.4内核ext2_delete_inode实现
    /*
    * Called at the last iput() if i_nlink is zero.
    */
    void ext2_delete_inode (struct inode * inode)
    {
        lock_kernel();

        if (is_bad_inode(inode) ||
                inode->i_ino == EXT2_ACL_IDX_INO ||
                inode->i_ino == EXT2_ACL_DATA_INO)
            goto no_delete;

        // 在基类中使用子类的数据成员
        inode->u.ext2_i.i_dtime = CURRENT_TIME;

        mark_inode_dirty(inode);
        ext2_update_inode(inode, IS_SYNC(inode));
        inode->i_size = 0;
        if (inode->i_blocks)
            ext2_truncate (inode);
        ext2_free_inode (inode);

        unlock_kernel();
        return;
    no_delete:
        unlock_kernel();
        clear_inode(inode); 
    }

    // 2.6内核ext2_delete_inode实现
    void ext2_delete_inode (struct inode * inode)
    {
        truncate_inode_pages(&inode->i_data, 0);

        if (is_bad_inode(inode))
            goto no_delete;

        // 从基类中抽取子类
        EXT2_I(inode)->i_dtime  = get_seconds();

        mark_inode_dirty(inode);
        ext2_update_inode(inode, inode_needs_sync(inode));

        inode->i_size = 0;
        if (inode->i_blocks)
            ext2_truncate (inode);
        ext2_free_inode (inode);

        return;
    no_delete:
        clear_inode(inode);     
    }

其中EXT2_I()函数是定义在ext2.h头文件中的静态内联函数：

    static inline struct ext2_inode_info *EXT2_I(struct inode *inode)
    {
        return container_of(inode, struct ext2_inode_info, vfs_inode);
    }

说明：这是因为采用子类采用执行父类结构的指针来实现继承而不是通过嵌入父类结构实现继承。在UNIX中，一般是在子类的结构体中第一个声明父类结构（注意，不是指针），这样可以直接将一个结构覆盖在一个内存区域上实现类似与OOP中的强制类型转换。

#### 4. 其他OOP特性

至此，我们已经讨论完三大OOP特性，其实其他的OOP特性在C源码中也大量见到。

##### 4.1 在C中实现面向对象的异常处理机制

在面向对象中有try-catch-finally异常处理机制，但是在C中没有。但我们可以使用如下备受责备的goto实现类似的功能。
例子：假设有如下Java代码：

    try{
        //try to do some works
        If ExceptionA encounter, throw new ExceptionA(...);
        If ExceptionB encounter, throw new ExceptionB(...);
        ...
    }catch(ExceptionA ex){
        //deal with ExceptionA
    }catch(ExceptionB ex){
        //deal with ExceptionB
    }finally{
        //release resources
    }

等价的C代码如下所示：

    /* try to do some works */
    If ExceptionA encounter, goto ExceptionA;
    If ExceptionB encounter, goto ExceptionB;
    ...
    goto Finally; 

    ExceptionA:
    /* deal with ExceptionA */
    goto Finally;

    ExceptionB:
    /* deal with ExceptionB */
    //goto finally;

    Finally:
    /* release resources */

另一种避免太多goto 标记（label）的方法是使用一个status记录错误代码（error code），统一跳到ERROR标号处，再根据错误代码分情况处理：

    /* try to do some works */
    If ExceptionA encounter{
    errcode=ExceptionA;
    goto ERROR;
    }
    If ExceptionB encounter{
    errcode=ExceptionB;
    goto ERROR;
    }
    ...
    goto Finally; 

    ERROR:
    switch(errcode){
    case ExceptionA:
    /* deal with ExceptionA */
    goto Finally;

    case ExceptionB:
    /* deal with ExceptionB */
    //goto finally;
    default：...
    }
    ...
    Finally:
    /* release resources */

注：很多C开源代码都是采用这种做法，如TCP/IP，Berkeley DB等。不难想象面向对象的出现是一种很自然的事情。一个实际的例子如下：

    /*Note: 
    You will notice that every database operation checks the operation's status return code, and if an error (non-zero) status is returned, we log the error and then go to a err label in our program. Unlike object-oriented programs such as C++ or Java, we do not have try blocks in C. Therefore, this is the best way for us to implement cascading error handling for this example. 
    */
    /*
    * Indicate that we want db to perform lock detection internally.
    * Also indicate that the transaction with the fewest number of
    * write locks will receive the deadlock notification in 
    * the event of a deadlock.
    */  
    ret = envp->set_lk_detect(envp, DB_LOCK_MINWRITE);
    if (ret != 0) {
        fprintf(stderr, "Error setting lock detect: %s\n",
                db_strerror(ret));
        goto err;
    } 
    Now we open our environment. 
    /*
    * If we had utility threads (for running checkpoints or 
    * deadlock detection, for example) we would spawn those
    * here. However, for a simple example such as this,
    * that is not required.
    */

    /* Now actually open the environment */
    ret = envp->open(envp, db_home_dir, env_flags, 0);
    if (ret != 0) {
        fprintf(stderr, "Error opening environment: %s\n",
                db_strerror(ret));
        goto err;
    } 
        Now we call the function that will open our database for us. This is not very interesting, except that you will notice that we are specifying DB_DUPSORT. This is required purely by the data that we are writing to the database, and it is only necessary if you run the application more than once without first deleting the environment. 
    Also, we do not provide any error logging here because the open_db() function does that for us. (The implementation of open_db() is described later in this section.) 
        /* Open the database */
        ret = open_db(&dbp, prog_name, file_name, envp, DB_DUPSORT);
    if (ret != 0)
        goto err;  
        Finally, to wrap up main(), we close out our database and environment handle, as is normal for any DB application. Notice that this is where our err label is placed in our application. If any database operation prior to this point in the program returns an error status, the program simply jumps to this point and closes our handles if necessary before exiting the application completely. 
        err:
        /* Close our database handle, if it was opened. */
        if (dbp != NULL) {
            ret_t = dbp->close(dbp, 0);
            if (ret_t != 0) {
                fprintf(stderr, "%s database close failed: %s\n",
                        file_name, db_strerror(ret_t));
                ret = ret_t;
            }
        }

    /* Close our environment, if it was opened. */
    if (envp != NULL) {
        ret_t = envp->close(envp, 0);
        if (ret_t != 0) {
            fprintf(stderr, "environment close failed: %s\n",
                    db_strerror(ret_t));
            ret = ret_t;
        }
    }

    /* Final status message and return. */
    printf("I'm all done.\n");
    return (ret == 0 ? EXIT_SUCCESS : EXIT_FAILURE);

#### 5. 高级OOP特性：在C中实现泛型编程（如何实现模板）

我们可以通过宏拓展来实现模板。一个具体的实例就是Berkeley DB中关于链表的定义：

    /* include/queue.h */
    /*
    * List definitions.
    */
    #define LIST_HEAD(name, type)                       \
    struct name {                               \
        struct type *lh_first;  /* first element */         \
    }

    #define LIST_ENTRY(type)                        \
    struct {                                \
        struct type *le_next;   /* next element */          \
        struct type **le_prev;  /* address of previous next element */  \
    }

#### 小结

从上面可以看出很多OOP特性其实在很多C开源项目中早有影子，当这些设计模式成为一种需要的时候，必然导致程序员开始思考能不能将这种思想从应用级别抽取出来，放在编译器层面实现，于是自然而然的就有了面向对象语言了。

#### 补记：关于C中的继承

今天早上看了《代码之美》中的第16章：Linux内核驱动模型：协作的好处。里面对C中实现OOP继承有非常好的例子和描述，并且从演变的过程讲述了linux内核驱动模型中的几个重要的数据对象（device, kobject, kref）是怎么来的。

演变一：device>>xxx_device

首先是struct device，这个结构被设计为内核中所有设备的"基"类。
    struct device{
        struct list_head node; //  brother
        struct list_head children;
        struct device *parent;
        char name[DEVICE_NAME_SIZE]; // 描述性ascii字符串
        char bus_id[BUS_ID_SIZE]; // 在父总线上的位置
        spinlock_t lock; 
        atomic_t refcount; 
        struct driver_dir_entry *dir;
        struct device_driver *driver; // 哪一个驱动分配了这台设备
        void *driver_data; // 设备的私有数据
        void *platform_data; // 特定平台的数据（如ACPI，设备相关的BIOS数据）
        u32 current_state; // 当前的操作状态，在ACPI中为D0-D3,D0开启全部功能，D3关闭

        unsigned char *saved_state; //保存的设备状态
    };

为使用这个结构体，你需要把它嵌入到另一个结构体中，从而在某种意义上，使新的结构体"继承"自上面的"基"结构体。如struct usb_interface

    struct usb_interface{
        struct usb_interface_descriptor *altsetting;
        
        int act_altsetting;  // 活动备用设置
        int num_altsetting;  // 备用设置数目
        int max_altsetting;  // 总内存分配
        struct usb_driver *driver; // 驱动
        struct device dev; // 特定接口设备信息
    };

内核驱动模块通过在不同的代码之间传递执行device结构的指针来运作，它操作结构体中的基本字段，这些字段在所有类型的设备中都存在。当指针被传递给实现某种功能的总线特定代码(bus-specific code)时，它就需要被转换成包含它的实际结构体的类型。为了实现这种转换，总线特定代码基于指针在内存中的位置把它转换成原来的结构体类型。这是通过以下这个宏来实现的：

    #define container_of(ptr, type, member) ({ \
        const typeof( ((type *)0)->member ) *__mptr = (ptr); \
        (type *)( (char *)__mptr - offsetof(type, member) ); })

通过以上代码，前面的struct usb_interface可以将一个指向其数据成员struct device的成员指针转换回原来的指针：

    int probe(struct device *d){
        struct usb_interface *intf;

        intf = container_of(d, struct usb_interface, dev);

        ...
    }

其中d是一个指向struct device的指针。
将上面的container_of宏展开，将会产生如下代码：

    intf = ({
            const typeof( ((struct usb_interface *)0)->dev ) *__mptr = (d);
                (struct usb_interface *)( (char *)__mptr - offsetof(struct usb_interface, dev) ); 
    });

要理解这段代码，就要记着dev是struct usb_interface的成员。

通过这种非常简单的方法，Linux内核允许常规的C结构体被"继承"并通过强有力的方法来操纵。通过这种"继承"基本struct device的方法，在2.5版内核的开发过程中，所有的不同的驱动子系统都被统一处理了。现在他们共享公共的核心代码，从而使内核能把所有设备间的连接关系显示给用户。这也促使了一些有用工具的出现，如udev。

该文章后面还介绍了kobject和kref的由来。这里就不一一介绍了，有兴趣的同学可以自己去看看。

说明：在UNIX中，一般是在子类的结构体中第一个声明父类结构（注意，不是指针），这样可以直接将一个结构覆盖在一个内存区域上实现类似与OOP中的强制类型转换。上面的usb_interface没有将父类声明为第一个数据成员，所以采用了container_of宏来从父类指针获取子类地址。

在老的unix系统中还有通过在父类中增加类型标志符来帮助子类函数实现类型转换。如所有socket_addr{}结构就是前两个字段是int family和char length。特定函数可以通过family实现父类往具体子类的转换（在具有元数据编程的OOP中，如Java，可以直接通过instance of操作符判断子类类型），而通过length可以实现变长结构。


