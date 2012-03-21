---
layout: post
title: BDB中的共享区域
---

2009-5-23 星期六 雨天

#### 1. 数据结构

###### 环境数据结构 DB_ENV{}

    struct __db_env {
        int         db_lorder;    /* Byte order. */

                        /* Error message callback. */
        void (*db_errcall) __P((const char *, char *));
        FILE        *db_errfile;    /* Error message file stream. */
        const char    *db_errpfx;    /* Error message prefix. */
        int         db_verbose;    /* Generate debugging messages. */
        int         db_panic;    /* Panic flag, callback function. */
        void (*db_paniccall) __P((DB_ENV *, int));

        /* User paths. */
        char        *db_home;    /* Database home. */
        char        *db_log_dir;    /* Database log file directory. */
        char        *db_tmp_dir;    /* Database tmp file directory. */

        char           **db_data_dir;    /* Database data file directories. */
        int         data_cnt;    /* Database data file slots. */
        int         data_next;    /* Next Database data file slot. */

        /* Locking. */
        DB_LOCKTAB    *lk_info;    /* Return from lock_open(). */
        const u_int8_t    *lk_conflicts;    /* Two dimensional conflict matrix. */
        u_int32_t     lk_modes;    /* Number of lock modes in table. */
        u_int32_t     lk_max;    /* Maximum number of locks. */
        u_int32_t     lk_detect;    /* Deadlock detect on all conflicts. */

        /* Logging. */
        DB_LOG        *lg_info;    /* Return from log_open(). */
        u_int32_t     lg_max;    /* Maximum file size. */

        /* Memory pool. */
        DB_MPOOL    *mp_info;    /* Return from memp_open(). */
        size_t         mp_mmapsize;    /* Maximum file size for mmap. */
        size_t         mp_size;    /* Bytes in the mpool cache. */

        /* Transactions. */
        DB_TXNMGR    *tx_info;    /* Return from txn_open(). */
        u_int32_t     tx_max;    /* Maximum number of transactions. */
        int (*tx_recover)        /* Dispatch function for recovery. */
            __P((DB_LOG *, DBT *, DB_LSN *, int, void *));

        /*
        * XA support.
        *
        * !!!
        * Explicit representations of structures in queue.h.
        *
        * TAILQ_ENTRY(__db_env);
        */
        struct {
            struct __db_env *tqe_next;
            struct __db_env **tqe_prev;
        } links;
        int         xa_rmid;    /* XA Resource Manager ID. */
        DB_TXN        *xa_txn;    /* XA Current transaction. */

    #define    DB_ENV_APPINIT        0x01    /* Paths initialized by db_appinit(). */
    #define    DB_ENV_CDB        0x02    /* Concurrent DB product. */
    #define    DB_ENV_STANDALONE    0x04    /* Test: freestanding environment. */
    #define    DB_ENV_THREAD        0x08    /* DB_ENV is multi-threaded. */
        u_int32_t     flags;        /* Flags. */
    };

###### RLAYOUT{} 数据结构

RLAYOUT{} 是所有共享区域共用的数据结构，它相当于共享区域的描述符（或者头部）。

    /*******************************************************
    * Shared memory regions.
    *******************************************************/
    /*
    * The shared memory regions share an initial structure so that the general
    * region code can handle races between the region being deleted and other
    * processes waiting on the region mutex.
    *
    * !!!
    * Note, the mutex must be the first entry in the region; see comment above.
    */
    typedef struct _rlayout {
        db_mutex_t lock;        /* Region mutex. */
    #define    DB_REGIONMAGIC    0x120897
        u_int32_t  valid;        /* Valid magic number. */
        u_int32_t  refcnt;        /* Region reference count. */
        size_t       size;        /* Region length. */
        int       majver;        /* Major version number. */
        int       minver;        /* Minor version number. */
        int       patch;        /* Patch version number. */
        int       panic;        /* Region is dead. */
    #define    INVALID_SEGID    -1
        int       segid;        /* shmget(2) ID, or Win16 segment ID. */

    #define    REGION_ANONYMOUS    0x01    /* Region is/should be in anon mem. */
        u_int32_t  flags;
    } RLAYOUT;


每个模块有自己定义的每进程私有和所有进程共用的共享区域数据结构：

###### 封锁管理器

每进程私有的共享区域数据结构：__db_locktab{}

    /*
    * The lock table is the per-process cookie returned from a lock_open call.
    */
    struct __db_locktab {
        DB_ENV        *dbenv;        /* Environment. */
        REGINFO         reginfo;    /* Region information. */
        DB_LOCKREGION    *region;    /* Address of shared memory region. */
        DB_HASHTAB     *hashtab;     /* Beginning of hash table. */
        void        *mem;        /* Beginning of string space. */
        u_int8_t     *conflicts;    /* Pointer to conflict matrix. */
    };

其中REGINFO{}数据结构是每进程私有的数据结构，用于描述共享内存区域在改进程的地址空间的映射情况，比如映射地址。它既是充当参数也是返回值。
这点我们从注释可以看出（前半部分是Arguments，后半部分是Results）：

REGINFO{}数据结构 

    /*
    * The interface to region attach is nasty, there is a lot of complex stuff
    * going on, which has to be eetained between create/attach and detach. The
    * REGINFO structure keeps track of it.
    */
    struct __db_reginfo; typedef struct __db_reginfo REGINFO;
    struct __db_reginfo {
    /* Arguments. */
    DB_ENV *dbenv; /* Region naming info. */
    APPNAME appname; /* Region naming info. */
    char *path; /* Region naming info. */
    const char *file; /* Region naming info. */
    int mode; /* Region mode, if a file. */
    size_t size; /* Region size. */
    u_int32_t dbflags; /* Region file open flags, if a file. */

    /* Results. */
    char *name; /* Region name. */
    void *addr; /* Region address. */
    int fd; /* Fcntl(2) locking file descriptor.
    NB: this is only valid if a regular
    file is backing the shared region,
    and mmap(2) is being used to map it
    into our address space. */
    int segid; /* shmget(2) ID, or Win16 segment ID. */
    void *wnt_handle; /* Win/NT HANDLE. */

    /* Shared flags. */
    /* 0x0001 COMMON MASK with RLAYOUT structure. */
    #define REGION_CANGROW 0x0002 /* Can grow. */
    #define REGION_CREATED 0x0004 /* Created. */
    #define REGION_HOLDINGSYS 0x0008 /* Holding system resources. */
    #define REGION_LASTDETACH 0x0010 /* Delete on last detach. */
    #define REGION_MALLOC 0x0020 /* Created in malloc'd memory. */
    #define REGION_PRIVATE 0x0040 /* Private to thread/process. */
    #define REGION_REMOVED 0x0080 /* Already deleted. */
    #define REGION_SIZEDEF 0x0100 /* Use default region size if exists. */
    u_int32_t flags;
    };

而DB_LOCKREGION{} 数据结构是所有进程共用的数据结构，描述了封锁管理器的共享内存区域。注意到它的第一个字段就是RLAYOUT{}数据结构，因为它就是一个共享内存区域：

    /*
    * The lock region consists of:
    *    The DB_LOCKREGION structure (sizeof(DB_LOCKREGION)).
    *    The conflict matrix of nmodes * nmodes bytes (nmodes * nmodes).
    *    The hash table for object lookup (hashsize * sizeof(DB_OBJ *)).
    *    The locks themselves (maxlocks * sizeof(struct __db_lock).
    *    The objects being locked (maxlocks * sizeof(DB_OBJ)).
    *    String space to represent the DBTs that are the objects being locked.
    */
    struct __db_lockregion {
        RLAYOUT        hdr;        /* Shared region header. */
        u_int32_t    magic;        /* lock magic number */
        u_int32_t    version;    /* version number */
        u_int32_t    id;        /* unique id generator */
        u_int32_t    need_dd;    /* flag for deadlock detector */
        u_int32_t    detect;        /* run dd on every conflict */
        SH_TAILQ_HEAD(lock_header) free_locks;    /* free lock header */
        SH_TAILQ_HEAD(obj_header) free_objs;    /* free obj header */
        u_int32_t    maxlocks;    /* maximum number of locks in table */
        u_int32_t    table_size;    /* size of hash table */
        u_int32_t    nmodes;        /* number of lock modes */
        u_int32_t    numobjs;    /* number of objects */
        u_int32_t    nlockers;    /* number of lockers */
        size_t        increment;    /* how much to grow region */
        size_t        hash_off;    /* offset of hash table */
        size_t        mem_off;    /* offset of memory region */
        size_t        mem_bytes;    /* number of bytes in memory region */
        u_int32_t    nconflicts;    /* number of lock conflicts */
        u_int32_t    nrequests;    /* number of lock gets */
        u_int32_t    nreleases;    /* number of lock puts */
        u_int32_t    ndeadlocks;    /* number of deadlocks */
    };

另外注意到__db_locktab{}结构中有个DB_HASHTAB *hashtab;指针字段，指向DB_HASHTAB{}结构。这个数据结构定义在db_shash.h文件中：

    /* Hash Headers */
    typedef    SH_TAILQ_HEAD(hash_head) DB_HASHTAB;

实际上真正的散列表位于锁共享区域中，也就是我们上面结束的__db_lockregion{}数据结构中。该结构中有个字段：

    size_t  hash_off;    /* offset of hash table */

就是指向散列表在它空间中的偏移量，使用偏移量是因为这是一个共享区域，不同的进程会将其映射到不同的地址空间，因此不能使用绝对地址，这也是为什么__db_locktab{}结构要添加DB_HASHTAB *hashtab;字段的原因了。这样就不用每次都根据整个共享区域在本地址空间的映射地址+散列表在共享区域的偏移量，得出共享散列表在本地址空间的地址了。因为这些计算只需要在初始化时计算并保存在hashtab字段就可以了。


#### 2. 初始化过程

共享区域的最大特点就是共享，这意味着它在物理内存中只有一份映像，每个使用的进程都只是将其映射到自己的地址空间而已。操作的还是被映射的共享内存区域。有个重要的问题随之产生：就是一个进程怎么知道它是需要创建共享区域，还是加入该共享区域呢？因为共享区域总是要有人来创建的，这个创建者一般就是第一个要求使用共享区域的进程。所以一个进程必须能够区分一个共享区域是否已经存在，如果存在那么只要加入即可（即将该共享区域映射到自己的地址空间中），否则，先创建之，再映射之。要能够知道共享区域的存在与否，那么这个区域必须是有名字的，并且可根据该名字进行寻址的，就像文件系统一样。而事实上为了保证共享区域的名字的唯一性，BDB就使用了文件系统的路径来命名一个共享区域，每个共享区域有个对应的backing file。对于上面介绍的每进程私有的数据结构则每个进程都要自己创建。

在BDB的注释中有这么一段话：

    /*
     * There are three cases we support:
     *    1. Named anonymous memory (shmget(2)).
     *    2. Unnamed anonymous memory (mmap(2): MAP_ANON/MAP_ANONYMOUS).
     *    3. Memory backed by a regular file (mmap(2)).
     *
     * We instantiate a backing file in all cases, which contains at least
     * the RLAYOUT structure, and in case #3, contains the actual region.
     * This is necessary for a couple of reasons:
     *
     * First, the mpool region uses temporary files to name regions, and
     * since you may have multiple regions in the same directory, we need
     * a filesystem name to ensure that they don't collide.
     *
     * Second, applications are allowed to forcibly remove regions, even
     * if they don't know anything about them other than the name.  If a
     * region is backed by anonymous memory, there has to be some way for
     * the application to find out that information, and, in some cases,
     * determine ID information for the anonymous memory.

我们以共享锁区域为例，进行分析：

    int lock_open(path, flags, mode, dbenv, ltp)
        const char *path;
        u_int32_t flags;
        int mode;
        DB_ENV *dbenv;
        DB_LOCKTAB **ltp;
    {
        DB_LOCKTAB *lt;
        u_int32_t lock_modes, maxlocks, regflags;
        int ret;

        /* Validate arguments. */
        ....

        /* Create the lock table structure. */
        if ((ret = __os_calloc(1, sizeof(DB_LOCKTAB), &lt)) != 0)
            return (ret);
        lt->dbenv = dbenv;

        /* Grab the values that we need to compute the region size. */
        ....

        /* Join/create the lock region. */
        lt->reginfo.dbenv = dbenv;
        lt->reginfo.appname = DB_APP_NONE;
        if (path == NULL)
            lt->reginfo.path = NULL;
        else
            if ((ret = __os_strdup(path, &lt->reginfo.path)) != 0)
                goto err;
        lt->reginfo.file = DB_DEFAULT_LOCK_FILE;
        lt->reginfo.mode = mode;
        lt->reginfo.size =
            LOCK_REGION_SIZE(lock_modes, maxlocks, __db_tablesize(maxlocks));
        lt->reginfo.dbflags = flags;
        lt->reginfo.addr = NULL;
        lt->reginfo.fd = -1;
        lt->reginfo.flags = regflags;

        // 创建或者加入一个共享区域
        if ((ret = __db_rattach(&lt->reginfo)) != 0)
            goto err;

        /* Now set up the pointer to the region. */
        lt->region = lt->reginfo.addr;

        /* Initialize the region if we created it. */
        if (F_ISSET(&lt->reginfo, REGION_CREATED)) {
            lt->region->maxlocks = maxlocks;
            lt->region->nmodes = lock_modes;
            if ((ret = __lock_tabinit(dbenv, lt->region)) != 0)
                goto err;
        } else {
            /* Check for an unexpected region. */
            if (lt->region->magic != DB_LOCKMAGIC) {
                __db_err(dbenv,
                    "lock_open: %s: bad magic number", path);
                ret = EINVAL;
                goto err;
            }
        }

        /* Check for automatic deadlock detection. */
        ....

        /* Set up remaining pointers into region. */
        lt->conflicts = (u_int8_t *)lt->region + sizeof(DB_LOCKREGION);
        lt->hashtab =
            (DB_HASHTAB *)((u_int8_t *)lt->region + lt->region->hash_off);
        lt->mem = (void *)((u_int8_t *)lt->region + lt->region->mem_off);

        UNLOCK_LOCKREGION(lt);
        *ltp = lt;
        return (0);

    err:    if (lt->reginfo.addr != NULL) {
            UNLOCK_LOCKREGION(lt);
            (void)__db_rdetach(&lt->reginfo);
            if (F_ISSET(&lt->reginfo, REGION_CREATED))
                (void)lock_unlink(path, 1, dbenv);
        }

        if (lt->reginfo.path != NULL)
            __os_freestr(lt->reginfo.path);
        __os_free(lt, sizeof(*lt));
        return (ret);
    }

我们看到在lock_open()函数中调用了__db_rattach()函数创建或加入一个共享区域。REGINFO既是输入参数也是返回值。

    /*
    * __db_rattach --
    *    Optionally create and attach to a shared memory region.
    *
    * PUBLIC: int __db_rattach __P((REGINFO *));
    */
    int
    __db_rattach(infop)
        REGINFO *infop;
    {
        RLAYOUT *rlp, rl;
        size_t grow_region, size;
        ssize_t nr, nw;
        u_int32_t flags, mbytes, bytes;
        u_int8_t *p;
        int malloc_possible, ret, retry_cnt;

        /* Round off the requested size to the next page boundary. */
        ....

        /* Intialize the return information in the REGINFO structure. */
    loop:    infop->addr = NULL;
        infop->fd = -1;
        infop->segid = INVALID_SEGID;
        if (infop->name != NULL) {
            __os_freestr(infop->name);
            infop->name = NULL;
        }
        F_CLR(infop, REGION_CANGROW | REGION_CREATED);

    #ifndef HAVE_SPINLOCKS
        /*
        * XXX
        * Lacking spinlocks, we must have a file descriptor for fcntl(2)
        * locking, which implies using mmap(2) to map in a regular file.
        * (Theoretically, we could probably get a file descriptor to lock
        * other types of shared regions, but I don't see any reason to
        * bother.)
        *
        * Since we may be using shared memory regions, e.g., shmget(2),
        * and not mmap of regular files, the backing file may be only a
        * few tens of bytes in length.  So, this depends on the ability
        * to fcntl lock file offsets much larger than the physical file.
        */
        malloc_possible = 0;
    #endif

        /* 如果是私有内存区域，直接调用malloc就行了。
        * If a region is truly private, malloc the memory.  That's faster
        * than either anonymous memory or a shared file.
        */
        if (malloc_possible && F_ISSET(infop, REGION_PRIVATE)) {
            if ((ret = __os_malloc(infop->size, NULL, &infop->addr)) != 0)
                return (ret);
    
            /*
            * It's sometimes significantly faster to page-fault in all of
            * the region's pages before we run the application, as we see
            * nasty side-effects when we page-fault while holding various
            * locks, i.e., the lock takes a long time to acquire because
            * of the underlying page fault, and the other threads convoy
            * behind the lock holder.
            */
            if (DB_GLOBAL(db_region_init))
                for (p = infop->addr;
                    p < (u_int8_t *)infop->addr + infop->size;
                    p += DB_VMPAGESIZE)
                    p[0] = '\0';

            F_SET(infop, REGION_CREATED | REGION_MALLOC);
            goto region_init;
        }

        /* 得到欲加入或创建的共享区域的名称，这就是我们前面说到的共享区域的命名与寻址。
        * Get the name of the region (creating the file if a temporary file
        * is being used).  The dbenv contains the current DB environment,
        * including naming information.  The path argument may be a file or
        * a directory.  If path is a directory, it must exist and file is the
        * file name to be created inside the directory.  If path is a file,
        * then file must be NULL.
        */
        if ((ret = __db_appname(infop->dbenv, infop->appname, infop->path,
            infop->file, infop->dbflags, &infop->fd, &infop->name)) != 0)
            return (ret);
        if (infop->fd != -1) ///加入了一个共享区域
            F_SET(infop, REGION_CREATED);
    
        /* 创建共享区域
        * Try to create the file, if we have authority.  We have to make sure
        * that multiple threads/processes attempting to simultaneously create
        * the region are properly ordered, so we open it using DB_CREATE and
        * DB_EXCL, so two attempts to create the region will return failure in
        * one.
        */
        if (infop->fd == -1 && infop->dbflags & DB_CREATE) {
            flags = infop->dbflags;
            LF_SET(DB_EXCL);
            if ((ret = __db_open(infop->name,
                flags, flags, infop->mode, &infop->fd)) == 0)
                F_SET(infop, REGION_CREATED);
            else
                if (ret != EEXIST)
                    goto errmsg;
        }

        /* If we couldn't create the file, try and open it. */
        if (infop->fd == -1) {
            flags = infop->dbflags;
            LF_CLR(DB_CREATE | DB_EXCL);
            if ((ret = __db_open(infop->name,
                flags, flags, infop->mode, &infop->fd)) != 0)
                goto errmsg;
        }

        /*
        * There are three cases we support:
        *    1. Named anonymous memory (shmget(2)).
        *    2. Unnamed anonymous memory (mmap(2): MAP_ANON/MAP_ANONYMOUS).
        *    3. Memory backed by a regular file (mmap(2)).
        *
        * We instantiate a backing file in all cases, which contains at least
        * the RLAYOUT structure, and in case #3, contains the actual region.
        * This is necessary for a couple of reasons:
        *
        * First, the mpool region uses temporary files to name regions, and
        * since you may have multiple regions in the same directory, we need
        * a filesystem name to ensure that they don't collide.
        *
        * Second, applications are allowed to forcibly remove regions, even
        * if they don't know anything about them other than the name.  If a
        * region is backed by anonymous memory, there has to be some way for
        * the application to find out that information, and, in some cases,
        * determine ID information for the anonymous memory.
        */
        if (F_ISSET(infop, REGION_CREATED)) {
            /*
            * If we're using anonymous memory to back this region, set
            * the flag.
            */
            if (DB_GLOBAL(db_region_anon))
                F_SET(infop, REGION_ANONYMOUS);

            /*
            * If we're using a regular file to back a region we created,
            * grow it to the specified size.
            */
            if (!DB_GLOBAL(db_region_anon) &&
                (ret = __db_growregion(infop, infop->size)) != 0)
                goto err;
        } else {
            /*
            * If we're joining a region, figure out what it looks like.
            *
            * XXX
            * We have to figure out if the file is a regular file backing
            * a region that we want to map into our address space, or a
            * file with the information we need to find a shared anonymous
            * region that we want to map into our address space.
            *
            * All this noise is because some systems don't have a coherent
            * VM and buffer cache, and worse, if you mix operations on the
            * VM and buffer cache, half the time you hang the system.
            *
            * There are two possibilities.  If the file is the size of an
            * RLAYOUT structure, then we know that the real region is in
            * shared memory, because otherwise it would be bigger.  (As
            * the RLAYOUT structure size is smaller than a disk sector,
            * the only way it can be this size is if deliberately written
            * that way.)  In which case, retrieve the information we need
            * from the RLAYOUT structure and use it to acquire the shared
            * memory.
            *
            * If the structure is larger than an RLAYOUT structure, then
            * the file is backing the shared memory region, and we use
            * the current size of the file without reading any information
            * from the file itself so that we don't confuse the VM.
            *
            * And yes, this makes me want to take somebody and kill them,
            * but I can't think of any other solution.
            */
            if ((ret = __os_ioinfo(infop->name,
                infop->fd, &mbytes, &bytes, NULL)) != 0)
                goto errmsg;
            size = mbytes * MEGABYTE + bytes;

            if (size <= sizeof(RLAYOUT)) {
                /*
                * If the size is too small, the read fails or the
                * valid flag is incorrect, assume it's because the
                * RLAYOUT information hasn't been written out yet,
                * and retry.
                */
                if (size < sizeof(RLAYOUT))
                    goto retry;
                if ((ret =
                    __os_read(infop->fd, &rl, sizeof(rl), &nr)) != 0)
                    goto retry;
                if (rl.valid != DB_REGIONMAGIC)
                    goto retry;

                /* Copy the size, memory id and characteristics. */
                size = rl.size;
                infop->segid = rl.segid;
                if (F_ISSET(&rl, REGION_ANONYMOUS))
                    F_SET(infop, REGION_ANONYMOUS);
            }

            /*
            * If the region is larger than we think, that's okay, use the
            * current size.  If it's smaller than we think, and we were
            * just using the default size, that's okay, use the current
            * size.  If it's smaller than we think and we really care,
            * save the size and we'll catch that further down -- we can't
            * correct it here because we have to have a lock to grow the
            * region.
            */
            if (infop->size > size && !F_ISSET(infop, REGION_SIZEDEF))
                grow_region = infop->size;
            infop->size = size;
        }

        /*
        * Map the region into our address space.  If we're creating it, the
        * underlying routines will make it the right size.
        *
        * There are at least two cases where we can "reasonably" fail when
        * we attempt to map in the region.  On Windows/95, closing the last
        * reference to a region causes it to be zeroed out.  On UNIX, when
        * using the shmget(2) interfaces, the region will no longer exist
        * if the system was rebooted.  In these cases, the underlying map call
        * returns EAGAIN, and we *remove* our file and try again.  There are
        * obvious races in doing this, but it should eventually settle down
        * to a winner and then things should proceed normally.
        */
        if ((ret = __db_mapregion(infop->name, infop)) != 0)
            if (ret == EAGAIN) {
                /*
                * Pretend we created the region even if we didn't so
                * that our error processing unlinks it.
                */
                F_SET(infop, REGION_CREATED);
                ret = 0;
                goto retry;
            } else
                goto err;

    region_init:
        /*
        * Initialize the common region information.
        *
        * !!!
        * We have to order the region creates so that two processes don't try
        * to simultaneously create the region.  This is handled by using the
        * DB_CREATE and DB_EXCL flags when we create the "backing" region file.
        *
        * We also have to order region joins so that processes joining regions
        * never see inconsistent data.  We'd like to play permissions games
        * with the backing file, but we can't because WNT filesystems won't
        * open a file mode 0.
        */
        rlp = (RLAYOUT *)infop->addr;
        if (F_ISSET(infop, REGION_CREATED)) {
            /*
            * The process creating the region acquires a lock before it
            * sets the valid flag.  Any processes joining the region will
            * check the valid flag before acquiring the lock.
            *
            * Check the return of __db_mutex_init() and __db_mutex_lock(),
            * even though we don't usually check elsewhere.  This is the
            * first lock we initialize and acquire, and we have to know if
            * it fails.  (It CAN fail, e.g., SunOS, when using fcntl(2)
            * for locking, with an in-memory filesystem specified as the
            * database home.)
            */
            if ((ret = __db_mutex_init(&rlp->lock,
                MUTEX_LOCK_OFFSET(rlp, &rlp->lock))) != 0 ||
                (ret = __db_mutex_lock(&rlp->lock, infop->fd)) != 0)
                goto err;

            /* Initialize the remaining region information. */
            rlp->refcnt = 1;
            rlp->size = infop->size;
            db_version(&rlp->majver, &rlp->minver, &rlp->patch);
            rlp->panic = 0;
            rlp->segid = infop->segid;
            rlp->flags = 0;
            if (F_ISSET(infop, REGION_ANONYMOUS))
                F_SET(rlp, REGION_ANONYMOUS);

            /*
            * Fill in the valid field last -- use a magic number, memory
            * may not be zero-filled, and we want to minimize the chance
            * for collision.
            */
            rlp->valid = DB_REGIONMAGIC;

            /*
            * If the region is anonymous, write the RLAYOUT information
            * into the backing file so that future region join and unlink
            * calls can find it.
            *
            * XXX
            * We MUST do the seek before we do the write.  On Win95, while
            * closing the last reference to an anonymous shared region
            * doesn't discard the region, it does zero it out.  So, the
            * REGION_CREATED may be set, but the file may have already
            * been written and the file descriptor may be at the end of
            * the file.
            */
            if (F_ISSET(infop, REGION_ANONYMOUS)) {
                if ((ret = __os_seek(infop->fd, 0, 0, 0, 0, 0)) != 0)
                    goto err;
                if ((ret =
                    __os_write(infop->fd, rlp, sizeof(*rlp), &nw)) != 0)
                    goto err;
            }
        } else {
            /* Check to see if the region has had catastrophic failure. */
            if (rlp->panic) {
                ret = DB_RUNRECOVERY;
                goto err;
            }

            /*
            * Check the valid flag to ensure the region is initialized.
            * If the valid flag has not been set, the mutex may not have
            * been initialized, and an attempt to get it could lead to
            * random behavior.
            */
            if (rlp->valid != DB_REGIONMAGIC)
                goto retry;

            /* Get the region lock. */
            (void)__db_mutex_lock(&rlp->lock, infop->fd);

            /*
            * We now own the region.  There are a couple of things that
            * may have gone wrong, however.
            *
            * Problem #1: while we were waiting for the lock, the region
            * was deleted.  Detected by re-checking the valid flag, since
            * it's cleared by the delete region routines.
            */
            if (rlp->valid != DB_REGIONMAGIC) {
                (void)__db_mutex_unlock(&rlp->lock, infop->fd);
                goto retry;
            }

            /*
            * Problem #3: when we checked the size of the file, it was
            * still growing as part of creation.  Detected by the fact
            * that infop->size isn't the same size as the region.
            */
            if (infop->size != rlp->size) {
                (void)__db_mutex_unlock(&rlp->lock, infop->fd);
                goto retry;
            }

            /* Increment the reference count. */
            ++rlp->refcnt;
        }

        /* Return the region in a locked condition. */

        if (0) {
    errmsg:        __db_err(infop->dbenv, "%s: %s", infop->name, strerror(ret));

    err:
    retry:        /* Discard the region. */
            if (infop->addr != NULL) {
                (void)__db_unmapregion(infop);
                infop->addr = NULL;
            }

            /* Discard the backing file. */
            if (infop->fd != -1) {
                (void)__os_close(infop->fd);
                infop->fd = -1;

                if (F_ISSET(infop, REGION_CREATED))
                    (void)__os_unlink(infop->name);
            }

            /* Discard the name. */
            if (infop->name != NULL) {
                __os_freestr(infop->name);
                infop->name = NULL;
            }

            /*
            * If we had a temporary error, wait a few seconds and
            * try again.
            */
            if (ret == 0) {
                if (++retry_cnt <= 3) {
                    __os_sleep(retry_cnt * 2, 0);
                    goto loop;
                }
                ret = EAGAIN;
            }
        }

        /* XXX
        * There's a problem in cleaning this up on application exit, or on
        * application failure.  If an application opens a database without
        * an environment, we create a temporary backing mpool region for it.
        * That region is marked REGION_PRIVATE, but as HP-UX won't permit
        * mutexes to live in anything but shared memory, we instantiate a
        * real file plus a memory region of some form.  If the application
        * crashes, the necessary information to delete the backing file and
        * any system region (e.g., the shmget(2) segment ID) is no longer
        * available.  We can't completely fix the problem, but we try.
        *
        * The underlying UNIX __db_mapregion() code preferentially uses the
        * mmap(2) interface with the MAP_ANON/MAP_ANONYMOUS flags for regions
        * that are marked REGION_PRIVATE.  This means that we normally aren't
        * holding any system resources when we get here, in which case we can
        * delete the backing file.  This results in a short race, from the
        * __db_open() call above to here.
        *
        * If, for some reason, we are holding system resources when we get
        * here, we don't have any choice -- we can't delete the backing file
        * because we may need it to detach from the resources.  Set the
        * REGION_LASTDETACH flag, so that we do all necessary cleanup when
        * the application closes the region.
        */
        if (F_ISSET(infop, REGION_PRIVATE) && !F_ISSET(infop, REGION_MALLOC))
            if (F_ISSET(infop, REGION_HOLDINGSYS))
                F_SET(infop, REGION_LASTDETACH);
            else {
                F_SET(infop, REGION_REMOVED);
                F_CLR(infop, REGION_CANGROW);

                (void)__os_close(infop->fd);
                (void)__os_unlink(infop->name);
            }

        return (ret);
    }

__db_rattach()函数调用了__db_appname()函数来命名和定位一个共享区域：

    if ((ret = __db_appname(infop->dbenv, infop->appname, infop->path,
            infop->file, infop->dbflags, &infop->fd, &infop->name)) != 0)
            return (ret);

其中appname是一个枚举类型，表示应用程序的类型：

    /* Type passed to __db_appname(). */
    typedef enum {
        DB_APP_NONE=0,            /* No type (region). */
        DB_APP_DATA,            /* Data file. */
        DB_APP_LOG,            /* Log file. */
        DB_APP_TMP            /* Temporary file. */
    } APPNAME;

我们在__db_appname()中会看到它的作用。

对于打开或创建Lock共享区域，上述调用参数为：

    infop->appname=DB_APP_NONE（"/home/database"）, infop->path = NULL, infop->file=DB_DEFAULT_LOCK_FILE("__db_lock.share")。

这样调用__db_appname()将返回区域名字为/DB_HOME/file，也就是"/home/database/__db_lock.share"。然后回到db_rattach()函数，db_rattach()会使用DB_CREATE 和 DB_EXCL尝试打开这个文件，如果文件已经存在那么表示区域已经存在，不需要创建，直接加入即可。否则，创建该文件，再加入。

    /*
    * __db_appname --
    *    Given an optional DB environment, directory and file name and type
    *    of call, build a path based on the db_appinit(3) rules, and return
    *    it in allocated space.
    *
    * PUBLIC: int __db_appname __P((DB_ENV *,
    * PUBLIC:    APPNAME, const char *, const char *, u_int32_t, int *, char **));
    */
    int
    __db_appname(dbenv, appname, dir, file, tmp_oflags, fdp, namep)
        DB_ENV *dbenv;
        APPNAME appname;
        const char *dir, *file;
        u_int32_t tmp_oflags;
        int *fdp;
        char **namep;
    {
        DB_ENV etmp;
        size_t len;
        int data_entry, ret, slash, tmp_create, tmp_free;
        const char *a, *b, *c;
        char *p, *start;

        a = b = c = NULL;
        data_entry = -1;
        tmp_create = tmp_free = 0;

        /*
        * We don't return a name when creating temporary files, just an fd.
        * Default to error now.
        */
        if (fdp != NULL)
            *fdp = -1;
        if (namep != NULL)
            *namep = NULL;

        /*
        * Absolute path names are never modified.  If the file is an absolute
        * path, we're done.  If the directory is, simply append the file and
        * return.
        */
        if (file != NULL && __os_abspath(file))
            return (__os_strdup(file, namep));
        if (dir != NULL && __os_abspath(dir)) {
            a = dir;
            goto done;
        }

        /*
        * DB_ENV  DIR       APPNAME       RESULT
        * -------------------------------------------
        * null       null       none           <tmp>/file
        * null       set       none           DIR/file
        * set       null       none           DB_HOME/file
        * set       set       none           DB_HOME/DIR/file
        *
        * DB_ENV  FILE       APPNAME       RESULT
        * -------------------------------------------
        * null       null       DB_APP_DATA       <tmp>/<create>
        * null       set       DB_APP_DATA       ./file
        * set       null       DB_APP_DATA       <tmp>/<create>
        * set       set       DB_APP_DATA       DB_HOME/DB_DATA_DIR/file
        *
        * DB_ENV  DIR       APPNAME       RESULT
        * -------------------------------------------
        * null       null       DB_APP_LOG       <tmp>/file
        * null       set       DB_APP_LOG       DIR/file
        * set       null       DB_APP_LOG       DB_HOME/DB_LOG_DIR/file
        * set       set       DB_APP_LOG       DB_HOME/DB_LOG_DIR/DIR/file
        *
        * DB_ENV       APPNAME       RESULT
        * -------------------------------------------
        * null           DB_APP_TMP*       <tmp>/<create>
        * set           DB_APP_TMP*       DB_HOME/DB_TMP_DIR/<create>
        */
    retry:    switch (appname) {
        case DB_APP_NONE:
            if (dbenv == NULL || !F_ISSET(dbenv, DB_ENV_APPINIT)) {
                if (dir == NULL)
                    goto tmp;
                a = dir;
            } else {
                a = dbenv->db_home;
                b = dir;
            }
            break;
        case DB_APP_DATA:
            if (dir != NULL) {
                __db_err(dbenv,
                    "DB_APP_DATA: illegal directory specification");
                return (EINVAL);
            }

            if (file == NULL) {
                tmp_create = 1;
                goto tmp;
            }
            if (dbenv == NULL || !F_ISSET(dbenv, DB_ENV_APPINIT))
                a = PATH_DOT;
            else {
                a = dbenv->db_home;
                if (dbenv->db_data_dir != NULL &&
                    (b = dbenv->db_data_dir[++data_entry]) == NULL) {
                    data_entry = -1;
                    b = dbenv->db_data_dir[0];
                }
            }
            break;
        case DB_APP_LOG:
            if (dbenv == NULL || !F_ISSET(dbenv, DB_ENV_APPINIT)) {
                if (dir == NULL)
                    goto tmp;
                a = dir;
            } else {
                a = dbenv->db_home;
                b = dbenv->db_log_dir;
                c = dir;
            }
            break;
        case DB_APP_TMP:
            if (dir != NULL || file != NULL) {
                __db_err(dbenv,
                "DB_APP_TMP: illegal directory or file specification");
                return (EINVAL);
            }

            tmp_create = 1;
            if (dbenv == NULL || !F_ISSET(dbenv, DB_ENV_APPINIT))
                goto tmp;
            else {
                a = dbenv->db_home;
                b = dbenv->db_tmp_dir;
            }
            break;
        }

        /* Reference a file from the appropriate temporary directory. */
        if (0) {
    tmp:        if (dbenv == NULL || !F_ISSET(dbenv, DB_ENV_APPINIT)) {
                memset(&etmp, 0, sizeof(etmp));
                if ((ret = __os_tmpdir(&etmp, DB_USE_ENVIRON)) != 0)
                    return (ret);
                tmp_free = 1;
                a = etmp.db_tmp_dir;
            } else
                a = dbenv->db_tmp_dir;
        }

    done:    len =
            (a == NULL ? 0 : strlen(a) + 1) +
            (b == NULL ? 0 : strlen(b) + 1) +
            (c == NULL ? 0 : strlen(c) + 1) +
            (file == NULL ? 0 : strlen(file) + 1);

        /*
        * Allocate space to hold the current path information, as well as any
        * temporary space that we're going to need to create a temporary file
        * name.
        */
    #define    DB_TRAIL    "XXXXXX"
        if ((ret =
            __os_malloc(len + sizeof(DB_TRAIL) + 10, NULL, &start)) != 0) {
            if (tmp_free)
                __os_freestr(etmp.db_tmp_dir);
            return (ret);
        }

        slash = 0;
        p = start;
        DB_ADDSTR(a);
        DB_ADDSTR(b);
        DB_ADDSTR(file);
        *p = '\0';

        /* Discard any space allocated to find the temp directory. */
        if (tmp_free) {
            __os_freestr(etmp.db_tmp_dir);
            tmp_free = 0;
        }

        /*
        * If we're opening a data file, see if it exists.  If it does,
        * return it, otherwise, try and find another one to open.
        */
        if (data_entry != -1 && __os_exists(start, NULL) != 0) {
            __os_freestr(start);
            a = b = c = NULL;
            goto retry;
        }

        /* Create the file if so requested. */
        if (tmp_create &&
            (ret = __db_tmp_open(dbenv, tmp_oflags, start, fdp)) != 0) {
            __os_freestr(start);
            return (ret);
        }

        if (namep == NULL)
            __os_freestr(start);
        else
            *namep = start;
        return (0);
    }


#### 3. 最后是函数调用过程


#### 4. 相关文档

##### Building transaction protected applications

Creating transaction protected applications using the Berkeley DB access methods is quite easy. In almost all cases, applications use db_appinit to perform initialization of all of the Berkeley DB subsystems. As transaction support requires all five Berkeley DB subsystems, the DB_INIT_MPOOL, DB_INIT_LOCK, DB_INIT_LOG and DB_INIT_TXN flags should be specified.

Once the application has called db_appinit, it should open the databases it will use. Once the databases are opened, the application can make changes to the databases, grouped inside of transaction calls. Each set of changes should be surrounded by the appropriate txn_begin, txn_commit and txn_abort calls.

The Berkeley DB access methods will make the appropriate calls into the lock, log and memory pool subsystems in order to guarantee transaction semantics. When the application is ready to exit, all outstanding transactions should have been committed or aborted.

Databases accessed by a transaction must not be closed during the transaction. Once all outstanding transactions are finished, all open Berkeley DB files should be closed. When the Berkeley DB database files have been closed, the environment should be closed by calling db_appexit.

也就是说如果要建立事务保护的应用程序，必须在打开数据库文件之前显示的调用db_appinit()初始化所有BDB子系统（包括MPOOL）。在退出的时候，必须在关闭数据库文件之后，显示调用db_appexit()函数清除BDB所有子系统（引用计数减1）。

####### Creating an Environment

The db_appinit function is the standard function for creating or joining a database environment. Every transaction-protected or multi-process application should call db_appinit before making any other calls to the Berkeley DB library.

Almost all applications either specify only the DB_INIT_MPOOL flag or they specify all four flags, DB_INIT_MPOOL, DB_INIT_LOCK, DB_INIT_LOG and DB_INIT_TXN. The former configuration is for applications that simply want to use the basic Access Method interfaces with a shared underlying buffer pool, but don't care about recoverability after failure. The latter is for applications that need recoverability. There are situations where other combinations of the initialization flags make sense, but they are quite rare.


一旦使用db_appinit()建立environment对象之后，就可以用这个env打开数据库了。
Opening databases within the environment

Once the environment has been created, the returned handle should be passed as an argument to the db_open call. This causes the database being opened to be opened within the environment. File naming and database operations will all be done as specified for the environment, e.g., if the DB_INIT_LOCK flag was specified when the environment was created or joined, database operations will automatically perform all necessary locking operations for the application.

###### Shared Memory Regions

Each of the Berkeley DB subsystems is described by one or more shared memory regions. These regions live in the environment home directory, and contain all of the shared information, including mutexes, that describes the Berkeley DB environment.

The Berkeley DB library uses the POSIX mmap (or other similar) interface to map the shared memory regions. Most remote file systems (e.g., the Network File System (NFS) and the Andrew File System (AFS)), do not support mapping files into process memory. For this reason, we strongly recommend that the database home directory reside in a local filesystem.

For remote file systems that do allow system files to be mapped into process memory, it is important to note that home directories accessed via remote file systems cannot be used simultaneously from multiple clients. None of the commercial remote file systems available today implement a coherent, distributed shared memory paradigm for remote-mounted files. As a result, different machines will see different versions of these shared regions and the system behavior is undefined.

Databases, log files and temporary files may be placed on remote filesystems, although the application may incur a performance penalty for doing so. 

