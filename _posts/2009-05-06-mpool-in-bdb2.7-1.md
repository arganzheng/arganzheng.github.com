---
layout: post
title: MPOOL共享内存
---

2009-5-6 星期三 晴朗

在BDB中所有的共享内存都定义为XXX_REGION{}结构。 比如MPOOL缓冲区，每一个进程有自己的缓存数据结构DB_MPOOL{}：

    /*
    * DB_MPOOL --
    *    Per-process memory pool structure.
    */
    struct __db_mpool {
    /* These fields need to be protected for multi-threaded support. */
        db_mutex_t    *mutexp;    /* Structure lock. */

                        /* List of pgin/pgout routines. */
        LIST_HEAD(__db_mpregh, __db_mpreg) dbregq;

                        /* List of DB_MPOOLFILE's. */
        TAILQ_HEAD(__db_mpoolfileh, __db_mpoolfile) dbmfq;

    /* These fields are not protected. */
        DB_ENV     *dbenv;        /* Reference to error information. */
        REGINFO        reginfo;        /* Region information. */

        MPOOL       *mp;            /* Address of the shared MPOOL. */

        void       *addr;        /* Address of shalloc() region. */

        DB_HASHTAB *htab;        /* Hash table of bucket headers. */

    #define    MP_LOCKHANDLE    0x01        /* Threaded, lock handles and region. */
    #define    MP_LOCKREGION    0x02        /* Concurrent access, lock region. */
        u_int32_t  flags;
    };

其中
reginfo字段指向共享内存区域描述信息结构REGINFO{}。 
mp字段指向分配在共享缓冲区的，用于描述该缓冲区的MPOOL结构的起始地址。该结构位于共享区域的开始。
addr字段指向共享内存的在其地址空间的映射的起始地址，位于:共享区域地址mp+sizeof(mp)，具体参见

    int __memp_ropen(dbmp, path, cachesize, mode, is_private, flags);函数中：
    /*  * The MPOOL structure is first in the region, the rest of the region
        * is free space.  */
        dbmp->mp = dbmp->reginfo.addr;
        dbmp->addr = (u_int8_t *)dbmp->mp + sizeof(MPOOL);
---

    /*
    * MPOOL --
    *    Shared memory pool region.  One of these is allocated in shared
    *    memory, and describes the pool.
    */
    struct __mpool {
        RLAYOUT        rlayout;        /* General region information. */

        SH_TAILQ_HEAD(__bhq) bhq;    /* LRU list of buckets. */
        SH_TAILQ_HEAD(__bhfq) bhfq;    /* Free buckets. */
        SH_TAILQ_HEAD(__mpfq) mpfq;    /* List of MPOOLFILEs. */

        /*
        * We make the assumption that the early pages of the file are far
        * more likely to be retrieved than the later pages, which means
        * that the top bits are more interesting for hashing since they're
        * less likely to collide.  On the other hand, since 512 4K pages
        * represents a 2MB file, only the bottom 9 bits of the page number
        * are likely to be set.  We XOR in the offset in the MPOOL of the
        * MPOOLFILE that backs this particular page, since that should also
        * be unique for the page.
        */
    #define    BUCKET(mp, mf_offset, pgno)                    \
        (((pgno) ^ ((mf_offset) << 9)) % (mp)->htab_buckets)

        size_t        htab;        /* Hash table offset. */
        size_t        htab_buckets;    /* Number of hash table entries. */

        DB_LSN        lsn;        /* Maximum checkpoint LSN. */
        u_int32_t   lsn_cnt;        /* Checkpoint buffers left to write. */

        DB_MPOOL_STAT stat;        /* Global mpool statistics. */

    #define    MP_LSN_RETRY    0x01        /* Retry all BH_WRITE buffers. */
        u_int32_t  flags;
    };

其中rlayout字段是每个REGION的头部数据结构，是各个REGION的通用信息。定义在db_int.h文件中。

剩下的信息都是应用相关的，在MPOOL中就是描述用于内存缓冲区的共享区域信息。
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

可以看到它包含描述一个共享区域的所有必要通用信息，包括：
对描述信息的并发访问控制信号量：lock。
共享区域的引用计数：refcnt。
贡献区域的大小：size。
共享区域的id，如果是使用shmget(2)的话：segid。
还有一些标志符：flags。

数据结构大概描述完成，下面来看看是如何分配和处理这些数据结构的。

在db_appinit（）函数中开始初始化所有共享内存区域，我们以缓冲区为例。它将调用memp_open（）函数（定义在mp_open.c文件中）：

    /*
    * memp_open --
    *    Initialize and/or join a memory pool.
    */
    int
    memp_open(path, flags, mode, dbenv, retp)
        const char *path;
        u_int32_t flags;
        int mode;
        DB_ENV *dbenv;
        DB_MPOOL **retp;
    {
        DB_MPOOL *dbmp;
        size_t cachesize;
        int is_private, ret;

        /* Validate arguments. */
        ......
        /* Extract fields from DB_ENV structure. */
        cachesize = dbenv == NULL ? 0 : dbenv->mp_size;
    
    /* 对于每个进程私有的数据结构，一律用malloc之类的库函数/系统调用 */
        /* Create and initialize the DB_MPOOL structure. */
        if ((ret = __os_calloc(1, sizeof(DB_MPOOL), &dbmp)) != 0)
            return (ret);
        LIST_INIT(&dbmp->dbregq);
        TAILQ_INIT(&dbmp->dbmfq);

        dbmp->dbenv = dbenv;

        /* Decide if it's possible for anyone else to access the pool. */
        is_private =
            (dbenv == NULL && path == NULL) || LF_ISSET(DB_MPOOL_PRIVATE);

        /*
        * Map in the region.  We do locking regardless, as portions of it are
        * implemented in common code (if we put the region in a file, that is).
        */
        F_SET(dbmp, MP_LOCKREGION);
        if ((ret = __memp_ropen(dbmp,
            path, cachesize, mode, is_private, LF_ISSET(DB_CREATE))) != 0)
            goto err;
        F_CLR(dbmp, MP_LOCKREGION);

        /*
        * If there's concurrent access, then we have to lock the region.
        * If it's threaded, then we have to lock both the handles and the
        * region, and we need to allocate a mutex for that purpose.
        */
        if (!is_private)
            F_SET(dbmp, MP_LOCKREGION);
        if (LF_ISSET(DB_THREAD)) {
            F_SET(dbmp, MP_LOCKHANDLE | MP_LOCKREGION);
            LOCKREGION(dbmp);
            ret = __memp_alloc(dbmp,
                sizeof(db_mutex_t), NULL, &dbmp->mutexp);
            UNLOCKREGION(dbmp);
            if (ret != 0) {
                (void)memp_close(dbmp);
                goto err;
            }
            LOCKINIT(dbmp, dbmp->mutexp);
        }

        *retp = dbmp;
        return (0);

    err:    if (dbmp != NULL)
            __os_free(dbmp, sizeof(DB_MPOOL));
        return (ret);
    }

该函数在 调用__os_calloc()创建了per-process数据结构DB_MPOOL{}之后，调用 __memp_ropen()创建或打开缓冲区共享内存区域（定义在mp_region.c文件中）。

该函数首先利用传递的信息构造REGINFO{}结构，然后掉哟给你__db_rattach(...)函数创建或打开共享内存区域：

    if ((ret = __db_rattach(&dbmp->reginfo)) != 0)

如果这是一个新创建的共享区域 （ if (F_ISSET(&dbmp->reginfo, REGION_CREATED)) ），那么它还负责对MPOOL共享内存区域进行初始化操作（通用的共享区域头部信息RLAYOUT{}的初始化由__db_rattach()在分配后完成，不是它的职责）。

    /*
        * The MPOOL structure is first in the region, the rest of the region
        * is free space.
        */
        dbmp->mp = dbmp->reginfo.addr;
        dbmp->addr = (u_int8_t *)dbmp->mp + sizeof(MPOOL);

        /* Initialize a created region. */
        if (F_ISSET(&dbmp->reginfo, REGION_CREATED)) {
            mp = dbmp->mp;
            SH_TAILQ_INIT(&mp->bhq);
            SH_TAILQ_INIT(&mp->bhfq);
            SH_TAILQ_INIT(&mp->mpfq);

            __db_shalloc_init(dbmp->addr, rlen - sizeof(MPOOL));

            mp->htab_buckets =
                __db_tablesize((cachesize / (1 * 1024)) / 10);

            /* Allocate hash table space and initialize it. */
            if ((ret = __db_shalloc(dbmp->addr,
                mp->htab_buckets * sizeof(DB_HASHTAB),
                0, &dbmp->htab)) != 0)
                goto err;
            __db_hashinit(dbmp->htab, mp->htab_buckets);
            mp->htab = R_OFFSET(dbmp, dbmp->htab);

            ZERO_LSN(mp->lsn);
            mp->lsn_cnt = 0;

            memset(&mp->stat, 0, sizeof(mp->stat));
            mp->stat.st_cachesize = cachesize;

            mp->flags = 0;
        }

        /* Get the local hash table address. */
        dbmp->htab = R_ADDR(dbmp, dbmp->mp->htab);

        UNLOCKREGION(dbmp);
        return (0);

需要注意的是：BDB对共享区域进行了内存管理，这是因为缓冲区经常要对空间要求与释放。其实现类似于libc中的malloc实现。

具体接口和定义在db_salloc.c文件中，我们只列举了上面用到的两个函数db_shalloc_init和initdb_shalloc：

    /*
    * Implement shared memory region allocation, using simple first-fit algorithm.
    * The model is that we take a "chunk" of shared memory store and begin carving
    * it up into areas, similarly to how malloc works.  We do coalescing on free.
    *
    * The "len" field in the __data struct contains the length of the free region
    * (less the size_t bytes that holds the length).  We use the address provided
    * by the caller to find this length, which allows us to free a chunk without
    * requiring that the caller pass in the length of the chunk they're freeing.
    */
    SH_LIST_HEAD(__head);
    struct __data {
        size_t len;
        SH_LIST_ENTRY links;
    };

    /*
    * __db_shalloc_init --
    *    Initialize the area as one large chunk.
    *
    * PUBLIC: void __db_shalloc_init __P((void *, size_t));
    */
    void
    __db_shalloc_init(area, size)
        void *area;
        size_t size;
    {
        struct __data *elp;
        struct __head *hp;

        hp = area;
        SH_LIST_INIT(hp);

        elp = (struct __data *)(hp + 1);
        elp->len = size - sizeof(struct __head) - sizeof(elp->len);
        SH_LIST_INSERT_HEAD(hp, elp, links, __data);
    }

    /*
    * __db_shalloc --
    *    Allocate some space from the shared region.
    *
    * PUBLIC: int __db_shalloc __P((void *, size_t, size_t, void *));
    */
    int
    __db_shalloc(p, len, align, retp)
        void *p, *retp;
        size_t len, align;
    {
        struct __data *elp;
        size_t *sp;
        void *rp;

        /*
        * We never allocate less than the size of a struct __data, align
        * to less than a size_t boundary, or align to something that's not
        * a multiple of a size_t.
        */
        if (len < sizeof(struct __data))
            len = sizeof(struct __data);
        align = align <= sizeof(size_t) ?
            sizeof(size_t) : ALIGN(align, sizeof(size_t));

        /* Walk the list, looking for a slot. */
        for (elp = SH_LIST_FIRST((struct __head *)p, __data);
            elp != NULL;
            elp = SH_LIST_NEXT(elp, links, __data)) {
            /*
            * Calculate the value of the returned pointer if we were to
            * use this chunk.
            *    + Find the end of the chunk.
            *    + Subtract the memory the user wants.
            *    + Find the closest previous correctly-aligned address.
            */
            rp = (u_int8_t *)elp + sizeof(size_t) + elp->len;
            rp = (u_int8_t *)rp - len;
            rp = (u_int8_t *)((ALIGNTYPE)rp & ~(align - 1));

            /*
            * Rp may now point before elp->links, in which case the chunk
            * was too small, and we have to try again.
            */
            if ((u_int8_t *)rp < (u_int8_t *)&elp->links)
                continue;

            *(void **)retp = rp;

    #define    SHALLOC_FRAGMENT    32
            /*
            * If there are at least SHALLOC_FRAGMENT additional bytes of
            * memory, divide the chunk into two chunks.
            */
            if ((u_int8_t *)rp >=
                (u_int8_t *)&elp->links + SHALLOC_FRAGMENT) {
                sp = rp;
                *--sp = elp->len -
                    ((u_int8_t *)rp - (u_int8_t *)&elp->links);
                elp->len -= *sp + sizeof(size_t);
                return (0);
            }

            /*
            * Otherwise, we return the entire chunk, wasting some amount
            * of space to keep the list compact.  However, because the
            * address we're returning to the user may not be the address
            * of the start of the region for alignment reasons, set the
            * size_t length fields back to the "real" length field to a
            * flag value, so that we can find the real length during free.
            */
    #define    ILLEGAL_SIZE    1
            SH_LIST_REMOVE(elp, links, __data);
            for (sp = rp; (u_int8_t *)--sp >= (u_int8_t *)&elp->links;)
                *sp = ILLEGAL_SIZE;
            return (0);
        }

        /* Nothing found large enough; need to grow the region. */
        return (ENOMEM);
    }

---

    /*
    * __memp_ropen --
    *    Attach to, and optionally create, the mpool region.
    *
    * PUBLIC: int __memp_ropen
    * PUBLIC:    __P((DB_MPOOL *, const char *, size_t, int, int, u_int32_t));
    */
    int
    __memp_ropen(dbmp, path, cachesize, mode, is_private, flags)
        DB_MPOOL *dbmp;
        const char *path;
        size_t cachesize;
        int mode, is_private;
        u_int32_t flags;
    {
        MPOOL *mp;
        size_t rlen;
        int defcache, ret;

        /*
        * Unlike other DB subsystems, mpool can't simply grow the region
        * because it returns pointers into the region to its clients.  To
        * "grow" the region, we'd have to allocate a new region and then
        * store a region number in the structures that reference regional
        * objects.  It's reasonable that we fail regardless, as clients
        * shouldn't have every page in the region pinned, so the only
        * "failure" mode should be a performance penalty because we don't
        * find a page in the cache that we'd like to have found.
        *
        * Up the user's cachesize by 25% to account for our overhead.
        */
        defcache = 0;
        if (cachesize < DB_CACHESIZE_MIN)
            if (cachesize == 0) {
                defcache = 1;
                cachesize = DB_CACHESIZE_DEF;
            } else
                cachesize = DB_CACHESIZE_MIN;
        rlen = cachesize + cachesize / 4;

        /*
        * Map in the region.
        *
        * If it's a private mpool, use malloc, it's a lot faster than
        * instantiating a region.
        */
        dbmp->reginfo.dbenv = dbmp->dbenv;
        dbmp->reginfo.appname = DB_APP_NONE;
        if (path == NULL)
            dbmp->reginfo.path = NULL;
        else
            if ((ret = __os_strdup(path, &dbmp->reginfo.path)) != 0)
                return (ret);
        dbmp->reginfo.file = DB_DEFAULT_MPOOL_FILE;
        dbmp->reginfo.mode = mode;
        dbmp->reginfo.size = rlen;
        dbmp->reginfo.dbflags = flags;
        dbmp->reginfo.flags = 0;
        if (defcache)
            F_SET(&dbmp->reginfo, REGION_SIZEDEF);

        /*
        * If we're creating a temporary region, don't use any standard
        * naming.
        */
        if (is_private) {
            dbmp->reginfo.appname = DB_APP_TMP;
            dbmp->reginfo.file = NULL;
            F_SET(&dbmp->reginfo, REGION_PRIVATE);
        }

        if ((ret = __db_rattach(&dbmp->reginfo)) != 0) {
            if (dbmp->reginfo.path != NULL)
                __os_freestr(dbmp->reginfo.path);
            return (ret);
        }

        /*
        * The MPOOL structure is first in the region, the rest of the region
        * is free space.
        */
        dbmp->mp = dbmp->reginfo.addr;
        dbmp->addr = (u_int8_t *)dbmp->mp + sizeof(MPOOL);

        /* Initialize a created region. */
        if (F_ISSET(&dbmp->reginfo, REGION_CREATED)) {
            mp = dbmp->mp;
            SH_TAILQ_INIT(&mp->bhq);
            SH_TAILQ_INIT(&mp->bhfq);
            SH_TAILQ_INIT(&mp->mpfq);

            __db_shalloc_init(dbmp->addr, rlen - sizeof(MPOOL));

            /*
            * Assume we want to keep the hash chains with under 10 pages
            * on each chain.  We don't know the pagesize in advance, and
            * it may differ for different files.  Use a pagesize of 1K for
            * the calculation -- we walk these chains a lot, they should
            * be short.
            */
            mp->htab_buckets =
                __db_tablesize((cachesize / (1 * 1024)) / 10);

            /* Allocate hash table space and initialize it. */
            if ((ret = __db_shalloc(dbmp->addr,
                mp->htab_buckets * sizeof(DB_HASHTAB),
                0, &dbmp->htab)) != 0)
                goto err;
            __db_hashinit(dbmp->htab, mp->htab_buckets);
            mp->htab = R_OFFSET(dbmp, dbmp->htab);

            ZERO_LSN(mp->lsn);
            mp->lsn_cnt = 0;

            memset(&mp->stat, 0, sizeof(mp->stat));
            mp->stat.st_cachesize = cachesize;

            mp->flags = 0;
        }

        /* Get the local hash table address. */
        dbmp->htab = R_ADDR(dbmp, dbmp->mp->htab);

        UNLOCKREGION(dbmp);
        return (0);

    err:    UNLOCKREGION(dbmp);
        (void)__db_rdetach(&dbmp->reginfo);
        if (F_ISSET(&dbmp->reginfo, REGION_CREATED))
            (void)memp_unlink(path, 1, dbmp->dbenv);

        if (dbmp->reginfo.path != NULL)
            __os_freestr(dbmp->reginfo.path);
        return (ret);
    }

这个函数用到了一个新的数据结构DB_REGINFO{}，这是一个在函数调用之间传递的用户描述一个共享区域的小数据结构（cookies）：

    /*
    * The interface to region attach is nasty, there is a lot of complex stuff
    * going on, which has to be retained between create/attach and detach.  The
    * REGINFO structure keeps track of it.
    */
    struct __db_reginfo;    typedef struct __db_reginfo REGINFO;
    struct __db_reginfo {
                        /* Arguments. */
        DB_ENV       *dbenv;        /* Region naming info. */
        APPNAME        appname;        /* Region naming info. */
        char       *path;        /* Region naming info. */
        const char *file;        /* Region naming info. */
        int        mode;        /* Region mode, if a file. */
        size_t        size;        /* Region size. */
        u_int32_t   dbflags;        /* Region file open flags, if a file. */

                        /* Results. */
        char       *name;        /* Region name. */
        void       *addr;        /* Region address. */
        int        fd;            /* Fcntl(2) locking file descriptor.
                        NB: this is only valid if a regular
                        file is backing the shared region,
                        and mmap(2) is being used to map it
                        into our address space. */
        int        segid;        /* shmget(2) ID, or Win16 segment ID. */
        void       *wnt_handle;        /* Win/NT HANDLE. */

                        /* Shared flags. */
    /*                0x0001    COMMON MASK with RLAYOUT structure. */
    #define    REGION_CANGROW        0x0002    /* Can grow. */
    #define    REGION_CREATED        0x0004    /* Created. */
    #define    REGION_HOLDINGSYS    0x0008    /* Holding system resources. */
    #define    REGION_LASTDETACH    0x0010    /* Delete on last detach. */
    #define    REGION_MALLOC        0x0020    /* Created in malloc'd memory. */
    #define    REGION_PRIVATE        0x0040    /* Private to thread/process. */
    #define    REGION_REMOVED        0x0080    /* Already deleted. */
    #define    REGION_SIZEDEF        0x0100    /* Use default region size if exists. */
        u_int32_t   flags;
    };

memp_ropen()函数调用所有特殊共享区域（如缓冲区，锁表，日志缓冲区）最终调用的函数int __db_rattach(REGINFO* infop);来创建或打开一个通用共享内存区域，

其中infop既是输入信息（要创建或打开的信息，共享区域的大小size，backup文件路径以构造名字或打开backup文件，还有各种标志flags）， 也是输出信息（创建或打开后的信息，主要是共享区域的起始地址addr，文件描述符fd，shmget的内部标识符segid，名字等）：

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

        grow_region = 0;
        malloc_possible = 1;
        ret = retry_cnt = 0;

        /* Round off the requested size to the next page boundary. */
        DB_ROUNDOFF(infop->size, DB_VMPAGESIZE);

        /* Some architectures have hard limits on the maximum region size. */
    #ifdef DB_REGIONSIZE_MAX
        if (infop->size > DB_REGIONSIZE_MAX) {
            __db_err(infop->dbenv, "__db_rattach: cache size too large");
            return (EINVAL);
        }
    #endif

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

    #ifdef __hppa
        /*
        * XXX
        * HP-UX won't permit mutexes to live in anything but shared memory.
        * Instantiate a shared region file on that architecture, regardless.
        */
        malloc_possible = 0;
    #endif
        /*
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

        /*
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
        if (infop->fd != -1)
            F_SET(infop, REGION_CREATED);

        /*
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

        /*
        * XXX
        * HP-UX won't permit mutexes to live in anything but shared memory.
        * Instantiate a shared region file on that architecture, regardless.
        *
        * XXX
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

可以看到这个函数打开或创建一个共享区域：如果是私有缓冲区则直接调用malloc分配指定缓冲区大小就足够了：

    ret = __os_malloc(infop->size, NULL, &infop->addr)) != 0

否则，就需要使用操作系统的共享内存机制：

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

然后调用__db_mapregion()将其映射到进程自己的地址空间，这样进程就可以对该共享区域进行访问了。

    if ((ret = __db_mapregion(infop->name, infop)) != 0)，定义在mp_region.c文件中：
    /*
    * __db_mapregion --
    *    Attach to a shared memory region.
    *
    * PUBLIC: int __db_mapregion __P((char *, REGINFO *));
    */
    int
    __db_mapregion(path, infop)
        char *path;
        REGINFO *infop;
    {
        int called, ret;

        called = 0;
        ret = EINVAL;

        /* If the user replaces the map call, call through their interface. */
        if (__db_jump.j_map != NULL) {
            F_SET(infop, REGION_HOLDINGSYS);
            return (__db_jump.j_map(path, infop->fd, infop->size,
                1, F_ISSET(infop, REGION_ANONYMOUS), 0, &infop->addr));
        }

        if (F_ISSET(infop, REGION_ANONYMOUS)) {
            /*
            * !!!
            * If we're creating anonymous regions:
            *
            * If it's private, we use mmap(2).  The problem with using
            * shmget(2) is that we may be creating a region of which the
            * application isn't aware, and if the application crashes
            * we'll have no way to remove the system resources for the
            * region.
            *
            * If it's not private, we use the shmget(2) interface if it's
            * available, because it allows us to name anonymous memory.
            * If shmget(2) isn't available, use the mmap(2) calls.
            *
            * In the case of anonymous memory, using mmap(2) means the
            * memory isn't named and only the single process and its
            * threads can access the region.
            */
    #ifdef    HAVE_MMAP
    #ifdef    MAP_ANON
    #define    HAVE_MMAP_ANONYMOUS    1
    #else
    #ifdef    MAP_ANONYMOUS
    #define    HAVE_MMAP_ANONYMOUS    1
    #endif
    #endif
    #endif
    #ifdef HAVE_MMAP_ANONYMOUS
            if (!called && F_ISSET(infop, REGION_PRIVATE)) {
                called = 1;
                ret = __os_map(path,
                    infop->fd, infop->size, 1, 1, 0, &infop->addr);
            }
    #endif
    #ifdef HAVE_SHMGET
            if (!called) {
                called = 1;
                ret = __os_shmget(infop);
            }
    #endif
    #ifdef HAVE_MMAP
            /*
            * If we're trying to join an unnamed anonymous region, fail --
            * that's not possible.
            */
            if (!called) {
                called = 1;

                if (!F_ISSET(infop, REGION_CREATED)) {
                    __db_err(infop->dbenv,
                    "cannot join region in unnamed anonymous memory");
                    return (EINVAL);
                }

                ret = __os_map(path,
                    infop->fd, infop->size, 1, 1, 0, &infop->addr);
            }
    #endif
        } else {
            /*
            * !!!
            * If we're creating normal regions, we use the mmap(2)
            * interface if it's available because it's POSIX 1003.1
            * standard and we trust it more than we do shmget(2).
            */
    #ifdef HAVE_MMAP
            if (!called) {
                called = 1;

                /* Mmap(2) regions that aren't anonymous can grow. */
                F_SET(infop, REGION_CANGROW);

                ret = __os_map(path,
                    infop->fd, infop->size, 1, 0, 0, &infop->addr);
            }
    #endif
    #ifdef HAVE_SHMGET
            if (!called) {
                called = 1;
                ret = __os_shmget(infop);
            }
    #endif
        }
        return (ret);
    }

在完成将共享区域映射到进程自己的地址空间之后，进程开始对其进行初始化：

    region_init:
        /*
        * Initialize the common region information.
        */

首先对共享区域通用信息进行初始化：

    rlp = (RLAYOUT *)infop->addr;

        /* Initialize the remaining region information. */
        rlp->refcnt = 1;
        rlp->size = infop->size;
        db_version(&rlp->majver, &rlp->minver, &rlp->patch);
        rlp->panic = 0;
        rlp->segid = infop->segid;
    ......


