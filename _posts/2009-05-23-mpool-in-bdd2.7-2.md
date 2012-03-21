---
layout: post
title: BDB中的共享区域──MPOOL
---

2009-5-23 星期六 大雨

#### 1. 数据结构

首先是DB_ENV{}和REGLAYOUT{}对于所有贡献区域都是一样的，在前面已经介绍过了，这里就不赘述了。
我们直接来看MPOOL共享区域数据结构MPOOL{}定义：

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

可以看到它根BDB1.8.6版本的MPOOL{}结构几乎就是一样的。关键在于它现在是共享区域了，而在1.8.6版本的时候还只是每进程私有的缓冲区。

介绍完缓冲区管理器的共享区域数据结构，现在让我们看一下每进程私有的数据结构：

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

注意到这个DB_MPOOL{}与DB_LOCKTAB{}结构是多么的相似，这是一个特点，每进程私有数据结构都含有一个指向共享区域数据结构的指针MPOOL *mp;一个描述共享区域的REGINFO reginfo字段。

#### 2. 初始化过程

现在让我们看一下初始化过程。

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
    #ifdef HAVE_SPINLOCKS
    #define    OKFLAGS    (DB_CREATE | DB_MPOOL_PRIVATE | DB_NOMMAP | DB_THREAD)
    #else
    #define    OKFLAGS    (DB_CREATE | DB_MPOOL_PRIVATE | DB_NOMMAP)
    #endif
        if ((ret = __db_fchk(dbenv, "memp_open", flags, OKFLAGS)) != 0)
            return (ret);

        /* Extract fields from DB_ENV structure. */
        cachesize = dbenv == NULL ? 0 : dbenv->mp_size;

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

在memp_open()中调用__memp_ropen()来真正加入或创建一个MPOOL区域：

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

我们看到与Lock区域一样，在加入或创建一个共享区域之前都需要构造名字来寻找以后区域或命名新区域。

    dbmp->reginfo.dbenv = dbmp->dbenv;
        dbmp->reginfo.appname = DB_APP_NONE;
        if (path == NULL)
            dbmp->reginfo.path = NULL;
        else
            if ((ret = __os_strdup(path, &dbmp->reginfo.path)) != 0)
                return (ret);
        dbmp->reginfo.file = DB_DEFAULT_MPOOL_FILE;

其中：                    /* Default mpool name. */

    #define    DB_DEFAULT_MPOOL_FILE    "__db_mpool.share"

appname仍为DB_APP_NONE表示寻找一个区域：

    *******************************************************/
    /* Type passed to __db_appname(). */
    typedef enum {
        DB_APP_NONE=0,            /* No type (region). */
        DB_APP_DATA,            /* Data file. */
        DB_APP_LOG,            /* Log file. */
        DB_APP_TMP            /* Temporary file. */
    } APPNAME;

后面接了一句注释，如果使用临时区域，那么不使用标准命名：

    /*
     * If we're creating a temporary region, don't use any standard
     * naming.
     */
    if (is_private) {
        dbmp->reginfo.appname = DB_APP_TMP;
        dbmp->reginfo.file = NULL;
        F_SET(&dbmp->reginfo, REGION_PRIVATE);
    }

开始调用__db_rattach()函数加入或创建一个区域：

    if ((ret = __db_rattach(&dbmp->reginfo)) != 0) {
        if (dbmp->reginfo.path != NULL)
            __os_freestr(dbmp->reginfo.path);
        return (ret);
    }

如果__db_rattach()返回成功，那么表示MPOOL区域已经正常创建。此时，MPOOL{}结构位于共享区域的开始，后面的区域都是自由空间，需要我们对其进行初始化。这个初始化逻辑与BDB1.8.6是相似的。

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

注意：细心的读者会注意到这里MPOOL共享区域的名字与是DB_HOME/DB_DEFAULT_MPOOL_FILE，即"/home/database/__db_mpool.share"。或如果是临时缓冲区的话，为DB_HOME/DB_TMP_DIR/<create>，即"/home/database/var/tmp"。也就是说根本不是它要缓存的数据库文件。这点与BDB1.8.6是很不同的。在1.8.6中，MPOOL内存区域是没有名字的，DB传给它需要缓存的文件的描述符fd，mpool_open()将返回一个MPOOL{}结构地址给DB。但是MPOOL仍然是缓存数据库文件页面。必然要告诉它需要缓存的文件，以便它读出和写入磁盘。这是通过另一个函数提供的：memp_fopen()，这也引出了MPOOL的另外几个重要的数据结构：MPOOLFILE{}，DB_MPOOL_FINFO{}和DB_MPOOLFILE{}。与前面一样，我们先介绍数据结构。

###### 1. MPOOLFILE{}

    /*
    * MPOOLFILE --
    *    Shared DB_MPOOLFILE information.
    */
    struct __mpoolfile {
        SH_TAILQ_ENTRY  q;        /* List of MPOOLFILEs */

        u_int32_t ref;            /* Reference count. */

        int      ftype;        /* File type. */

        int32_t      lsn_off;        /* Page's LSN offset. */
        u_int32_t clear_len;        /* Bytes to clear on page create. */

        size_t      path_off;        /* File name location. */
        size_t      fileid_off;        /* File identification location. */

        size_t      pgcookie_len;        /* Pgin/pgout cookie length. */
        size_t      pgcookie_off;        /* Pgin/pgout cookie location. */

        u_int32_t lsn_cnt;        /* Checkpoint buffers left to write. */

        db_pgno_t last_pgno;        /* Last page in the file. */
        db_pgno_t orig_last_pgno;    /* Original last page in the file. */

    #define    MP_CAN_MMAP    0x01        /* If the file can be mmap'd. */
    #define    MP_TEMP        0x02        /* Backing file is a temporary. */
        u_int32_t  flags;

        DB_MPOOL_FSTAT stat;        /* Per-file mpool statistics. */
    };

###### 2.  DB_MPOOLFILE{}：这个就很像1.8.6中包含在MPOOL中的缓存文件的信息。

    /*
    * DB_MPOOLFILE --
    *    Per-process DB_MPOOLFILE information.
    */
    struct __db_mpoolfile {
    /* These fields need to be protected for multi-threaded support. */
        db_mutex_t    *mutexp;    /* Structure lock. */

        int       fd;            /* Underlying file descriptor. */

        u_int32_t ref;            /* Reference count. */

        /*
        * !!!
        * This field is a special case -- it's protected by the region lock
        * NOT the thread lock.  The reason for this is that we always have
        * the region lock immediately before or after we modify the field,
        * and we don't want to use the structure lock to protect it because
        * then I/O (which is done with the structure lock held because of
        * the race between the seek and write of the file descriptor) will
        * block any other put/get calls using this DB_MPOOLFILE structure.
        */
        u_int32_t pinref;        /* Pinned block reference count. */

    /* These fields are not protected. */
        TAILQ_ENTRY(__db_mpoolfile) q;    /* Linked list of DB_MPOOLFILE's. */

        DB_MPOOL  *dbmp;        /* Overlying DB_MPOOL. */
        MPOOLFILE *mfp;            /* Underlying MPOOLFILE. */

        void      *addr;        /* Address of mmap'd region. */
        size_t       len;            /* Length of mmap'd region. */

    /* These fields need to be protected for multi-threaded support. */
    #define    MP_READONLY    0x01        /* File is readonly. */
    #define    MP_UPGRADE    0x02        /* File descriptor is readwrite. */
    #define    MP_UPGRADE_FAIL    0x04        /* Upgrade wasn't possible. */
        u_int32_t  flags;
    };

###### 3. DB_MPOOL_FINFO{}

    /* Mpool file open information structure. */
    struct __db_mpool_finfo {
        int       ftype;        /* File type. */
        DBT      *pgcookie;        /* Byte-string passed to pgin/pgout. */
        u_int8_t  *fileid;        /* Unique file ID. */
        int32_t       lsn_offset;        /* LSN offset in page. */
        u_int32_t  clear_len;        /* Cleared length on created pages. */
    };

现在我们可以开始看一下memp_open（）函数了：

    /*
    * memp_fopen --
    *    Open a backing file for the memory pool.
    */
    int
    memp_fopen(dbmp, path, flags, mode, pagesize, finfop, retp)
        DB_MPOOL *dbmp;
        const char *path;
        u_int32_t flags;
        int mode;
        size_t pagesize;
        DB_MPOOL_FINFO *finfop;
        DB_MPOOLFILE **retp;
    {
        int ret;

        MP_PANIC_CHECK(dbmp);

        /* Validate arguments. */
        if ((ret = __db_fchk(dbmp->dbenv,
            "memp_fopen", flags, DB_CREATE | DB_NOMMAP | DB_RDONLY)) != 0)
            return (ret);

        /* Require a non-zero pagesize. */
        if (pagesize == 0) {
            __db_err(dbmp->dbenv, "memp_fopen: pagesize not specified");
            return (EINVAL);
        }
        if (finfop != NULL && finfop->clear_len > pagesize)
            return (EINVAL);

        return (__memp_fopen(dbmp,
            NULL, path, flags, mode, pagesize, 1, finfop, retp));
    }

    /*
    * __memp_fopen --
    *    Open a backing file for the memory pool; internal version.
    *
    * PUBLIC: int __memp_fopen __P((DB_MPOOL *, MPOOLFILE *, const char *,
    * PUBLIC:    u_int32_t, int, size_t, int, DB_MPOOL_FINFO *, DB_MPOOLFILE **));
    */
    int
    __memp_fopen(dbmp, mfp, path, flags, mode, pagesize, needlock, finfop, retp)
        DB_MPOOL *dbmp;
        MPOOLFILE *mfp;
        const char *path;
        u_int32_t flags;
        int mode, needlock;
        size_t pagesize;
        DB_MPOOL_FINFO *finfop;
        DB_MPOOLFILE **retp;
    {
        DB_ENV *dbenv;
        DB_MPOOLFILE *dbmfp;
        DB_MPOOL_FINFO finfo;
        db_pgno_t last_pgno;
        size_t maxmap;
        u_int32_t mbytes, bytes;
        int ret;
        u_int8_t idbuf[DB_FILE_ID_LEN];
        char *rpath;

        dbenv = dbmp->dbenv;
        ret = 0;
        rpath = NULL;

        /*
        * If mfp is provided, we take the DB_MPOOL_FINFO information from
        * the mfp.  We don't bother initializing everything, because some
        * of them are expensive to acquire.  If no mfp is provided and the
        * finfop argument is NULL, we default the values.
        */
        if (finfop == NULL) {
            memset(&finfo, 0, sizeof(finfo));
            if (mfp != NULL) {
                finfo.ftype = mfp->ftype;
                finfo.pgcookie = NULL;
                finfo.fileid = NULL;
                finfo.lsn_offset = mfp->lsn_off;
                finfo.clear_len = mfp->clear_len;
            } else {
                finfo.ftype = 0;
                finfo.pgcookie = NULL;
                finfo.fileid = NULL;
                finfo.lsn_offset = -1;
                finfo.clear_len = 0;
            }
            finfop = &finfo;
        }

        /* Allocate and initialize the per-process structure. */
        if ((ret = __os_calloc(1, sizeof(DB_MPOOLFILE), &dbmfp)) != 0)
            return (ret);
        dbmfp->dbmp = dbmp;
        dbmfp->fd = -1;
        dbmfp->ref = 1;
        if (LF_ISSET(DB_RDONLY))
            F_SET(dbmfp, MP_READONLY);

        if (path == NULL) {
            if (LF_ISSET(DB_RDONLY)) {
                __db_err(dbenv,
                    "memp_fopen: temporary files can't be readonly");
                ret = EINVAL;
                goto err;
            }
            last_pgno = 0;
        } else {
            /* Get the real name for this file and open it. */
            if ((ret = __db_appname(dbenv,
                DB_APP_DATA, NULL, path, 0, NULL, &rpath)) != 0)
                goto err;
            if ((ret = __db_open(rpath,
            LF_ISSET(DB_CREATE | DB_RDONLY),
            DB_CREATE | DB_RDONLY, mode, &dbmfp->fd)) != 0) {
                __db_err(dbenv, "%s: %s", rpath, strerror(ret));
                goto err;
            }

            /*
            * Don't permit files that aren't a multiple of the pagesize,
            * and find the number of the last page in the file, all the
            * time being careful not to overflow 32 bits.
            *
            * !!!
            * We can't use off_t's here, or in any code in the mainline
            * library for that matter.  (We have to use them in the os
            * stubs, of course, as there are system calls that take them
            * as arguments.)  The reason is that some customers build in
            * environments where an off_t is 32-bits, but still run where
            * offsets are 64-bits, and they pay us a lot of money.
            */
            if ((ret = __os_ioinfo(rpath,
                dbmfp->fd, &mbytes, &bytes, NULL)) != 0) {
                __db_err(dbenv, "%s: %s", rpath, strerror(ret));
                goto err;
            }

            /* Page sizes have to be a power-of-two, ignore mbytes. */
            if (bytes % pagesize != 0) {
                __db_err(dbenv,
                    "%s: file size not a multiple of the pagesize",
                    rpath);
                ret = EINVAL;
                goto err;
            }

            last_pgno = mbytes * (MEGABYTE / pagesize);
            last_pgno += bytes / pagesize;

            /* Correction: page numbers are zero-based, not 1-based. */
            if (last_pgno != 0)
                --last_pgno;

            /*
            * Get the file id if we weren't given one.  Generated file id's
            * don't use timestamps, otherwise there'd be no chance of any
            * other process joining the party.
            */
            if (finfop->fileid == NULL) {
                if ((ret = __os_fileid(dbenv, rpath, 0, idbuf)) != 0)
                    goto err;
                finfop->fileid = idbuf;
            }
        }

        /*
        * If we weren't provided an underlying shared object to join with,
        * find/allocate the shared file objects.  Also allocate space for
        * for the per-process thread lock.
        */
        if (needlock)
            LOCKREGION(dbmp);

        if (mfp == NULL)
            ret = __memp_mf_open(dbmp,
                path, pagesize, last_pgno, finfop, &mfp);
        else {
            ++mfp->ref;
            ret = 0;
        }
        if (ret == 0 &&
            F_ISSET(dbmp, MP_LOCKHANDLE) && (ret =
            __memp_alloc(dbmp, sizeof(db_mutex_t), NULL, &dbmfp->mutexp)) == 0)
            LOCKINIT(dbmp, dbmfp->mutexp);

        if (needlock)
            UNLOCKREGION(dbmp);
        if (ret != 0)
            goto err;

        dbmfp->mfp = mfp;

        /*
        * If a file:
        *    + is read-only
        *    + isn't temporary
        *    + doesn't require any pgin/pgout support
        *    + the DB_NOMMAP flag wasn't set
        *    + and is less than mp_mmapsize bytes in size
        *
        * we can mmap it instead of reading/writing buffers.  Don't do error
        * checking based on the mmap call failure.  We want to do normal I/O
        * on the file if the reason we failed was because the file was on an
        * NFS mounted partition, and we can fail in buffer I/O just as easily
        * as here.
        *
        * XXX
        * We'd like to test to see if the file is too big to mmap.  Since we
        * don't know what size or type off_t's or size_t's are, or the largest
        * unsigned integral type is, or what random insanity the local C
        * compiler will perpetrate, doing the comparison in a portable way is
        * flatly impossible.  Hope that mmap fails if the file is too large.
        */
    #define    DB_MAXMMAPSIZE    (10 * 1024 * 1024)    /* 10 Mb. */
        if (F_ISSET(mfp, MP_CAN_MMAP)) {
            if (!F_ISSET(dbmfp, MP_READONLY))
                F_CLR(mfp, MP_CAN_MMAP);
            if (path == NULL)
                F_CLR(mfp, MP_CAN_MMAP);
            if (finfop->ftype != 0)
                F_CLR(mfp, MP_CAN_MMAP);
            if (LF_ISSET(DB_NOMMAP))
                F_CLR(mfp, MP_CAN_MMAP);
            maxmap = dbenv == NULL || dbenv->mp_mmapsize == 0 ?
                DB_MAXMMAPSIZE : dbenv->mp_mmapsize;
            if (mbytes > maxmap / MEGABYTE ||
                (mbytes == maxmap / MEGABYTE && bytes >= maxmap % MEGABYTE))
                F_CLR(mfp, MP_CAN_MMAP);
        }
        dbmfp->addr = NULL;
        if (F_ISSET(mfp, MP_CAN_MMAP)) {
            dbmfp->len = (size_t)mbytes * MEGABYTE + bytes;
            if (__db_mapfile(rpath,
                dbmfp->fd, dbmfp->len, 1, &dbmfp->addr) != 0) {
                dbmfp->addr = NULL;
                F_CLR(mfp, MP_CAN_MMAP);
            }
        }
        if (rpath != NULL)
            __os_freestr(rpath);

        LOCKHANDLE(dbmp, dbmp->mutexp);
        TAILQ_INSERT_TAIL(&dbmp->dbmfq, dbmfp, q);
        UNLOCKHANDLE(dbmp, dbmp->mutexp);

        *retp = dbmfp;
        return (0);

    err:    /*
        * Note that we do not have to free the thread mutex, because we
        * never get to here after we have successfully allocated it.
        */
        if (rpath != NULL)
            __os_freestr(rpath);
        if (dbmfp->fd != -1)
            (void)__os_close(dbmfp->fd);
        if (dbmfp != NULL)
            __os_free(dbmfp, sizeof(DB_MPOOLFILE));
        return (ret);
    }

3. 函数调用

