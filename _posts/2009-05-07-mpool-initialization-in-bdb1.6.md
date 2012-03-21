---
layout: post
title: BDB1.6中的初始化过程──MPOOL初始化
---

我们前面的讨论都是基于BDB2.7.7版本的，那时候的BDB已经是一个支持事务处理的相当复杂的模块了。现在让我们回过头来看一下BDB1.6版本的初始化过程，那时候的共享区域就只有MPOOL。

首先其入口函数是dbopen(...)函数:

    DB * dbopen(fname, flags, mode, type, openinfo)
        const char *fname;
        int flags, mode;
        DBTYPE type;
        const void *openinfo;
    {

    #define    DB_FLAGS    (DB_LOCK | DB_SHMEM | DB_TXN)
    #define    USE_OPEN_FLAGS                            \
        (O_CREAT | O_EXCL | O_EXLOCK | O_NONBLOCK | O_RDONLY |        \
        O_RDWR | O_SHLOCK | O_TRUNC)

        if ((flags & ~(USE_OPEN_FLAGS | DB_FLAGS)) == 0)
            switch (type) {
            case DB_BTREE:
                return (__bt_open(fname, flags & USE_OPEN_FLAGS,
                    mode, openinfo, flags & DB_FLAGS));
            case DB_HASH:
                return (__hash_open(fname, flags & USE_OPEN_FLAGS,
                    mode, openinfo, flags & DB_FLAGS));
            case DB_RECNO:
                return (__rec_open(fname, flags & USE_OPEN_FLAGS,
                    mode, openinfo, flags & DB_FLAGS));
            }
        errno = EINVAL;
        return (NULL);
    }

这个函数根据用户指定的DBTYPE参数调用相应的存取方法的打开函数，我们以B树存取方法为例：
            case DB_BTREE:
                return (__bt_open(fname, flags & USE_OPEN_FLAGS,
                    mode, openinfo, flags & DB_FLAGS));
具体代码如下：

    /***************** bt_open.c ********************/
    /*
    * __BT_OPEN -- Open a btree.
    *
    * Creates and fills a DB struct, and calls the routine that actually
    * opens the btree.
    *
    * Parameters:
    *    fname:    filename (NULL for in-memory trees)
    *    flags:    open flag bits
    *    mode:    open permission bits
    *    b:    BTREEINFO pointer
    *
    * Returns:
    *    NULL on failure, pointer to DB on success.
    *
    */
    DB *
    __bt_open(fname, flags, mode, openinfo, dflags)
        const char *fname;
        int flags, mode, dflags;
        const BTREEINFO *openinfo;
    {
        struct stat sb;
        BTMETA m;
        BTREE *t;
        BTREEINFO b;
        DB *dbp;
        pgno_t ncache;
        ssize_t nr;
        int machine_lorder;

        t = NULL;

        /*
        * Intention is to make sure all of the user's selections are okay
        * here and then use them without checking.  Can't be complete, since
        * we don't know the right page size, lorder or flags until the backing
        * file is opened.  Also, the file's page size can cause the cachesize
        * to change.
        */
        machine_lorder = byteorder();
        if (openinfo) {
            b = *openinfo;

            /* Flags: R_DUP. */
            if (b.flags & ~(R_DUP))
                goto einval;

            /*
            * Page size must be indx_t aligned and >= MINPSIZE.  Default
            * page size is set farther on, based on the underlying file
            * transfer size.
            */
            if (b.psize &&
                (b.psize < MINPSIZE || b.psize > MAX_PAGE_OFFSET + 1 ||
                b.psize & sizeof(indx_t) - 1))
                goto einval;

            /* Minimum number of keys per page; absolute minimum is 2. */
            if (b.minkeypage) {
                if (b.minkeypage < 2)
                    goto einval;
            } else
                b.minkeypage = DEFMINKEYPAGE;

            /* If no comparison, use default comparison and prefix. */
            if (b.compare == NULL) {
                b.compare = __bt_defcmp;
                if (b.prefix == NULL)
                    b.prefix = __bt_defpfx;
            }

            if (b.lorder == 0)
                b.lorder = machine_lorder;
        } else {
            b.compare = __bt_defcmp;
            b.cachesize = 0;
            b.flags = 0;
            b.lorder = machine_lorder;
            b.minkeypage = DEFMINKEYPAGE;
            b.prefix = __bt_defpfx;
            b.psize = 0;
        }

        /* Check for the ubiquitous PDP-11. */
        if (b.lorder != BIG_ENDIAN && b.lorder != LITTLE_ENDIAN)
            goto einval;

    /* 创建并初始化DB{}和BTREE{}结构，这两者是继承的关系，具体参见笔记*/
        /* Allocate and initialize DB and BTREE structures. */
        if ((t = (BTREE *)malloc(sizeof(BTREE))) == NULL)
            goto err;
        memset(t, 0, sizeof(BTREE));
        t->bt_fd = -1;            /* Don't close unopened fd on error. */
        t->bt_lorder = b.lorder;
        t->bt_order = NOT;
        t->bt_cmp = b.compare;
        t->bt_pfx = b.prefix;
        t->bt_rfd = -1;

        if ((t->bt_dbp = dbp = (DB *)malloc(sizeof(DB))) == NULL)
            goto err;
        memset(t->bt_dbp, 0, sizeof(DB));
        if (t->bt_lorder != machine_lorder)
            F_SET(t, B_NEEDSWAP);

        dbp->type = DB_BTREE;
        dbp->internal = t;
        dbp->close = __bt_close;
        dbp->del = __bt_delete;
        dbp->fd = __bt_fd;
        dbp->get = __bt_get;
        dbp->put = __bt_put;
        dbp->seq = __bt_seq;
        dbp->sync = __bt_sync;

        /*
        * If no file name was supplied, this is an in-memory btree and we
        * open a backing temporary file.  Otherwise, it's a disk-based tree.
        */
        if (fname) {
            switch (flags & O_ACCMODE) {
            case O_RDONLY:
                F_SET(t, B_RDONLY);
                break;
            case O_RDWR:
                break;
            case O_WRONLY:
            default:
                goto einval;
            }
            打开数据库文件
            if ((t->bt_fd = open(fname, flags, mode)) < 0)
                goto err;

        } else {
            if ((flags & O_ACCMODE) != O_RDWR)
                goto einval;
            if ((t->bt_fd = tmp()) == -1)
                goto err;
            F_SET(t, B_INMEM);
        }

        if (fcntl(t->bt_fd, F_SETFD, 1) == -1)
            goto err;

        if (fstat(t->bt_fd, &sb))
            goto err;
        if (sb.st_size) {
            if ((nr = read(t->bt_fd, &m, sizeof(BTMETA))) < 0)
                goto err;
            if (nr != sizeof(BTMETA))
                goto eftype;

        /** 读入元数据，根据里面的信息（如页面大小等）建立缓冲区共享区域 **/
            /*
            * Read in the meta-data.  This can change the notion of what
            * the lorder, page size and flags are, and, when the page size
            * changes, the cachesize value can change too.  If the user
            * specified the wrong byte order for an existing database, we
            * don't bother to return an error, we just clear the NEEDSWAP
            * bit.
            */
            if (m.magic == BTREEMAGIC)
                F_CLR(t, B_NEEDSWAP);
            else {
                F_SET(t, B_NEEDSWAP);
                M_32_SWAP(m.magic);
                M_32_SWAP(m.version);
                M_32_SWAP(m.psize);
                M_32_SWAP(m.free);
                M_32_SWAP(m.nrecs);
                M_32_SWAP(m.flags);
            }
            if (m.magic != BTREEMAGIC || m.version != BTREEVERSION)
                goto eftype;
            if (m.psize < MINPSIZE || m.psize > MAX_PAGE_OFFSET + 1 ||
                m.psize & sizeof(indx_t) - 1)
                goto eftype;
            if (m.flags & ~SAVEMETA)
                goto eftype;
            b.psize = m.psize;
            F_SET(t, m.flags);
            t->bt_free = m.free;
            t->bt_nrecs = m.nrecs;
        } else {
            /*
            * Set the page size to the best value for I/O to this file.
            * Don't overflow the page offset type.
            */
            if (b.psize == 0) {
                b.psize = sb.st_blksize;
                if (b.psize < MINPSIZE)
                    b.psize = MINPSIZE;
                if (b.psize > MAX_PAGE_OFFSET + 1)
                    b.psize = MAX_PAGE_OFFSET + 1;
            }

            /* Set flag if duplicates permitted. */
            if (!(b.flags & R_DUP))
                F_SET(t, B_NODUPS);

            t->bt_free = P_INVALID;
            t->bt_nrecs = 0;
            F_SET(t, B_METADIRTY);
        }

        t->bt_psize = b.psize;

        /* Set the cache size; must be a multiple of the page size. */
        if (b.cachesize && b.cachesize & b.psize - 1)
            b.cachesize += (~b.cachesize & b.psize - 1) + 1;
        if (b.cachesize < b.psize * MINCACHE)
            b.cachesize = b.psize * MINCACHE;

        /* Calculate number of pages to cache. */
        ncache = (b.cachesize + t->bt_psize - 1) / t->bt_psize;

        /*
        * The btree data structure requires that at least two keys can fit on
        * a page, but other than that there's no fixed requirement.  The user
        * specified a minimum number per page, and we translated that into the
        * number of bytes a key/data pair can use before being placed on an
        * overflow page.  This calculation includes the page header, the size
        * of the index referencing the leaf item and the size of the leaf item
        * structure.  Also, don't let the user specify a minkeypage such that
        * a key/data pair won't fit even if both key and data are on overflow
        * pages.
        */
        t->bt_ovflsize = (t->bt_psize - BTDATAOFF) / b.minkeypage -
            (sizeof(indx_t) + NBLEAFDBT(0, 0));
        if (t->bt_ovflsize < NBLEAFDBT(NOVFLSIZE, NOVFLSIZE) + sizeof(indx_t))
            t->bt_ovflsize =
                NBLEAFDBT(NOVFLSIZE, NOVFLSIZE) + sizeof(indx_t);

        /*** 现在可以更加提取到的元数据创建缓冲区共享区域了 ***/
        /* Initialize the buffer pool. */
        if ((t->bt_mp =
            mpool_open(NULL, t->bt_fd, t->bt_psize, ncache)) == NULL)
            goto err;
        if (!F_ISSET(t, B_INMEM))
            mpool_filter(t->bt_mp, __bt_pgin, __bt_pgout, t);

        /* Create a root page if new tree. */
        if (nroot(t) == RET_ERROR)
            goto err;

        /* Global flags. */
        if (dflags & DB_LOCK)
            F_SET(t, B_DB_LOCK);
        if (dflags & DB_SHMEM)
            F_SET(t, B_DB_SHMEM);
        if (dflags & DB_TXN)
            F_SET(t, B_DB_TXN);

        return (dbp);

    einval:    errno = EINVAL;
        goto err;

    eftype:    errno = EFTYPE;
        goto err;

    err:    if (t) {
            if (t->bt_dbp)
                free(t->bt_dbp);
            if (t->bt_fd != -1)
                (void)close(t->bt_fd);
            free(t);
        }
        return (NULL);
    }

我们看到B树的打开方法现分配DB{}和BTREE{}数据结构并初始化它们（这两者是抽象与具体的继承关系）。然后它打开指定的数据库文件，读进元数据，根据这些元数据提供的信息创建MPOOL共享区域。

    /*
    * mpool_open --
    *    Initialize a memory pool.
    */
    MPOOL *
    mpool_open(key, fd, pagesize, maxcache)
        void *key;
        int fd;
        pgno_t pagesize, maxcache;
    {
        struct stat sb;
        MPOOL *mp;
        int entry;

        /*
        * Get information about the file.
        *
        * XXX
        * We don't currently handle pipes, although we should.
        */
        if (fstat(fd, &sb))
            return (NULL);
        if (!S_ISREG(sb.st_mode)) {
            errno = ESPIPE;
            return (NULL);
        }

        /* Allocate and initialize the MPOOL cookie. */
        if ((mp = (MPOOL *)calloc(1, sizeof(MPOOL))) == NULL)
            return (NULL);
        CIRCLEQ_INIT(&mp->lqh);
        for (entry = 0; entry < HASHSIZE; ++entry)
            CIRCLEQ_INIT(&mp->hqh[entry]);
        mp->maxcache = maxcache;
        mp->npages = sb.st_size / pagesize;
        mp->pagesize = pagesize;
        mp->fd = fd;
        return (mp);
    }

从这里我们可以看到不像2.7.7，这里MPOOL{}数据结构并不是共享内存区域，而是每一个进程私有的内存空间，因为它是通过calloc分配的。

    typedef struct MPOOL {
        CIRCLEQ_HEAD(_lqh, _bkt) lqh;    /* lru queue head */
                        /* hash queue array */
        CIRCLEQ_HEAD(_hqh, _bkt) hqh[HASHSIZE];
        pgno_t    curcache;        /* current number of cached pages */
        pgno_t    maxcache;        /* max number of cached pages */
        pgno_t    npages;            /* number of pages in the file */
        u_long    pagesize;        /* file page size */
        int    fd;            /* file descriptor */
                        /* page in conversion routine */
        void    (*pgin) __P((void *, pgno_t, void *));
                        /* page out conversion routine */
        void    (*pgout) __P((void *, pgno_t, void *));
        void    *pgcookie;        /* cookie for page in/out routines */
    #ifdef STATISTICS
        u_long    cachehit;
        u_long    cachemiss;
        u_long    pagealloc;
        u_long    pageflush;
        u_long    pageget;
        u_long    pagenew;
        u_long    pageput;
        u_long    pageread;
        u_long    pagewrite;
    #endif
    } MPOOL;

MPOOL{}结构中有个rlu队列：

    CIRCLEQ_HEAD(_lqh, _bkt) lqh;

其中CIRCLEQ_HEAD宏定义如下：
    /*
    * Circular queue definitions.
    */
    #define CIRCLEQ_HEAD(name, type)                    \
    struct name {                                \
        struct type *cqh_first;        /* first element */        \
        struct type *cqh_last;        /* last element */        \
    }

替换成：

    struct _lqh{
        struct _bkt * cqh_first;
        struct _bkt* cqh_last;
    }

可见它是含有一个指向结构_bkt{}的头部和尾部指针的结构。

还有一个散列表：

    CIRCLEQ_HEAD(_hqh, _bkt) hqh[HASHSIZE];

展开成：

    struct _hqh{
        struct _bkt* cqh_first;
        struct _bkt* cqh_last;
    }hqh[HASHSIZE];
即是一个有HASHSIZE个指向_bkt{}结构的数组。

也就是说这里MPOOL的双向队列中每一个元素都是一个桶_bkt{}结构，定义如下（在mpool.h文件中）：
    /*
    * The memory pool scheme is a simple one.  Each in-memory page is referenced
    * by a bucket which is threaded in up to two of three ways.  All active pages
    * are threaded on a hash chain (hashed by page number) and an lru chain.
    * Inactive pages are threaded on a free chain.  Each reference to a memory
    * pool is handed an opaque MPOOL cookie which stores all of this information.
    */
    #define    HASHSIZE    128
    #define    HASHKEY(pgno)    ((pgno - 1 + HASHSIZE) % HASHSIZE)

    /* The BKT structures are the elements of the queues. */
    typedef struct _bkt {
        CIRCLEQ_ENTRY(_bkt) hq;        /* hash queue */
        CIRCLEQ_ENTRY(_bkt) q;        /* lru queue */
        void    *page;            /* page */
        pgno_t   pgno;            /* page number */

    #define    MPOOL_DIRTY    0x01        /* page needs to be written */
    #define    MPOOL_PINNED    0x02        /* page is pinned into memory */
    #define    MPOOL_INUSE    0x04        /* page address is valid */
        u_int8_t flags;            /* flags */
    } BKT;

可以看到每个_bkt{}结构中包含如下元素：

用于散列表开环的指针hq，指向下一个_bkt元素：

    CIRCLEQ_ENTRY(_bkt) hq;        /* hash queue */

和用于lru队列的指针q，指向lru队列中的下一个元素：

    CIRCLEQ_ENTRY(_bkt) q;        /* lru queue */

其中CIRCLEQ_ENTRY宏定义如下：

    #define CIRCLEQ_ENTRY(type)                        \
    struct {                                \
        struct type *cqe_next;        /* next element */        \
        struct type *cqe_prev;        /* previous element */        \
    }

这些是双向列表的要求，这里由于一个_bkt{}元素可以（往往）既是属于rlu队列，又属于hash队列，所以需要两个指针字段。

下面的字段是真正用于缓存的字段：

一个指向页面数据的指针：

    void* page;  /* page */

和该页面的页面编号：

    pgno_t pgno;  /* page number */

我们在后面会看到页面有自己的结构定义，其中就包含了页面编号pgno，这里包含页面编号的原因在于通过pgno的散列并不是一一映射的，所以需要开环散列，需要沿着开环链比较下去。

另外，我们也可以看到这里的_bkt{}结构就好像页面的控制块信息，它建立起页面编号与页面地址的关系以用于页面查找，并且用于决定页面替换策略。这个实现与UNIX的文件系统缓冲区实现是十分类型的。
真正的页面空间在page指向的内存区域，这个内存区域是在mpool_bkt(...)函数中创建的：

    new:    if ((bp = (BKT *)malloc(sizeof(BKT) + mp->pagesize)) == NULL)
            return (NULL);

并且在创建后有这么一句：

    bp->page = (char *)bp + sizeof(BKT);

这显示了作为页面控制块的BKT{}结构和其控制的页面在内存上的布局是紧挨着的，就像头部（header）和实体（body）一样。

说明：为什么采用这种内存布局方式，而不是将头部控制块信息与页面数据信息分别分配呢，既然前者是MPOOL模块内部使用，而后者是分配给用户的缓冲区区域？答案是为了保持用户的使用透明性。如果我们仅仅将分配的页面缓冲区地址返回给用户，那么当用户返回该缓冲区给MPOOL的时候，由于该缓冲区是void* ，由用户自由填充，MPOOL无法找到其相应的控制块，这样就没有办法回收了。但是如果采用控制块与缓冲区页面紧挨的做法，那么MPOOL在收到一个缓冲区起始地址时，它只要向前移动sizeof(BKT)就可以找到该页面的控制块信息了。事实上，这种做法正是libc中malloc的实现。采用这种实现，用户对于控制块完全透明。因此接口为：

    void    *mpool_new __P((MPOOL *, pgno_t *, u_int));
    void    *mpool_get __P((MPOOL *, pgno_t, u_int));
    int     mpool_delete __P((MPOOL *, void *));
    int     mpool_put __P((MPOOL *, void *, u_int));

而在Gray中不是采用这种做法，所以它的释放页面缓冲区的接口与申请页面缓冲区接口一样，都要将页面编号进行散列比较得到相应的页面控制块，才能释放该页面：

    typedef struct{
        PAGEID pageid;
        PAGEPTR pageaddress;
        int index;
        semaphore* sme;
        Boolean modifies;
        Boolean invalid;
    }BUFFER_ACC_CB;

    Boolean bufferunfix(BUFFER_ACC_CBP);

整个mpool_bkt()函数是关键，所以将其实现完整地列在下面：
    /*
    * mpool_bkt
    *    Get a page from the cache (or create one).
    */
    static BKT *
    mpool_bkt(mp)
        MPOOL *mp;
    {
        struct _hqh *head;
        BKT *bp;

        /* If under the max cached, always create a new page. */
        if (mp->curcache < mp->maxcache)
            goto new;

        /*
        * If the cache is max'd out, walk the lru list for a buffer we
        * can flush.  If we find one, write it (if necessary) and take it
        * off any lists.  If we don't find anything we grow the cache anyway.
        * The cache never shrinks.
        */
        for (bp = mp->lqh.cqh_first;
            bp != (void *)&mp->lqh; bp = bp->q.cqe_next)
            if (!(bp->flags & MPOOL_PINNED)) {
                /* Flush if dirty. */
                if (bp->flags & MPOOL_DIRTY &&
                    mpool_write(mp, bp) == RET_ERROR)
                    return (NULL);
    #ifdef STATISTICS
                ++mp->pageflush;
    #endif
                /* Remove from the hash and lru queues. */
                head = &mp->hqh[HASHKEY(bp->pgno)];
                CIRCLEQ_REMOVE(head, bp, hq);
                CIRCLEQ_REMOVE(&mp->lqh, bp, q);
    #ifdef DEBUG
                { void *spage;
                    spage = bp->page;
                    memset(bp, 0xff, sizeof(BKT) + mp->pagesize);
                    bp->page = spage;
                }
    #endif
                bp->flags = 0;
                return (bp);
            }

    new:    if ((bp = (BKT *)malloc(sizeof(BKT) + mp->pagesize)) == NULL)
            return (NULL);
    #ifdef STATISTICS
        ++mp->pagealloc;
    #endif
    #if defined(DEBUG) || defined(PURIFY)
        memset(bp, 0xff, sizeof(BKT) + mp->pagesize);
    #endif
        bp->page = (char *)bp + sizeof(BKT);
        bp->flags = 0;
        ++mp->curcache;
        return (bp);
    }

最后，让我们看一下作为实体的页面是怎样定义的：显然这跟存取路径与文件组织有关，对于B树它需要存放B树节点，而且每一个节点就是一个页面。我们仍然以B树为例：

在btree.h文件中有如下每个页面的页头数据结构PAGE{}定义：

    /*
    * Page 0 of a btree file contains a copy of the meta-data.  This page is also
    * used as an out-of-band page, i.e. page pointers that point to nowhere point
    * to page 0.  Page 1 is the root of the btree.
    */
    #define    P_INVALID     0        /* Invalid tree page number. */
    #define    P_META         0        /* Tree metadata page number. */
    #define    P_ROOT         1        /* Tree root page number. */

    /*
    * There are five page layouts in the btree: btree internal pages (BINTERNAL),
    * btree leaf pages (BLEAF), recno internal pages (RINTERNAL), recno leaf pages
    * (RLEAF) and overflow pages.  All five page types have a page header (PAGE).
    * This implementation requires that values within structures NOT be padded.
    * (ANSI C permits random padding.)  If your compiler pads randomly you'll have
    * to do some work to get this package to run.
    */
    typedef struct _page {
        pgno_t    pgno;            /* this page's page number */
        pgno_t    prevpg;            /* left sibling */
        pgno_t    nextpg;            /* right sibling */

    #define    P_BINTERNAL    0x01        /* btree internal page */
    #define    P_BLEAF        0x02        /* leaf page */
    #define    P_OVERFLOW    0x04        /* overflow page */
    #define    P_RINTERNAL    0x08        /* recno internal page */
    #define    P_RLEAF        0x10        /* leaf page */
    #define P_TYPE        0x1f        /* type mask */
    #define    P_PRESERVE    0x20        /* never delete this chain of pages */
        u_int32_t flags;

        indx_t    lower;            /* lower bound of free space on page */
        indx_t    upper;            /* upper bound of free space on page */
        indx_t    linp[1];        /* indx_t-aligned VAR. LENGTH DATA */
    } PAGE;

页面中的记录的格式又根据B树节点类型不同而不同。这里说有5中节点类型，但是其中有两种是recno存取方法的。我们对此不与理会。

首先是B树内部节点记录定义：

    /*
    * For the btree internal pages, the item is a key.  BINTERNALs are {key, pgno}
    * pairs, such that the key compares less than or equal to all of the records
    * on that page.  For a tree without duplicate keys, an internal page with two
    * consecutive keys, a and b, will have all records greater than or equal to a
    * and less than b stored on the page associated with a.  Duplicate keys are
    * somewhat special and can cause duplicate internal and leaf page records and
    * some minor modifications of the above rule.
    */
    typedef struct _binternal {
        u_int32_t ksize;        /* key size */
        pgno_t    pgno;            /* page number stored on */
    #define    P_BIGDATA    0x01        /* overflow data */
    #define    P_BIGKEY    0x02        /* overflow key */
        u_char    flags;
        char    bytes[1];        /* data */
    } BINTERNAL;

然后是叶子节点记录定义：

    /* For the btree leaf pages, the item is a key and data pair. */
    typedef struct _bleaf {
        u_int32_t    ksize;        /* size of key */
        u_int32_t    dsize;        /* size of data */
        u_char    flags;            /* P_BIGDATA, P_BIGKEY */
        char    bytes[1];        /* data */
    } BLEAF;

最后是一种特殊的页面：溢出页（overflow page），是一个只有页头（PAGE{}）的页面。

说明：如果将元数据的存放页面也算进B树的页面的话，那么还有一个BTMETA{}结构：

    typedef struct _btmeta{
        u_int32_t magic;
        u_int32_t version;
        u_int32_t psize;
        u_int32_t free;
        u_int32_t flags;
    }BTMETA;

其中最重要的两个字段是free和psize。

一个用于使用自由链表管理磁盘的自由空间，一个是确定这个B树文件组织的数据文件的缓冲区大小。

总结： 每个缓冲区页面有一个缓冲区控制块（_bkt{}），其后紧跟着缓冲区页面（void* page)，每个缓冲区页面有一个页头（PAGE{}），页头之后又紧紧跟着页面数据信息，也就是记录的存放空间（由页头中的slinp[]指向）。记录的具体格式由存取方法和文件组织决定，比如B树的记录格式就与Hash记录不同，另外B树本身也有多种记录格式。

