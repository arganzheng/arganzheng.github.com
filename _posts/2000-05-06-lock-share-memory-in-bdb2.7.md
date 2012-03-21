---
layout: post
title: BDB锁共享区域
---

这一次我们以锁共享区域分析。

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

与MPOOL一样，有一个每进程都有的数据结构__db_locktab{}，其中reginfo字段指向共享内存区域描述信息结构REGINFO{}。

region字段指向分配在共享缓冲区的，用于描述该缓冲区的DB_LOCKREGION{}结构的起始地址。该结构位于共享区域的开始。

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

每一个共享区域都有一个存放通用信息的区域头RLAYOUT{}。
其他的信息都是针对共享区域起始地址的偏移量，这是因为共享区域在各个进程的地址空间映射地址是各不相同的（即addr各不相同），因此不能使用相对进程的内存地址。

我们来看一下锁区域是怎样创建的。

在db_appinit()函数中调用lock_open():

    if (LF_ISSET(DB_INIT_LOCK) && (ret = lock_open(NULL,
        LF_ISSET(DB_CREATE | DB_THREAD),
        mode, dbenv, &dbenv->lk_info)) != 0)

我们来看一下lock_open()函数：

    int
    lock_open(path, flags, mode, dbenv, ltp)
        const char *path;
        u_int32_t flags;
        int mode;
        DB_ENV *dbenv;
        DB_LOCKTAB **ltp;
    {
        DB_LOCKTAB *lt;
        u_int32_t lock_modes, maxlocks, regflags;
        int ret;

    /* 首先验证参数的合法性 */
        /* Validate arguments. */
    ......

        /* 然后使用malloc创建每个进程都有的锁表数据结构 */
        /* Create the lock table structure. */
        if ((ret = __os_calloc(1, sizeof(DB_LOCKTAB), &lt)) != 0)
            return (ret);
        lt->dbenv = dbenv;

        /* Grab the values that we need to compute the region size. */
        lock_modes = DB_LOCK_RW_N;
        maxlocks = DB_LOCK_DEFAULT_N;
        regflags = REGION_SIZEDEF;
        if (dbenv != NULL) {
            if (dbenv->lk_modes != 0) {
                lock_modes = dbenv->lk_modes;
                regflags = 0;
            }
            if (dbenv->lk_max != 0) {
                maxlocks = dbenv->lk_max;
                regflags = 0;
            }
        }

        /* 现在可以创建或链接到一个共享锁区域 */
        /* Join/create the lock region. */
        /* 填充reginfo信息以调用底层的__db_rattach()函数 */
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

        /* 调用__db_rattach()函数创建或打开一个共享区域 */
        if ((ret = __db_rattach(&lt->reginfo)) != 0)
            goto err;

        /* Now set up the pointer to the region. */
        lt->region = lt->reginfo.addr;

        /* 如果是该共享区域是新创建的，初始化之 */
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
        if (dbenv != NULL && dbenv->lk_detect != DB_LOCK_NORUN) {
            if (lt->region->detect != DB_LOCK_NORUN &&
                dbenv->lk_detect != DB_LOCK_DEFAULT &&
                lt->region->detect != dbenv->lk_detect) {
                __db_err(dbenv,
                "lock_open: incompatible deadlock detector mode");
                ret = EINVAL;
                goto err;
            }
            if (lt->region->detect == DB_LOCK_NORUN)
                lt->region->detect = dbenv->lk_detect;
        }
        
        /* 填充区域的指针字段：注意我们前面说过的映射问题。这里进程利用公共DB_LOCKREGION类型字段region的相对地址在自己的锁表中填写绝对地址 */
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

    /*
    * __lock_panic --
    *    Panic a lock region.
    *
    * PUBLIC: void __lock_panic __P((DB_ENV *));
    */
    void
    __lock_panic(dbenv)
        DB_ENV *dbenv;
    {
        if (dbenv->lk_info != NULL)
            dbenv->lk_info->region->hdr.panic = 1;
    }

当第一个创建共享区域时，使用__lock_tabinit初始化锁区域（定义在lock_region.c文件中）
分为两部分：初始化描述共享区域的DB_LOCKREGINFO{}结构和共享区域本身。事实上，正如MPOOL共享区域，
这两部分都是进程共享的对象，因此都在共享区域中，而且在实现上是紧挨在一起的：curaddr = (u_int8_t *)lrp + sizeof(DB_LOCKREGION);一个值得注意的地方就是在共享区域中的指针必须以相对起始共享区域起始位置的偏移量存储，

原因我们在前面已经说明了：lrp->hash_off = curaddr - (u_int8_t *)lrp;另一个需要注意的地方是内存对齐问题。

    /*
    * __lock_tabinit --
    *    Initialize the lock region.
    */
    static int
    __lock_tabinit(dbenv, lrp)
        DB_ENV *dbenv;
        DB_LOCKREGION *lrp;
    {
        struct __db_lock *lp;
        struct lock_header *tq_head;
        struct obj_header *obj_head;
        DB_LOCKOBJ *op;
        u_int32_t i, nelements;
        const u_int8_t *conflicts;
        u_int8_t *curaddr;

        conflicts = dbenv == NULL || dbenv->lk_conflicts == NULL ?
            db_rw_conflicts : dbenv->lk_conflicts;

        lrp->table_size = __db_tablesize(lrp->maxlocks);
        lrp->magic = DB_LOCKMAGIC;
        lrp->version = DB_LOCKVERSION;
        lrp->id = 0;
        /*
        * These fields (lrp->maxlocks, lrp->nmodes) are initialized
        * in the caller, since we had to grab those values to size
        * the region.
        */
        lrp->need_dd = 0;
        lrp->detect = DB_LOCK_NORUN;
        lrp->numobjs = lrp->maxlocks;
        lrp->nlockers = 0;
        lrp->mem_bytes = ALIGN(STRING_SIZE(lrp->maxlocks), sizeof(size_t));
        lrp->increment = lrp->hdr.size / 2;
        lrp->nconflicts = 0;
        lrp->nrequests = 0;
        lrp->nreleases = 0;
        lrp->ndeadlocks = 0;

        /*
        * As we write the region, we've got to maintain the alignment
        * for the structures that follow each chunk.  This information
        * ends up being encapsulated both in here as well as in the
        * lock.h file for the XXX_SIZE macros.
        */
        /* Initialize conflict matrix. */
        curaddr = (u_int8_t *)lrp + sizeof(DB_LOCKREGION);
        memcpy(curaddr, conflicts, lrp->nmodes * lrp->nmodes);
        curaddr += lrp->nmodes * lrp->nmodes;

        /*
        * Initialize hash table.
        */
        curaddr = (u_int8_t *)ALIGNP(curaddr, LOCK_HASH_ALIGN);
        lrp->hash_off = curaddr - (u_int8_t *)lrp;
        nelements = lrp->table_size;
        __db_hashinit(curaddr, nelements);
        curaddr += nelements * sizeof(DB_HASHTAB);

        /*
        * Initialize locks onto a free list. Since locks contains mutexes,
        * we need to make sure that each lock is aligned on a MUTEX_ALIGNMENT
        * boundary.
        */
        curaddr = (u_int8_t *)ALIGNP(curaddr, MUTEX_ALIGNMENT);
        tq_head = &lrp->free_locks;
        SH_TAILQ_INIT(tq_head);

        for (i = 0; i++ < lrp->maxlocks;
            curaddr += ALIGN(sizeof(struct __db_lock), MUTEX_ALIGNMENT)) {
            lp = (struct __db_lock *)curaddr;
            lp->status = DB_LSTAT_FREE;
            SH_TAILQ_INSERT_HEAD(tq_head, lp, links, __db_lock);
        }

        /* Initialize objects onto a free list.  */
        obj_head = &lrp->free_objs;
        SH_TAILQ_INIT(obj_head);

        for (i = 0; i++ < lrp->maxlocks; curaddr += sizeof(DB_LOCKOBJ)) {
            op = (DB_LOCKOBJ *)curaddr;
            SH_TAILQ_INSERT_HEAD(obj_head, op, links, __db_lockobj);
        }

        /*
        * Initialize the string space; as for all shared memory allocation
        * regions, this requires size_t alignment, since we store the
        * lengths of malloc'd areas in the area.
        */
        curaddr = (u_int8_t *)ALIGNP(curaddr, sizeof(size_t));
        lrp->mem_off = curaddr - (u_int8_t *)lrp;
        __db_shalloc_init(curaddr, lrp->mem_bytes);
        return (0);
    }

