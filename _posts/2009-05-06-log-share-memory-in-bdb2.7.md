---
layout: post
title: BDB日志共享区域
---

这一次我们对日志共享区域进行分析。

每个进程都有一个DB_LOG{}数据结构：

    /*
    * DB_LOG
    *    Per-process log structure.
    */
    struct __db_log {
    /* These fields need to be protected for multi-threaded support. */
        db_mutex_t    *mutexp;    /* Mutex for thread protection. */

        DB_ENTRY *dbentry;        /* Recovery file-id mapping. */
    #define    DB_GROW_SIZE    64
        u_int32_t dbentry_cnt;        /* Entries.  Grows by DB_GROW_SIZE. */

    /*
    * These fields are always accessed while the region lock is held, so they do
    * not have to be protected by the thread lock as well OR, they are only used
    * when threads are not being used, i.e. most cursor operations are disallowed
    * on threaded logs.
    */
        u_int32_t lfname;        /* Log file "name". */
        int      lfd;            /* Log file descriptor. */

        DB_LSN      c_lsn;        /* Cursor: current LSN. */
        DBT      c_dbt;        /* Cursor: return DBT structure. */
        int      c_fd;            /* Cursor: file descriptor. */
        u_int32_t c_off;        /* Cursor: previous record offset. */
        u_int32_t c_len;        /* Cursor: current record length. */

    /* These fields are not protected. */
        LOG     *lp;            /* Address of the shared LOG. */

        DB_ENV     *dbenv;        /* Reference to error information. */
        REGINFO      reginfo;        /* Region information. */

        void     *addr;            /* Address of shalloc() region. */

        char     *dir;            /* Directory argument. */

    /*
    * These fields are used by XA; since XA forbids threaded execution, these
    * do not have to be protected.
    */
        void     *xa_info;        /* Committed transaction list that
                        * has to be carried between calls
                        * to xa_recover. */
        DB_LSN    xa_lsn;            /* Position of an XA recovery scan. */
        DB_LSN    xa_first;        /* LSN to which we need to roll back
                        for this XA recovery scan. */

        /*
        * !!!
        * Currently used to hold:
        *    DB_AM_THREAD    (a DB flag)
        *    DBC_RECOVER    (a DBC flag)
        * If they are ever the same bits, we're in serious trouble.
        */
    #if DB_AM_THREAD == DBC_RECOVER
        DB_AM_THREAD, DBC_RECOVER, FLAG MISMATCH
    #endif
        u_int32_t flags;
    };

其中reginfo字段指向共享内存区域描述信息结构REGINFO{}。lp字段指向分配在共享缓冲区的，用于描述该缓冲区的LOG{}结构的起始地址。

该结构位于共享区域的开始。addr字段指向用于分配的共享内存的起始地址，位于:共享区域地址mp+sizeof(mp)。

    /*
    * LOG --
    *    Shared log region.  One of these is allocated in shared memory,
    *    and describes the log.
    */
    struct __log {
        RLAYOUT      rlayout;        /* General region information. */

        LOGP      persist;        /* Persistent information. */

        SH_TAILQ_HEAD(__fq) fq;        /* List of file names. */

        /*
        * The lsn LSN is the file offset that we're about to write and which
        * we will return to the user.
        */
        DB_LSN      lsn;            /* LSN at current file offset. */

        /*
        * The s_lsn LSN is the last LSN that we know is on disk, not just
        * written, but synced.
        */
        DB_LSN      s_lsn;        /* LSN of the last sync. */

        u_int32_t len;            /* Length of the last record. */

        u_int32_t w_off;        /* Current write offset in the file. */

        DB_LSN      chkpt_lsn;        /* LSN of the last checkpoint. */
        time_t      chkpt;        /* Time of the last checkpoint. */

        DB_LOG_STAT stat;        /* Log statistics. */

        /*
        * The f_lsn LSN is the LSN (returned to the user) that "owns" the
        * first byte of the buffer.  If the record associated with the LSN
        * spans buffers, it may not reflect the physical file location of
        * the first byte of the buffer.
        */
        DB_LSN      f_lsn;        /* LSN of first byte in the buffer. */
        size_t      b_off;        /* Current offset in the buffer. */
        u_int8_t buf[4 * 1024];        /* Log buffer. */
    };

其中rlayout字段是每个REGION的头部数据结构，是各个REGION的通用信息。定义在db_int.h文件中。
剩下的信息都是应用相关的，在MPOOL中就是描述用于内存缓冲区的共享区域信息。

与前面介绍的两个共享区域略为不同，LOG结构中有个LOGP      persist;        /* Persistent information. */字段，是用于保存日志文件的持久化信息的。

    struct __log_persist {
        u_int32_t magic;        /* DB_LOGMAGIC */
        u_int32_t version;        /* DB_LOGVERSION */

        u_int32_t lg_max;        /* Maximum file size. */
        int      mode;            /* Log file mode. */
    };

现在让我们来看看LOG区域是怎样创建或打开的。

首先在db_appinit()函数中调用log_open()函数打开或连接到一个LOG共享区域：
    if (LF_ISSET(DB_INIT_LOG) && (ret = log_open(NULL,
        LF_ISSET(DB_CREATE | DB_THREAD),
        mode, dbenv, &dbenv->lg_info)) != 0)

我们来看一下log_open()函数的定义：

    /*
    * log_open --
    *    Initialize and/or join a log.
    */
    int
    log_open(path, flags, mode, dbenv, lpp)
        const char *path;
        u_int32_t flags;
        int mode;
        DB_ENV *dbenv;
        DB_LOG **lpp;
    {
        DB_LOG *dblp;
        LOG *lp;
        int ret;

        /* Validate arguments. */
    #ifdef HAVE_SPINLOCKS
    #define    OKFLAGS    (DB_CREATE | DB_THREAD)
    #else
    #define    OKFLAGS    (DB_CREATE)
    #endif
        if ((ret = __db_fchk(dbenv, "log_open", flags, OKFLAGS)) != 0)
            return (ret);

        /* Create and initialize the DB_LOG structure. */
        if ((ret = __os_calloc(1, sizeof(DB_LOG), &dblp)) != 0)
            return (ret);

        if (path != NULL && (ret = __os_strdup(path, &dblp->dir)) != 0)
            goto err;

        dblp->dbenv = dbenv;
        dblp->lfd = -1;
        ZERO_LSN(dblp->c_lsn);
        dblp->c_fd = -1;

        /*
        * The log region isn't fixed size because we store the registered
        * file names there.  Make it fairly large so that we don't have to
        * grow it.
        */
    #define    DEF_LOG_SIZE    (30 * 1024)

        /* Map in the region. */
        dblp->reginfo.dbenv = dbenv;
        dblp->reginfo.appname = DB_APP_LOG;
        if (path == NULL)
            dblp->reginfo.path = NULL;
        else
            if ((ret = __os_strdup(path, &dblp->reginfo.path)) != 0)
                goto err;
        dblp->reginfo.file = DB_DEFAULT_LOG_FILE;
        dblp->reginfo.mode = mode;
        dblp->reginfo.size = DEF_LOG_SIZE;
        dblp->reginfo.dbflags = flags;
        dblp->reginfo.flags = REGION_SIZEDEF;
        if ((ret = __db_rattach(&dblp->reginfo)) != 0)
            goto err;

        /*
        * The LOG structure is first in the region, the rest of the region
        * is free space.
        */
        dblp->lp = dblp->reginfo.addr;
        dblp->addr = (u_int8_t *)dblp->lp + sizeof(LOG);

        /* Initialize a created region. */
        if (F_ISSET(&dblp->reginfo, REGION_CREATED)) {
            __db_shalloc_init(dblp->addr, DEF_LOG_SIZE - sizeof(LOG));

            /* Initialize the LOG structure. */
            lp = dblp->lp;
            lp->persist.lg_max = dbenv == NULL ? 0 : dbenv->lg_max;
            if (lp->persist.lg_max == 0)
                lp->persist.lg_max = DEFAULT_MAX;
            lp->persist.magic = DB_LOGMAGIC;
            lp->persist.version = DB_LOGVERSION;
            lp->persist.mode = mode;
            SH_TAILQ_INIT(&lp->fq);

            /* Initialize LOG LSNs. */
            lp->lsn.file = 1;
            lp->lsn.offset = 0;
        }

        /* Initialize thread information, mutex. */
        if (LF_ISSET(DB_THREAD)) {
            F_SET(dblp, DB_AM_THREAD);
            if ((ret = __db_shalloc(dblp->addr,
                sizeof(db_mutex_t), MUTEX_ALIGNMENT, &dblp->mutexp)) != 0)
                goto err;
            (void)__db_mutex_init(dblp->mutexp, 0);
        }

        /*
        * If doing recovery, try and recover any previous log files before
        * releasing the lock.
        */
        if (F_ISSET(&dblp->reginfo, REGION_CREATED) &&
            (ret = __log_recover(dblp)) != 0)
            goto err;

        UNLOCK_LOGREGION(dblp);
        *lpp = dblp;
        return (0);

    err:    if (dblp->reginfo.addr != NULL) {
            if (dblp->mutexp != NULL)
                __db_shalloc_free(dblp->addr, dblp->mutexp);

            UNLOCK_LOGREGION(dblp);
            (void)__db_rdetach(&dblp->reginfo);
            if (F_ISSET(&dblp->reginfo, REGION_CREATED))
                (void)log_unlink(path, 1, dbenv);
        }

        if (dblp->reginfo.path != NULL)
            __os_freestr(dblp->reginfo.path);
        if (dblp->dir != NULL)
            __os_freestr(dblp->dir);
        __os_free(dblp, sizeof(*dblp));
        return (ret);
    }

