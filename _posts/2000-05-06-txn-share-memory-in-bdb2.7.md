---
layout: post
title: BDB事务共享区域
---

最后我们来看一下事务共享区域。

首先仍然是每个进程都有的一个数据结构__db_txnmgr{}：

    /*
    * The transaction manager encapsulates the transaction system.  It contains
    * references to the log and lock managers as well as the state that keeps
    * track of the shared memory region.
    */
    struct __db_txnmgr {
    /* These fields need to be protected for multi-threaded support. */
        db_mutex_t    *mutexp;    /* Synchronization. */
                        /* list of active transactions */
        TAILQ_HEAD(_chain, __db_txn)    txn_chain;

    /* These fields are not protected. */
        REGINFO        reginfo;    /* Region information. */
        DB_ENV        *dbenv;        /* Environment. */
        int (*recover)            /* Recovery dispatch routine */
            __P((DB_LOG *, DBT *, DB_LSN *, int, void *));
        u_int32_t     flags;        /* DB_TXN_NOSYNC, DB_THREAD */
        DB_TXNREGION    *region;    /* address of shared memory region */
        void        *mem;        /* address of the shalloc space */
    };

同样拥有一个指向描述共享区域的reginfo字段和指向共享区域的指针mem。

    /*
    * Layout of the shared memory region.
    * The region consists of a DB_TXNREGION structure followed by a large
    * pool of shalloc'd memory which is used to hold TXN_DETAIL structures
    * and thread mutexes (which are dynamically allocated).
    */
    struct __db_txnregion {
        RLAYOUT        hdr;        /* Shared memory region header. */
        u_int32_t    magic;        /* transaction magic number */
        u_int32_t    version;    /* version number */
        u_int32_t    maxtxns;    /* maximum number of active txns */
        u_int32_t    last_txnid;    /* last transaction id given out */
        DB_LSN        pending_ckp;    /* last checkpoint did not finish */
        DB_LSN        last_ckp;    /* lsn of the last checkpoint */
        time_t        time_ckp;    /* time of last checkpoint */
        u_int32_t    logtype;    /* type of logging */
        u_int32_t    locktype;    /* lock type */
        u_int32_t    naborts;    /* number of aborted transactions */
        u_int32_t    ncommits;    /* number of committed transactions */
        u_int32_t    nbegins;    /* number of begun transactions */
        SH_TAILQ_HEAD(_active) active_txn;    /* active transaction list */
    };

共享区域除了有个公用的头部hdr字段以外，最重要的是有一个所有活动事务的列表active_txn字段。

现在让我们看一下这些数据结构和共享区域的创建或打开以及初始化操作。

在db_appinit()函数中调用txn_open()函数打开或连接一个事务共享区域：

    if (LF_ISSET(DB_INIT_TXN) && (ret = txn_open(NULL,
        LF_ISSET(DB_CREATE | DB_THREAD | DB_TXN_NOSYNC),
        mode, dbenv, &dbenv->tx_info)) != 0)
        goto err;

其中txn_open定义如下：

    int
    txn_open(path, flags, mode, dbenv, mgrpp)
        const char *path;
        u_int32_t flags;
        int mode;
        DB_ENV *dbenv;
        DB_TXNMGR **mgrpp;
    {
        DB_TXNMGR *tmgrp;
        u_int32_t maxtxns;
        int ret;

        /* Validate arguments. */
        首先仍然是验证参数

        然后用malloc创建每个进程都有的事务管理器结构
        /* Now, create the transaction manager structure and set its fields. */
        if ((ret = __os_calloc(1, sizeof(DB_TXNMGR), &tmgrp)) != 0)
            return (ret);
        并初始化事务管理器结构
        /* Initialize the transaction manager structure. */
        tmgrp->mutexp = NULL;
        tmgrp->dbenv = dbenv;
        tmgrp->recover =
            dbenv->tx_recover == NULL ? __db_dispatch : dbenv->tx_recover;
        tmgrp->flags = LF_ISSET(DB_TXN_NOSYNC | DB_THREAD);
        TAILQ_INIT(&tmgrp->txn_chain);

        现在开始创建或连接到一个事务共享区域
        /* Join/create the txn region. */
        首先还是一样，填充区域描述信息REGINFO{}，然后调用__db_rattach()函数创建或打开一个共享区域
        tmgrp->reginfo.dbenv = dbenv;
        tmgrp->reginfo.appname = DB_APP_NONE;
        if (path == NULL)
            tmgrp->reginfo.path = NULL;
        else
            if ((ret = __os_strdup(path, &tmgrp->reginfo.path)) != 0)
                goto err;
        tmgrp->reginfo.file = DEFAULT_TXN_FILE;
        tmgrp->reginfo.mode = mode;
        tmgrp->reginfo.size = TXN_REGION_SIZE(maxtxns);
        tmgrp->reginfo.dbflags = flags;
        tmgrp->reginfo.addr = NULL;
        tmgrp->reginfo.fd = -1;
        tmgrp->reginfo.flags = dbenv->tx_max == 0 ? REGION_SIZEDEF : 0;
        if ((ret = __db_rattach(&tmgrp->reginfo)) != 0)
            goto err;

        /* Fill in region-related fields. */
        tmgrp->region = tmgrp->reginfo.addr;
        tmgrp->mem = &tmgrp->region[1];

        if (F_ISSET(&tmgrp->reginfo, REGION_CREATED)) {
            tmgrp->region->maxtxns = maxtxns;
            if ((ret = __txn_init(tmgrp->region)) != 0)
                goto err;

        } else if (tmgrp->region->magic != DB_TXNMAGIC) {
            /* Check if valid region. */
            __db_err(dbenv, "txn_open: Bad magic number");
            ret = EINVAL;
            goto err;
        }

        // 如果这是一个新的事务共享区域，初始化之。这里又和MPOOL一样，
        // 使用__shmalloc函数管理共享区域内存
        if (LF_ISSET(DB_THREAD)) {
            if ((ret = __db_shalloc(tmgrp->mem, sizeof(db_mutex_t),
                MUTEX_ALIGNMENT, &tmgrp->mutexp)) == 0)
                /*
                * Since we only get here if threading is turned on, we
                * know that we have spinlocks, so the offset is going
                * to be ignored.  We put 0 here as a valid placeholder.
                */
                __db_mutex_init(tmgrp->mutexp, 0);
            if (ret != 0)
                goto err;
        }

        UNLOCK_TXNREGION(tmgrp);
        *mgrpp = tmgrp;
        return (0);

    err:    if (tmgrp->reginfo.addr != NULL) {
            if (tmgrp->mutexp != NULL)
                __db_shalloc_free(tmgrp->mem, tmgrp->mutexp);

            UNLOCK_TXNREGION(tmgrp);
            (void)__db_rdetach(&tmgrp->reginfo);
            if (F_ISSET(&tmgrp->reginfo, REGION_CREATED))
                (void)txn_unlink(path, 1, dbenv);
        }

        if (tmgrp->reginfo.path != NULL)
            __os_freestr(tmgrp->reginfo.path);
        __os_free(tmgrp, sizeof(*tmgrp));
        return (ret);
    }

这个函数就是txn_open()调用来初始化事务共享区域的函数：

    /*
    * This file contains the top level routines of the transaction library.
    * It assumes that a lock manager and log manager that conform to the db_log(3)
    * and db_lock(3) interfaces exist.
    *
    * Initialize a transaction region in shared memory.
    * Return 0 on success, errno on failure.
    */
    static int
    __txn_init(txn_region)
        DB_TXNREGION *txn_region;
    {
        time_t now;

        (void)time(&now);

        /* maxtxns is already initialized. */
        txn_region->magic = DB_TXNMAGIC;
        txn_region->version = DB_TXNVERSION;
        txn_region->last_txnid = TXN_MINIMUM;
        /*
        * XXX
        * If we ever do more types of locking and logging, this changes.
        */
        txn_region->logtype = 0;
        txn_region->locktype = 0;
        txn_region->time_ckp = now;
        ZERO_LSN(txn_region->last_ckp);
        ZERO_LSN(txn_region->pending_ckp);
        SH_TAILQ_INIT(&txn_region->active_txn);
        __db_shalloc_init((void *)&txn_region[1],
            TXN_REGION_SIZE(txn_region->maxtxns) - sizeof(DB_TXNREGION));

        return (0);
    }

其中DB_TXN{}是每个事务的描述符（定义在db_int.h文件中）：

    /* The structure allocated for every transaction. */
    struct __db_txn {
        DB_TXNMGR    *mgrp;        /* Pointer to transaction manager. */
        DB_TXN        *parent;    /* Pointer to transaction's parent. */
        DB_LSN        last_lsn;    /* Lsn of last log write. */
        u_int32_t    txnid;        /* Unique transaction id. */
        size_t        off;        /* Detail structure within region. */
        TAILQ_ENTRY(__db_txn) links;    /* Links transactions off manager. */
        TAILQ_HEAD(__kids, __db_txn) kids; /* Child transactions. */
        TAILQ_ENTRY(__db_txn) klinks;    /* Links child transactions. */

    #define    TXN_MALLOC    0x01        /* Structure allocated by TXN system. */
        u_int32_t    flags;
    };


    typedef struct __txn_detail {
        u_int32_t txnid;        /* current transaction id
                        used to link free list also */
        DB_LSN    last_lsn;        /* last lsn written for this txn */
        DB_LSN    begin_lsn;        /* lsn of begin record */
        size_t    last_lock;        /* offset in lock region of last lock
                        for this transaction. */
        size_t    parent;            /* Offset of transaction's parent. */
    #define    TXN_UNALLOC    0
    #define    TXN_RUNNING    1
    #define    TXN_ABORTED    2
    #define    TXN_PREPARED    3
    #define    TXN_COMMITTED    4
        u_int32_t status;        /* status of the transaction */
        SH_TAILQ_ENTRY    links;        /* free/active list */

    #define    TXN_XA_ABORTED        1
    #define    TXN_XA_DEADLOCKED    2
    #define    TXN_XA_ENDED        3
    #define    TXN_XA_PREPARED        4
    #define    TXN_XA_STARTED        5
    #define    TXN_XA_SUSPENDED    6
        u_int32_t xa_status;        /* XA status */

        /*
        * XID (xid_t) structure: because these fields are logged, the
        * sizes have to be explicit.
        */
        DB_XID xid;            /* XA global transaction id */
        u_int32_t bqual;        /* bqual_length from XID */
        u_int32_t gtrid;        /* gtrid_length from XID */
        int32_t format;            /* XA format */
    } TXN_DETAIL;

