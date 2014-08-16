---
title: select、poll和epoll简介
layout: post
---

说明：本篇文章是网上各种资料杂烩而成。如有雷同，纯属自然。

select
------

### select功能介绍

select系统调用的功能是对多个文件描述符进行监视，当有文件描述符的文件读写操作完成，发生异常或者超时，该调用会返回这些文件描述符。

    int select(int nfds, fd_set *readfds, fd_set *writefds,fd_set *exceptfds,struct timeval *timeout);

### select关键结构体

这里先介绍一些关键的结构体：

    typedef struct {
        unsigned long *in, *out, *ex;
        unsigned long *res_in, *res_out, *res_ex;
    } fd_set_bits;

这个结构体负责保存select在用户态的参数。在select()中，每一个文件描述符用一个位表示，其中1表示这个文件是被监视的。in, out, ex指向的bit数组表示对应的读，写，异常文件的描述符。res_in, res_out,res_ex指向的bit数组表示对应的读，写，异常文件的描述符的检测结果。

    struct poll_wqueues {
        poll_table pt;
        struct poll_table_page * table;
        int error;
    };

这是最主要的结构体，它保存了select过程中的重要信息。它包括了两个最重要的结构体poll_table和struct poll_table_page。接下去看看这两个结构体。

    typedef void (*poll_queue_proc)(struct file *, wait_queue_head_t *, struct
poll_table_struct *);

    struct poll_table_struct {
        poll_queue_proc qproc;
    } poll_table;

在执行select操作时，会用到回调函数，poll_table就是用来保存回调函数的。这个回调函数非常重要，因为当文件执行poll操作时，一般都会调用这个回调函数。所以，这个回调函数非常重要，通常负责把进程放入等待队列等关键操作。下面可以看到，在select中，这个回调函数是__pollwait()，在epoll中，这个回调函数是 ep_ptable_queue_proc。

    struct poll_table_page {
        struct poll_table_page * next;
        struct poll_table_entry * entry;
        struct poll_table_entry entries[0];
    };

这个表记录了在select过程中生成的所有等待队列的结点。由于select要监视多个文件描述符，并且要把当前进程放入这些描述符的等待队列中，因此要分配等待队列的节点。这些节点可能如此之多，以至于不可能像通常做的那样，在堆栈中分配它们。所以，select以动态分配的方式把它保存在poll_table_page中。保存的方式是单向链表，每个节点以页为单位，分配多个poll_table_entry项。
    
现在看一下poll_table_entry: poll table的表项

    struct poll_table_entry {
        struct file * filp;
        wait_queue_t wait;
        wait_queue_head_t * wait_address;
    };

其中filp是select要监视的structfile结构体，wait_address是文件操作的等待
队列的队首，wait是等待队列的节点。

### select的实现 

在进行一系列参数检查后，sys_select调用do_select()。该函数会遍历所有需要监视的文件描述符，然后调用f_op->poll()，这个操作会做两件事：
     
1.查看文件操作的状态，如果这些文件操作完成或者有异常发生（下面统称为“用户感兴趣的事件”），在对应的fdset中标记这些文件描述符。
2.如果retval为0（在这一轮遍历中，迄今为止，文件没有发生感兴趣的事，这一点有些不明白，为什么不是通知所有监视的并且没有发生感兴趣事件的文件描述符，这样返回得更快）并且没有超时，那么，通知这些文件，让他们在文件操作完成时唤醒本进程。

如果发现文件具体通知的方式是：通过__pollwait()把自己挂到各个等待队列中。这样，当有文件操作完成时，select所在进程会被唤醒。这里涉及到一个回调函数__pollwait()。它是在什么时候被注册的呢？在进入for循环之前，有这样一行代码：

    poll_initwait(&table);

它的作用就是把poll_table中的回调函数设置为__pollwait。对文件描述符的遍历的循环会继续，直到发生以下事件：如果有文件操作完成或者发生异常，或者超时，或者收到信号，select会返回相应的值，否则，do_select会调用schedule_timeout()进入休眠，直到超时或者被再次唤醒（这表明有用户感兴趣的事件产生），然后重新执行for循环，但是这一次一定能跳出循环体。
   
通知的过程如下（以管道的poll函数为例，在pipe中，f_op->poll对应的函数是pipe_poll：）：当do_select遍历所有需要监视的文件描述符时，假设有一个文件描述符对应的是一个管道，那么，它执行的f_op->poll实际上是pipe_poll。pipe_poll->poll_wait->__pollwait。最终__pollwait会把当前进程挂到对应文件的inode中的文件描述符中。当执行pipe_write对管道进行写操作时，操作完成后会唤醒等待队列中所有的进程。

### select的性能分析      

从中可以看出，select需要遍历所有的文件描述符，就遍历操作而言，复杂度是O(N)，N是最大文件描述符加1。此外，select参数包括了所有的文件描述符的信息，所以select在遍历文件描述符时，需要检查文件描述符是不是自己感兴趣的。

poll
----

### poll的功能介绍      

poll与select实现了相同的功能，只是参数类型不同。它的原型是：

    int poll(struct pollfd *fds, nfds_t nfds, int timeout);

可以看到，poll的参数中，直接列出了要监视的文件描述符的信息，而不像select一样要列出从0开始到nfds-1的所有文件描述符。这样的好处是，poll不需要查询很多无关的文件描述符的信息，在一定场合下效率会有所提高

### poll的关键结构体 

poll用到的很多结构体与select是一样的，如struct poll_wqueues。这是因为poll的实现机制与select没有本质区别。poll也用到了一些不同的结构体，这是因为poll的参数类型与select不同，用来保存参数的结构体也不同：

相关的数据结构：

    struct poll_list {
        struct poll_list *next;
        int len;
        struct pollfd entries[0];
    };
      
这个结构体用来保存被监视的文件描述符的参数。其中struct pollfd entries[0]表示这是一个不定长数组。

    struct pollfd {
        int fd;
        short events;
        short revents;
    };

这个结构体记录被监视的文件描述符和它的状态。

### poll的实现    

poll的实现与select也十分类似。一个区别是使用的数据结构。poll中采用了poll_list结构体来记录要监视的文件描述符信息。poll_list中，每个pollfd代表一个要监视的文件描述符的信息。这些pollfd以数组的形式链接到poll_list链表中，每个数组的元素个数不能超过POLLFD_PER_PAGE。在把参数拷贝到内核态之后，sys_poll会调用do_poll()。在do_poll()中，函数遍历poll_list链表，然后调用do_pollfd()对每个poll_list节点中的pollfd数组进行遍历。在do_pollfd()中，检查数组中的每个fd，检查的过程与select类似，调用fd对应的poll函数指针：

    mask = file->f_op->poll(file, *pwait);

1.如果有必要，把当前进程挂到文件操作对应的等待列队中，同时也放到polltable中。
2.检查文件操作状态，保存到mask变量中。

在遍历了所有文件描述符后，调用timeout = schedule_timeout(timeout);让当前进程进入休眠状态，直到超时或者有文件操作完成，唤醒当前进程才返回。那么，在 f_op->poll中做了些什么呢？在sys_poll中有这样一行代码：

    poll_initwait(&table);

可以发现，在这里，它注册了和select()相同的回调函数__pollwait(),内部的实现机制也是一样的。在这里就不重复说了。

### poll的性能分析

poll的参数只包括了用户感兴趣的文件信息，所以poll在遍历文件描述符时不用像select一样检查文件描述符是否是自己感兴趣的。从这个意义上说，poll比select稍微要高效一些。前提是：要监视的文件描述符不连续，非常离散。

poll与select共同的问题是，他们都是遍历所有的文件描述符。当要监视的文件描述符很多，并且每次只返回很少的文件描述符时，select/poll每次都要反复地从用户态拷贝文件信息，每次都要重新遍历文件描述符，而且每次都要把当前进程挂到对应事件的等待队列和poll_table的等待队列中。这里事实上做了很多重复劳动。

epoll
-----

### epoll的功能介绍 

epoll与select/poll不同的一点是，它是由一组系统调用组成

* int epoll_create(int size);
* int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);
* int epoll_wait(int epfd, struct epoll_event *events,
int maxevents, int timeout);

epoll相关系统调用是在Linux 2.5.44开始引入的。该系统调用针对传统的select/poll系统调用的不足，设计上作了很大的改动。select/poll的缺点在于：

1. 每次调用时要重复地从用户态读入参数。
2. 每次调用时要重复地扫描文件描述符。
3. 每次在调用开始时，要把当前进程放入各个文件描述符的等待队列。在调用结束后，又把进程从各个等待队列中删除。

在实际应用中，select/poll监视的文件描述符可能会非常多，如果每次只是返回一小部分，那么，这种情况下select/poll显得不够高效。epoll的设计思路，是把select/poll单个的操作拆分为1个epoll_create + 多个epoll_ctrl + 一个wait。此外，内核针对epoll操作添加了一个文件系统“eventpollfs”，每一个或者多个要监视的文件描述符都有一个对应的eventpollfs文件系统的inode节点，主要信息保存在eventpoll结构体中。而被监视的文件的重要信息则保存在epitem结构体中。所以他们是一对多的关系。由于在执行epoll_create和epoll_ctrl时，已经把用户态的信息保存到内核态了，所以之后即使反复地调用epoll_wait，也不会重复地拷贝参数，扫描文件描述符，反复地把当前进程放入/放出等待队列。这样就避免了以上的三个缺点。接下去看看它们的实现。

### epoll的关键结构体

    /* Wrapper struct used by poll queueing */
    struct ep_pqueue {
        poll_table pt;
        struct epitem *epi;
    };

这个结构体类似于select/poll中的struct poll_wqueues。由于epoll需要在内核
态保存大量信息，所以光光一个回调函数指针已经不能满足要求，所以在这里引入了一个新的结构体struct epitem。

    /*
    * Each file descriptor added to the eventpoll interface will
    * have an entry of this type linked to the hash.
    */
    struct epitem {
        /* RB-Tree node used to link this structure to the eventpoll rb-tree */
        struct rb_node rbn;
        红黑树，用来保存eventpoll

        /* List header used to link this structure to the eventpoll ready list
        */
        struct list_head rdllink;
        双向链表，用来保存已经完成的eventpoll

        /* The file descriptor information this item refers to */
        struct epoll_filefd ffd;
        这个结构体对应的被监听的文件描述符信息

        /* Number of active wait queue attached to poll operations */
        int nwait;
        poll操作中事件的个数

        /* List containing poll wait queues */
        struct list_head pwqlist;
        双向链表，保存着被监视文件的等待队列，功能类似于select/poll中的poll_table
        
        /* The "container" of this item */
        struct eventpoll *ep;
        指向eventpoll，多个epitem对应一个eventpoll

        /* The structure that describe the interested events and the source fd
        */
        struct epoll_event event;
        记录发生的事件和对应的fd

        /*
        * Used to keep track of the usage count of the structure. This avoids
        * that the structure will desappear from underneath our processing.
        */
        atomic_t usecnt;
        引用计数

        /* List header used to link this item to the "struct file" itemslist */
        struct list_head fllink;
        双向链表，用来链接被监视的文件描述符对应的struct file。因为file里有f_ep_link，
        用来保存所有监视这个文件的epoll节点

        /* List header used to link the item to the transfer list */
        struct list_head txlink;
        双向链表，用来保存传输队列

        /*
        * This is used during the collection/transfer of events to userspace
        * to pin items empty events set.
        */
        unsigned int revents;
        文件描述符的状态，在收集和传输时用来锁住空的事件集合
    };

该结构体用来保存与epoll节点关联的多个文件描述符，保存的方式是使用红黑树实现的hash表。至于为什么要保存，下文有详细解释。它与被监听的文件描述符一一对应。

    struct eventpoll {
        /* Protect the this structure access */
        rwlock_t lock;
        读写锁
        
        /*
        * This semaphore is used to ensure that files are not removed
        * while epoll is using them. This is read-held during the event
        * collection loop and it is write-held during the file cleanup
        * path, the epoll file exit code and the ctl operations.
        */
        struct rw_semaphore sem;
        读写信号量
        
        /* Wait queue used by sys_epoll_wait() */
        wait_queue_head_t wq;
        
        /* Wait queue used by file->poll() */
        wait_queue_head_t poll_wait;
        
        /* List of ready file descriptors */
        struct list_head rdllist;
        已经完成的操作事件的队列。
        
        /* RB-Tree root used to store monitored fd structs */
        struct rb_root rbr;
        保存epoll监视的文件描述符
    };
       
这个结构体保存了epoll文件描述符的扩展信息，它被保存在file结构体的private_data中。它与epoll文件节点一一对应。通常一个epoll文件节点对应多个被监视的文件描述符。所以一个eventpoll结构体会对应多个epitem结构体。

那么，epoll中的等待事件放在哪里呢？见下面

    /* Wait structure used by the poll hooks */
    struct eppoll_entry {
        /* List header used to link this structure to the "structepitem" */
        struct list_head llink;

        /* The "base" pointer is set to the container "structepitem" */
        void *base;
        /*
        * Wait queue item that will be linked to the target file wait
        * queue head.
        */
        wait_queue_t wait;

        /* The wait queue head that linked the "wait" wait queue item */
        wait_queue_head_t *whead;
    };

与select/poll的struct poll_table_entry相比，epoll的表示等待队列节点的结构体只是稍有不同，与structpoll_table_entry比较一下。

    struct poll_table_entry {
        struct file * filp;
        wait_queue_t wait;
        wait_queue_head_t * wait_address;
    };

由于epitem对应一个被监视的文件，所以通过base可以方便地得到被监视的文件信息。又因为一个文件可能有多个事件发生，所以用llink链接这些事件。

### epoll的两种工作模式

令人高兴的是，2.6内核的epoll比其2.5开发版本的/dev/epoll简洁了许多，所以，大部分情况下，强大的东西往往是简单的。唯一有点麻烦是epoll有2种工作方式:LT和ET。

LT(level triggered)是缺省的工作方式，并且同时支持block和no-block socket.在这种做法中，内核告诉你一个文件描述符是否就绪了，然后你可以对这个就绪的fd进行IO操作。如果你不作任何操作，内核还是会继续通知你的，所以，这种模式编程出错误可能性要小一点。传统的select/poll都是这种模型的代表．

ET (edge-triggered)是高速工作方式，只支持no-block socket。在这种模式下，当描述符从未就绪变为就绪时，内核通过epoll告诉你。然后它会假设你知道文件描述符已经就绪，并且不会再为那个文件描述符发送更多的就绪通知，直到你做了某些操作导致那个文件描述符不再为就绪状态了(比如，你在发送，接收或者接收请求，或者发送接收的数据少于一定量时导致了一个EWOULDBLOCK 错误）。但是请注意，如果一直不对这个fd作IO操作(从而导致它再次变成未就绪)，内核不会发送更多的通知(only once),不过在TCP协议中，ET模式的加速效用仍需要更多的benchmark确认。

### epoll的实现

#### 1. epoll_create 的实现

epoll_create()的功能是创建一个eventpollfs文件系统的inode节点。具体由ep_getfd()完成。ep_getfd()先调用ep_eventpoll_inode()创建一个inode节点，然后调用d_alloc()为inode分配一个dentry。最后把file,dentry,inode三者关联起来。在执行了ep_getfd()之后，它又调用了ep_file_init(),分配了eventpoll结构体，并把eventpoll的指针赋给file结构体，这样eventpoll就与file结构体关联起来了。需要注意的是epoll_create()的参数size实际上只是起参考作用，只要它不小于等于0，就并不限制这个epoll inode关联的文件描述符数量。

#### 2. epoll_ctl 的实现 

epoll_ctl的功能是实现一系列操作，如把文件与eventpollfs文件系统的inode节点关联起来。这里要介绍一下eventpoll结构体，它保存在file->f_private中，记录了eventpollfs文件系统的inode节点的重要信息，其中成员rbr保存了该epoll文件节点监视的所有文件描述符。组织的方式是一棵红黑树，这种结构体在查找节点时非常高效。首先它调用ep_find()从eventpoll中的红黑树获得epitem结构体。然后根据op参数的不同而选择不同的操作。如果op为EPOLL_CTL_ADD，那么正常情况下epitem是不可能在eventpoll的红黑树中找到的，所以调用ep_insert创建一个epitem结构体并插入到对应的红黑树中。

ep_insert()首先分配一个epitem对象，对它初始化后，把它放入对应的红黑树。此外，这个函数还要作一个操作，就是把当前进程放入对应文件操作的等待队列。这一步是由下面的代码完成的。

    init_poll_funcptr(&epq.pt, ep_ptable_queue_proc);
    ...
    revents = tfile->f_op->poll(tfile, &epq.pt);

函数先调用init_poll_funcptr注册了一个回调函数ep_ptable_queue_proc，这个函数会在调用f_op->poll时被执行。该函数分配一个epoll等待队列结点eppoll_entry：一方面把它挂到文件操作的等待队列中，另一方面把它挂到epitem的队列中。此外，它还注册了一个等待队列的回调函数ep_poll_callback。当文件操作完成，唤醒当前进程之前，会调用ep_poll_callback()，把eventpoll放到epitem的完成队列中（注释：通过查看代码，此处应该是把epitem放到eventpoll的完成队列，只有这样才能在epoll_wait()中只要看eventpoll的完成队列即可得到所有的完成文件描述符），并唤醒等待进程。如果在执行f_op->poll以后，发现被监视的文件操作已经完成了，那么把它放在完成队列中了，并立即把等待操作的那些进程唤醒。

#### 3. epoll_wait 的实现       

epoll_wait的工作是等待文件操作完成并返回。它的主体是ep_poll()，该函数在for循环中（注释：这里没有循环。）检查epitem（注释：这里应该是eventpoll）中有没有已经完成的事件，有的话就把结果返回。没有的话调用schedule_timeout()进入休眠，直到进程被再度唤醒或者超时。

### epoll的性能分析

epoll机制是针对select/poll的缺陷设计的。通过新引入的eventpollfs文件系统，epoll把参数拷贝到内核态，在每次轮询时不会重复拷贝。通过把操作拆分为epoll_create,epoll_ctl,epoll_wait，避免了重复地遍历要监视的文件描述符。此外，由于调用epoll的进程被唤醒后，只要直接从epitem的完成队列中找出完成的事件，找出完成事件的复杂度由O(N)降到了O(1)。但是epoll的性能提高是有前提的，那就是监视的文件描述符非常多，而且每次完成操作的文件非常少。所以，epoll能否显著提高效率，取决于实际的应用场景。这方面需要进一步测试。


select、poll与epoll一些比较
---------------------------

### 1. 支持一个进程所能打开的最大连接数

**select**

单个进程所能打开的最大连接数有FD_SETSIZE宏定义，其大小是32个整数的大小（在32位的机器上，大小就是32*32，同理64位机器上 FD_SETSIZE为32*64），当然我们可以对它进行修改，然后重新编译内核，但是性能可能会受到影响，这需要进一步的测试。

**poll**

poll本质上和select没有区别，但是它没有最大连接数的限制，原因是它是基于链表来存储的

**epoll**

虽然连接数有上限，但是很大，1G内存的机器上可以打开10万左右的连接，2G内存的机器可以打开20万左右的连接。

### 2. FD剧增后带来的IO效率问题 

**select**

因为每次调用时都会对连接进行线性遍历，所以随着FD的增加会造成遍历速度慢的“线性下降性能问题”。

**poll**

同上

**epoll**

因为epoll内核中实现是根据每个fd上的callback函数来实现的，只有活跃的socket才会主动调用callback，所以在活跃socket较少的情况下，使用epoll没有前面两者的线性下降的性能问题，但是所有socket都很活跃的情况下，可能会有性能问题。

### 3. 消息传递方式

**select**

内核需要将消息传递到用户空间，都需要内核拷贝动作

**poll**

同上

**epoll**

epoll通过内核和用户空间共享一块内存来实现的

综上，在选择select，poll，epoll时要根据具体的使用场合以及这三种方式的自身特点。表面上看epoll的性能最好，但是在连接数少并且连接都十分活跃的情况下，select和poll的性能可能比epoll好，毕竟epoll的通知机制需要很多函数回调。
