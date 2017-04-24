---
title: Titan的pluggable storage backend
layout: post
---


如何实现一个Titan storage backend
-------------------------------


### primary backend storage

主要实现类在 com.thinkaurelius.titan.diskstorage 下。


Titan根据后端不同的存储类型，定义了相应的接口：

* KeyColumnValueStore(I): Interface to a data store that has a BigTable like representation of its data.
	* BaseKeyColumnValueAdapter(C)
		* OrderedKeyValueStoreAdapter(C): Wraps a OrderedKeyValueStore and exposes it as a KeyColumnValueStore.
	* InMemoryKeyColumnValueStore(C): An in-memory implementation of KeyColumnValueStore.
	* HBaseKeyColumnValueStore: HBase backend store
	* CassandraEmbeddedKeyColumnValueStore
	* CassandraThriftKeyColumnValueStore: A Titan KeyColumnValueStore backed by Cassandra. This uses the Cassandra Thrift API.
* KeyValueStore(I): Interface for a data store that represents data in the simple key->value data model where each key is uniquely associated with a value.
	* OrderedKeyValueStore(I): A KeyValueStore where the keys are ordered such that keys can be retrieved in order.
		* BerkeleyJEKeyValueStore(C): BerkeleyDB的实现

每一个具体的backend Store都有相应的Manager类管理:

* StoreManager(I): Generic interface to a backend storage engine
	* AbstractStoreManager(C): Abstract Store Manager used as the basis for concrete StoreManager implementations.
		* DistributedStoreManager(C): Abstract class that handles configuration options shared by all distributed storage backends.
			* AbstractCassandraStoreManager(C)
				* AstyanaxStoreManager(C)
				* CassandraEmbeddedStoreManager(C)
				* CassandraThriftStoreManager(C): This class creates CassandraThriftKeyColumnValueStore and handles Cassandra-backed allocation of vertex IDs for Titan (when so configured).
			* HBaseStoreManager(C): Storage Manager for HBase
		* LocalStoreManager(C): Abstract Store Manager used as the basis for local StoreManager implementations.
			* BerkeleyJEStoreManager(C)
	* KeyColumnValueStoreManager(I): KeyColumnValueStoreManager provides the persistence context to the graph database storage backend.
		* AbstractCassandraStoreManager
			* AstyanaxStoreManager(C)
			* CassandraEmbeddedStoreManager(C)
			* CassandraThriftStoreManager(C): This class creates CassandraThriftKeyColumnValueStore and handles Cassandra-backed allocation of vertex IDs for Titan (when so configured).
		* HBaseStoreManager(C): Storage Manager for HBase
		* InMemoryStoreManager(C): In-memory backend storage engine.
		* OrderedKeyValueStoreManagerAdapter: Wraps a OrderedKeyValueStoreManager and exposes it as a KeyColumnValueStoreManager.
	* KeyValueStoreManager(I): StoreManager for KeyValueStore.
		* OrderedKeyValueStoreManager(I): A KeyValueStoreManager where the stores maintain keys in their natural order.
			* BerkeleyJEStoreManager(C)

另外，还有一个可选的事务接口：

* BaseTransaction(I): Represents a transaction for a particular storage backend.
	* LoggableTransaction(I)
		* BackendTransaction(C): Bundles all storage/index transactions and provides a proxy for some of their methods for convenience. Also increases robustness of read call by attempting read calls multiple times on failure.
		* CacheTransaction(C)
		* IndexTransaction(C): Wraps the transaction handle of an index and buffers all mutations against an index for efficiency. Also acts as a proxy to the IndexProvider methods.
	* BaseTransactionConfigurable(I): exposes a configuration object of type BaseTransactionConfig for this particular transaction.
		* DefaultTransaction(C)
		* StoreTransaction(I): A transaction handle uniquely identifies a transaction on the storage backend.
			* AbstractStoreTransaction(C): Abstract implementation of StoreTransaction to be used as the basis for more specific implementations.
				* DynamoDBStoreTransaction(C): Transaction is used to store expected values of each column for each key in a transaction
				* InMemoryTransaction(C)
				* NoOpStoreTransaction(C)
				* HBaseTransaction: creates a transaction type specific to HBase, which lets us check for user errors like passing a Cassandra transaction into a HBase method.
				* BerkeleyJETx
		* CacheTransaction(C)
	* IndexTransaction(C): Wraps the transaction handle of an index and buffers all mutations against an index for efficiency. Also acts as a proxy to the IndexProvider methods.

其他一些辅助类：

* Backend: backend配置信息
* Mutation: Container for collection mutations against a data store. Mutations are either additions or deletions.
* Comparable<T>
	* StaticBuffer: A Buffer that only allows static access. This Buffer is immutable if any returned byte array or ByteBuffer is not mutated.
		* Entry (also extends MetaAnnotated): An entry is the primitive persistence unit used in the graph database storage backend.
			* ReadBuffer (also extends ScanBuffer): A Buffer that allows sequential reads and static reads. Should not be used by multiple threads.

**总结**

要作为Titan的storage backend，需要做的事情如下：

1. 实现KeyColumnValueStore或者KeyValueStore
2. 实现相应的StoreManager
3. 实现BaseTransaction [可选]


### external index providers


主要实现类在 com.thinkaurelius.titan.diskstorage.indexing 下。

* IndexInformation: An IndexInformation gives basic information on what a particular IndexProvider supports.
	* StandardIndexInformation
	* IndexProvider: External index for querying.
		* ElasticSearchIndex


具体实例分析
----------

### BerkeleyDB backend storage

真的就三个类。

#### 1、BerkeleyJEKeyValueStore

首先看看BerkeleyJEKeyValueStore，它实现了OrderedKeyValueStore接口:

	package com.thinkaurelius.titan.diskstorage.berkeleyje;

	public class BerkeleyJEKeyValueStore implements OrderedKeyValueStore {

	    private static final Logger log = LoggerFactory.getLogger(BerkeleyJEKeyValueStore.class);

	    /// DatabaseEntry 是 BerkeleyDB 记录的key和data的类型。
	    private static final StaticBuffer.Factory<DatabaseEntry> ENTRY_FACTORY = new StaticBuffer.Factory<DatabaseEntry>() {
	        @Override
	        public DatabaseEntry get(byte[] array, int offset, int limit) {
	            return new DatabaseEntry(array,offset,limit-offset);
	        }
	    };


	    private final Database db; /// BerkeleyDB实例引用
	    private final String name; 

	    /// 引用了manager，主要是为了在close()的时候移除自己：manager.removeDatabase(this)
	    private final BerkeleyJEStoreManager manager;

	    private boolean isOpen;

	    public BerkeleyJEKeyValueStore(String n, Database data, BerkeleyJEStoreManager m) {
	        db = data;
	        name = n;
	        manager = m;
	        isOpen = true;
	    }

	    public DatabaseConfig getConfiguration() throws BackendException {
	        try {
	            return db.getConfig();
	        } catch (DatabaseException e) {
	            throw new PermanentBackendException(e);
	        }
	    }

	    @Override
	    public String getName() {
	        return name;
	    }

	    private static final Transaction getTransaction(StoreTransaction txh) {
	        Preconditions.checkArgument(txh!=null);
	        return ((BerkeleyJETx) txh).getTransaction();
	    }

	    @Override
	    public synchronized void close() throws BackendException {
	        try {
	            if(isOpen) db.close();
	        } catch (DatabaseException e) {
	            throw new PermanentBackendException(e);
	        }
	        if (isOpen) manager.removeDatabase(this);
	        isOpen = false;
	    }

	    @Override
	    public StaticBuffer get(StaticBuffer key, StoreTransaction txh) throws BackendException {
	        Transaction tx = getTransaction(txh);
	        try {
	            DatabaseEntry dbkey = key.as(ENTRY_FACTORY);
	            DatabaseEntry data = new DatabaseEntry();

	            log.trace("db={}, op=get, tx={}", name, txh);

	            OperationStatus status = db.get(tx, dbkey, data, getLockMode(txh));

	            if (status == OperationStatus.SUCCESS) {
	                return getBuffer(data);
	            } else {
	                return null;
	            }
	        } catch (DatabaseException e) {
	            throw new PermanentBackendException(e);
	        }
	    }

	    @Override
	    public boolean containsKey(StaticBuffer key, StoreTransaction txh) throws BackendException {
	        return get(key,txh)!=null;
	    }

	    @Override
	    public void acquireLock(StaticBuffer key, StaticBuffer expectedValue, StoreTransaction txh) throws BackendException {
	        if (getTransaction(txh) == null) {
	            log.warn("Attempt to acquire lock with transactions disabled");
	        } //else we need no locking
	    }

	    @Override
	    public RecordIterator<KeyValueEntry> getSlice(KVQuery query, StoreTransaction txh) throws BackendException {
	        log.trace("beginning db={}, op=getSlice, tx={}", name, txh);
	        Transaction tx = getTransaction(txh);
	        Cursor cursor = null;
	        final StaticBuffer keyStart = query.getStart();
	        final StaticBuffer keyEnd = query.getEnd();
	        final KeySelector selector = query.getKeySelector();
	        final List<KeyValueEntry> result = new ArrayList<KeyValueEntry>();
	        try {
	            DatabaseEntry foundKey = keyStart.as(ENTRY_FACTORY);
	            DatabaseEntry foundData = new DatabaseEntry();

	            cursor = db.openCursor(tx, null);
	            /// getSearchKeyRange将游标定位到跟foundKey最接近的地方，再用Cursor.getNext就能得到游标下一个
	            OperationStatus status = cursor.getSearchKeyRange(foundKey, foundData, getLockMode(txh));
	            //Iterate until given condition is satisfied or end of records
	            while (status == OperationStatus.SUCCESS) {
	                StaticBuffer key = getBuffer(foundKey); /// getSearchKeyRange返回的key值

	                if (key.compareTo(keyEnd) >= 0)
	                    break;

	                if (selector.include(key)) {
	                    result.add(new KeyValueEntry(key, getBuffer(foundData)));
	                }

	                if (selector.reachedLimit())
	                    break;

	                /// 调用cursor.getNext得到下一个key，因为是key有序的，所以得到的就是一个key range。
	                status = cursor.getNext(foundKey, foundData, getLockMode(txh));
	            }
	            log.trace("db={}, op=getSlice, tx={}, resultcount={}", name, txh, result.size());

	            return new RecordIterator<KeyValueEntry>() {
	                private final Iterator<KeyValueEntry> entries = result.iterator();

	                @Override
	                public boolean hasNext() {
	                    return entries.hasNext();
	                }

	                @Override
	                public KeyValueEntry next() {
	                    return entries.next();
	                }

	                @Override
	                public void close() {
	                }

	                @Override
	                public void remove() {
	                    throw new UnsupportedOperationException();
	                }
	            };
	        } catch (Exception e) {
	            throw new PermanentBackendException(e);
	        } finally {
	            try {
	                if (cursor != null) cursor.close();
	            } catch (Exception e) {
	                throw new PermanentBackendException(e);
	            }
	        }
	    }

	    @Override
	    public Map<KVQuery,RecordIterator<KeyValueEntry>> getSlices(List<KVQuery> queries, StoreTransaction txh) throws BackendException {
	        throw new UnsupportedOperationException();
	    }

	    @Override
	    public void insert(StaticBuffer key, StaticBuffer value, StoreTransaction txh) throws BackendException {
	        insert(key, value, txh, true);
	    }

	    public void insert(StaticBuffer key, StaticBuffer value, StoreTransaction txh, boolean allowOverwrite) throws BackendException {
	        Transaction tx = getTransaction(txh);
	        try {
	            OperationStatus status;

	            log.trace("db={}, op=insert, tx={}", name, txh);

	            if (allowOverwrite)
	                status = db.put(tx, key.as(ENTRY_FACTORY), value.as(ENTRY_FACTORY));
	            else
	                status = db.putNoOverwrite(tx, key.as(ENTRY_FACTORY), value.as(ENTRY_FACTORY));

	            if (status != OperationStatus.SUCCESS) {
	                if (status == OperationStatus.KEYEXIST) {
	                    throw new PermanentBackendException("Key already exists on no-overwrite.");
	                } else {
	                    throw new PermanentBackendException("Could not write entity, return status: " + status);
	                }
	            }
	        } catch (DatabaseException e) {
	            throw new PermanentBackendException(e);
	        }
	    }


	    @Override
	    public void delete(StaticBuffer key, StoreTransaction txh) throws BackendException {
	        log.trace("Deletion");
	        Transaction tx = getTransaction(txh);
	        try {
	            log.trace("db={}, op=delete, tx={}", name, txh);
	            OperationStatus status = db.delete(tx, key.as(ENTRY_FACTORY));
	            if (status != OperationStatus.SUCCESS) {
	                throw new PermanentBackendException("Could not remove: " + status);
	            }
	        } catch (DatabaseException e) {
	            throw new PermanentBackendException(e);
	        }
	    }

	    private static StaticBuffer getBuffer(DatabaseEntry entry) {
	        return new StaticArrayBuffer(entry.getData(),entry.getOffset(),entry.getOffset()+entry.getSize());
	    }

	    private static LockMode getLockMode(StoreTransaction txh) {
	        return ((BerkeleyJETx)txh).getLockMode();
	    }
	}

实现其实蛮简单的，因为 OrderedKeyValueStore 的接口本身就很简单。基本都是简单的K-V增删改查，除了 getSlice ，用到了Cursor。

#### 2、BerkeleyJEStoreManager

然后我们来看看 BerkeleyJEStoreManager，它实现了 OrderedKeyValueStoreManager；并且因为是本地存储，所以它还拓展了LocalStoreManager：

	package com.thinkaurelius.titan.diskstorage.berkeleyje;

	@PreInitializeConfigOptions
	public class BerkeleyJEStoreManager extends LocalStoreManager implements OrderedKeyValueStoreManager {

	    public static final ConfigNamespace BERKELEY_NS =
	            new ConfigNamespace(GraphDatabaseConfiguration.STORAGE_NS, "berkeleyje", "BerkeleyDB JE configuration options");

	    public static final ConfigOption<Integer> JVM_CACHE =
	            new ConfigOption<Integer>(BERKELEY_NS,"cache-percentage",
	            "Percentage of JVM heap reserved for BerkeleyJE's cache",
	            ConfigOption.Type.MASKABLE, 65, ConfigOption.positiveInt());

	    public static final ConfigOption<String> LOCK_MODE =
	            new ConfigOption<>(BERKELEY_NS, "lock-mode",
	            "The BDB record lock mode used for read operations",
	            ConfigOption.Type.MASKABLE, String.class, LockMode.DEFAULT.toString(), disallowEmpty(String.class));

	    public static final ConfigOption<String> ISOLATION_LEVEL =
	            new ConfigOption<>(BERKELEY_NS, "isolation-level",
	            "The isolation level used by transactions",
	            ConfigOption.Type.MASKABLE,  String.class,
	            IsolationLevel.REPEATABLE_READ.toString(), disallowEmpty(String.class));

	    private final Map<String, BerkeleyJEKeyValueStore> stores;

        /// BerkeleyDB environment. Environments include support for some or all of caching, locking, logging and transactions.
	    protected Environment environment;
	    protected final StoreFeatures features;

	    public BerkeleyJEStoreManager(Configuration configuration) throws BackendException {
	        super(configuration);
	        stores = new HashMap<String, BerkeleyJEKeyValueStore>();

	        int cachePercentage = configuration.get(JVM_CACHE);
	        initialize(cachePercentage);

	        features = new StandardStoreFeatures.Builder()
	                    .orderedScan(true)
	                    .transactional(transactional)
	                    .keyConsistent(GraphDatabaseConfiguration.buildGraphConfiguration())
	                    .locking(true)
	                    .keyOrdered(true)
	                    .scanTxConfig(GraphDatabaseConfiguration.buildGraphConfiguration()
	                            .set(ISOLATION_LEVEL, IsolationLevel.READ_UNCOMMITTED.toString()))
	                    .supportsInterruption(false)
	                    .build();

	    }

	    private void initialize(int cachePercent) throws BackendException {
	        try {
	            EnvironmentConfig envConfig = new EnvironmentConfig();
	            envConfig.setAllowCreate(true);
	            envConfig.setTransactional(transactional);
	            envConfig.setCachePercent(cachePercent);

	            if (batchLoading) {
	                envConfig.setConfigParam(EnvironmentConfig.ENV_RUN_CHECKPOINTER, "false");
	                envConfig.setConfigParam(EnvironmentConfig.ENV_RUN_CLEANER, "false");
	            }

	            //Open the environment
	            environment = new Environment(directory, envConfig);


	        } catch (DatabaseException e) {
	            throw new PermanentBackendException("Error during BerkeleyJE initialization: ", e);
	        }

	    }

	    @Override
	    public StoreFeatures getFeatures() {
	        return features;
	    }

	    @Override
	    public List<KeyRange> getLocalKeyPartition() throws BackendException {
	        throw new UnsupportedOperationException();
	    }

	    @Override
	    public BerkeleyJETx beginTransaction(final BaseTransactionConfig txCfg) throws BackendException {
	        try {
	            Transaction tx = null;

	            Configuration effectiveCfg =
	                    new MergedConfiguration(txCfg.getCustomOptions(), getStorageConfig());

	            if (transactional) {
	                TransactionConfig txnConfig = new TransactionConfig();
	                ConfigOption.getEnumValue(effectiveCfg.get(ISOLATION_LEVEL),IsolationLevel.class).configure(txnConfig);
	                tx = environment.beginTransaction(null, txnConfig);
	            }
	            BerkeleyJETx btx = new BerkeleyJETx(tx, ConfigOption.getEnumValue(effectiveCfg.get(LOCK_MODE),LockMode.class), txCfg);

	            if (log.isTraceEnabled()) {
	                log.trace("Berkeley tx created", new TransactionBegin(btx.toString()));
	            }

	            return btx;
	        } catch (DatabaseException e) {
	            throw new PermanentBackendException("Could not start BerkeleyJE transaction", e);
	        }
	    }

	    @Override
	    public BerkeleyJEKeyValueStore openDatabase(String name) throws BackendException {
	        Preconditions.checkNotNull(name);
	        if (stores.containsKey(name)) {
	            BerkeleyJEKeyValueStore store = stores.get(name);
	            return store;
	        }
	        try {
	            DatabaseConfig dbConfig = new DatabaseConfig();
	            dbConfig.setReadOnly(false);
	            dbConfig.setAllowCreate(true);
	            dbConfig.setTransactional(transactional);

	            dbConfig.setKeyPrefixing(true);

	            if (batchLoading) {
	                dbConfig.setDeferredWrite(true);
	            }

	            Database db = environment.openDatabase(null, name, dbConfig);

	            log.debug("Opened database {}", name, new Throwable());

	            BerkeleyJEKeyValueStore store = new BerkeleyJEKeyValueStore(name, db, this);
	            stores.put(name, store);
	            return store;
	        } catch (DatabaseException e) {
	            throw new PermanentBackendException("Could not open BerkeleyJE data store", e);
	        }
	    }

	    @Override
	    public void mutateMany(Map<String, KVMutation> mutations, StoreTransaction txh) throws BackendException {
	        for (Map.Entry<String,KVMutation> muts : mutations.entrySet()) {
	            BerkeleyJEKeyValueStore store = openDatabase(muts.getKey());
	            KVMutation mut = muts.getValue();

	            if (!mut.hasAdditions() && !mut.hasDeletions()) {
	                log.debug("Empty mutation set for {}, doing nothing", muts.getKey());
	            } else {
	                log.debug("Mutating {}", muts.getKey());
	            }

	            if (mut.hasAdditions()) {
	                for (KeyValueEntry entry : mut.getAdditions()) {
	                    store.insert(entry.getKey(),entry.getValue(),txh);
	                    log.trace("Insertion on {}: {}", muts.getKey(), entry);
	                }
	            }
	            if (mut.hasDeletions()) {
	                for (StaticBuffer del : mut.getDeletions()) {
	                    store.delete(del,txh);
	                    log.trace("Deletion on {}: {}", muts.getKey(), del);
	                }
	            }
	        }
	    }

	    void removeDatabase(BerkeleyJEKeyValueStore db) {
	        if (!stores.containsKey(db.getName())) {
	            throw new IllegalArgumentException("Tried to remove an unkown database from the storage manager");
	        }
	        String name = db.getName();
	        stores.remove(name);
	        log.debug("Removed database {}", name);
	    }


	    @Override
	    public void close() throws BackendException {
	        if (environment != null) {
	            if (!stores.isEmpty())
	                throw new IllegalStateException("Cannot shutdown manager since some databases are still open");
	            try {
	                // TODO this looks like a race condition
	                //Wait just a little bit before closing so that independent transaction threads can clean up.
	                Thread.sleep(30);
	            } catch (InterruptedException e) {
	                //Ignore
	            }
	            try {
	                environment.close();
	            } catch (DatabaseException e) {
	                throw new PermanentBackendException("Could not close BerkeleyJE database", e);
	            }
	        }

	    }

	    @Override
	    public void clearStorage() throws BackendException {
	        if (!stores.isEmpty())
	            throw new IllegalStateException("Cannot delete store, since database is open: " + stores.keySet().toString());

	        Transaction tx = null;
	        for (String db : environment.getDatabaseNames()) {
	            environment.removeDatabase(tx, db);
	            log.debug("Removed database {} (clearStorage)", db);
	        }
	        close();
	        IOUtils.deleteFromDirectory(directory);
	    }

	    @Override
	    public String getName() {
	        return getClass().getSimpleName() + ":" + directory.toString();
	    }


	    public static enum IsolationLevel {
	        READ_UNCOMMITTED {
	            @Override
	            void configure(TransactionConfig cfg) {
	                cfg.setReadUncommitted(true);
	            }
	        }, READ_COMMITTED {
	            @Override
	            void configure(TransactionConfig cfg) {
	                cfg.setReadCommitted(true);

	            }
	        }, REPEATABLE_READ {
	            @Override
	            void configure(TransactionConfig cfg) {
	                // This is the default and has no setter
	            }
	        }, SERIALIZABLE {
	            @Override
	            void configure(TransactionConfig cfg) {
	                cfg.setSerializableIsolation(true);
	            }
	        };

	        abstract void configure(TransactionConfig cfg);
	    };

	    private static class TransactionBegin extends Exception {
	        private static final long serialVersionUID = 1L;

	        private TransactionBegin(String msg) {
	            super(msg);
	        }
	    }
	}

实现也是很直观，有三个成员变量：

	private final Map<String, BerkeleyJEKeyValueStore> stores;

	/// BerkeleyDB environment. Environments include support for some or all of caching, locking, logging and transactions.
	protected Environment environment;
	protected final StoreFeatures features;

一个是它管理的BerkeleyJEKeyValueStore，还有BerkeleyDB的environment，用于创建BerkeleyDB实例(openDatabase)和处理事务(beginTransaction)。
然后就是实现接口的方法。关键的接口就三个：

1、开始一个事务: `public BerkeleyJETx beginTransaction(final BaseTransactionConfig txCfg) throws BackendException;`

2、打开一个Database(这里是BerkeleyJEKeyValueStore): `public BerkeleyJEKeyValueStore openDatabase(String name) throws BackendException;`

3、处理批量更新: `public void mutateMany(Map<String, KVMutation> mutations, StoreTransaction txh) throws BackendException;`

这个接口定义在这里有点让人惊讶，因为对数据的真实操作应该是 BerkeleyJEKeyValueStore 要做的事情(Titan的接口和类层级定义过于复杂不够清晰。。)，而事实上，最终的操作确实也是委托给 BerkeleyJEKeyValueStore 进行更新。


#### 3、BerkeleyJETx

最后我们来看一下BerkeleyJETx，继承至AbstractStoreTransaction，主要是实现了commit和rollback接口：

	package com.thinkaurelius.titan.diskstorage.berkeleyje;

	import com.google.common.base.Preconditions;
	import com.sleepycat.je.Cursor;
	import com.sleepycat.je.DatabaseException;
	import com.sleepycat.je.LockMode;
	import com.sleepycat.je.Transaction;
	import com.thinkaurelius.titan.diskstorage.BackendException;
	import com.thinkaurelius.titan.diskstorage.PermanentBackendException;
	import com.thinkaurelius.titan.diskstorage.BaseTransactionConfig;
	import com.thinkaurelius.titan.diskstorage.common.AbstractStoreTransaction;

	public class BerkeleyJETx extends AbstractStoreTransaction {

	    private static final Logger log = LoggerFactory.getLogger(BerkeleyJETx.class);

	    private volatile Transaction tx;
	    private final List<Cursor> openCursors = new ArrayList<Cursor>();
	    private final LockMode lm;

	    public BerkeleyJETx(Transaction t, LockMode lockMode, BaseTransactionConfig config) {
	        super(config);
	        tx = t;
	        lm = lockMode;
	        // tx may be null
	        Preconditions.checkNotNull(lm);
	    }

	    public Transaction getTransaction() {
	        return tx;
	    }

	    void registerCursor(Cursor cursor) {
	        Preconditions.checkArgument(cursor != null);
	        synchronized (openCursors) {
	            //TODO: attempt to remove closed cursors if there are too many
	            openCursors.add(cursor);
	        }
	    }

	    private void closeOpenIterators() throws BackendException {
	        for (Cursor cursor : openCursors) {
	            cursor.close();
	        }
	    }

	    LockMode getLockMode() {
	        return lm;
	    }

	    @Override
	    public synchronized void rollback() throws BackendException {
	        super.rollback();
	        if (tx == null) return;
	        if (log.isTraceEnabled())
	            log.trace("{} rolled back", this.toString(), new TransactionClose(this.toString()));
	        try {
	            closeOpenIterators();
	            tx.abort();
	            tx = null;
	        } catch (DatabaseException e) {
	            throw new PermanentBackendException(e);
	        }
	    }

	    @Override
	    public synchronized void commit() throws BackendException {
	        super.commit();
	        if (tx == null) return;
	        if (log.isTraceEnabled())
	            log.trace("{} committed", this.toString(), new TransactionClose(this.toString()));

	        try {
	            closeOpenIterators();
	            tx.commit();
	            tx = null;
	        } catch (DatabaseException e) {
	            throw new PermanentBackendException(e);
	        }
	    }

	    @Override
	    public String toString() {
	        return getClass().getSimpleName() + (null == tx ? "nulltx" : tx.toString());
	    }

	    private static class TransactionClose extends Exception {
	        private static final long serialVersionUID = 1L;

	        private TransactionClose(String msg) {
	            super(msg);
	        }
	    }
	}

这个类主要是对`com.sleepycat.je.Transaction`的一个封装。所以Titan的事务实现其实是依赖于底层存储的具体实现的，并没有做什么事情。


### HBase backend storage

我们再分析一下HBase的实现，在 titan-hbase-core 项目中。比BerkeleyDB的实现来说，多了几个类：

* ConnectionMask.java: 封装 org.apache.hadoop.hbase.client.Connection，用于获取TableMask和AdminMask
* TableMask.java: 封装 org.apache.hadoop.hbase.client.Table/org.apache.hadoop.hbase.client.HTableInterface
* AdminMask.java: 封装 org.apache.hadoop.hbase.client.Admin/org.apache.hadoop.hbase.client.HBaseAdmin 
* HBaseCompat.java: 对 org.apache.hadoop.hbase.io.compress.Compression 进行封装
* HBaseCompatLoader.java: 动态加载HBaseCompat实例
* HBaseKeyColumnValueStore.java
* HBaseStoreManager.java
* HBaseTransaction.java

主要是多个三个Mask相关的类和两个Compat相关的类。这五个类主要是对HBase的一个封装，因为HBase0.9x和1.0的接口发生了变化。Titan想要同时支持这两个版本。这几个类并不是核心类，我们还是主要分析 HBaseKeyColumnValueStore.java、HBaseStoreManager.java 和 HBaseTransaction.java 这三个类。

#### 1、HBaseKeyColumnValueStore

跟BerkeleyDB不同，HBaseKeyColumnValueStore实现的接口是KeyColumnValueStore，这个接口相对复杂一些：

	/**
	 * Interface to a data store that has a BigTable like representation of its data. In other words, the data store is comprised of a set of rows
	 * each of which is uniquely identified by a key. Each row is composed of a column-value pairs. For a given key, a subset of the column-value
	 * pairs that fall within a column interval can be quickly retrieved.
	 * <p/>
	 * This interface provides methods for retrieving and mutating the data.
	 * <p/>
	 * In this generic representation keys, columns and values are represented as ByteBuffers.
	 * <p/>
	 */
	public interface KeyColumnValueStore {

	    public static final List<Entry> NO_ADDITIONS = ImmutableList.of();
	    public static final List<StaticBuffer> NO_DELETIONS = ImmutableList.of();

	    /**
	     * Retrieves the list of entries (i.e. column-value pairs) for a specified query.
	     *
	     * @param query Query to get results for
	     * @param txh   Transaction
	     * @return List of entries up to a maximum of "limit" entries
	     * @throws com.thinkaurelius.titan.diskstorage.BackendException when columnEnd < columnStart
	     * @see KeySliceQuery
	     */
	    public EntryList getSlice(KeySliceQuery query, StoreTransaction txh) throws BackendException;

	    /**
	     * Retrieves the list of entries (i.e. column-value pairs) as specified by the given {@link SliceQuery} for all
	     * of the given keys together.
	     *
	     * @param keys  List of keys
	     * @param query Slicequery specifying matching entries
	     * @param txh   Transaction
	     * @return The result of the query for each of the given keys as a map from the key to the list of result entries.
	     * @throws com.thinkaurelius.titan.diskstorage.BackendException
	     */
	    public Map<StaticBuffer,EntryList> getSlice(List<StaticBuffer> keys, SliceQuery query, StoreTransaction txh) throws BackendException;

	    /**
	     * Verifies acquisition of locks {@code txh} from previous calls to
	     * {@link #acquireLock(StaticBuffer, StaticBuffer, StaticBuffer, StoreTransaction)}
	     * , then writes supplied {@code additions} and/or {@code deletions} to
	     * {@code key} in the underlying data store. Deletions are applied strictly
	     * before additions. In other words, if both an addition and deletion are
	     * supplied for the same column, then the column will first be deleted and
	     * then the supplied Entry for the column will be added.
	     * <p/>
	     * Implementations which don't support locking should skip the initial lock
	     * verification step but otherwise behave as described above.
	     *
	     * @param key       the key under which the columns in {@code additions} and
	     *                  {@code deletions} will be written
	     * @param additions the list of Entry instances representing column-value pairs to
	     *                  create under {@code key}, or null to add no column-value pairs
	     * @param deletions the list of columns to delete from {@code key}, or null to
	     *                  delete no columns
	     * @param txh       the transaction to use
	     * @throws com.thinkaurelius.titan.diskstorage.locking.PermanentLockingException if locking is supported by the implementation and at least
	     *                          one lock acquisition attempted by
	     *                          {@link #acquireLock(StaticBuffer, StaticBuffer, StaticBuffer, StoreTransaction)}
	     *                          has failed
	     */
	    public void mutate(StaticBuffer key, List<Entry> additions, List<StaticBuffer> deletions, StoreTransaction txh) throws BackendException;

	    /**
	     * Attempts to claim a lock on the value at the specified {@code key} and
	     * {@code column} pair. These locks are discretionary.
 	     * <p/>
	     * 这个接口是可选的。
	     */
	    public void acquireLock(StaticBuffer key, StaticBuffer column, StaticBuffer expectedValue, StoreTransaction txh) throws BackendException;

	    /**
	     * Returns a {@link KeyIterator} over all keys that fall within the key-range specified by the given query and have one or more columns matching the column-range.
	     * Calling {@link KeyIterator#getEntries()} returns the list of all entries that match the column-range specified by the given query.
	     * <p/>
	     * This method is only supported by stores which keep keys in byte-order.
	     */
	    public KeyIterator getKeys(KeyRangeQuery query, StoreTransaction txh) throws BackendException;

	    /**
	     * Returns a {@link KeyIterator} over all keys in the store that have one or more columns matching the column-range. Calling {@link KeyIterator#getEntries()}
	     * returns the list of all entries that match the column-range specified by the given query.
	     * <p/>
	     * This method is only supported by stores which do not keep keys in byte-order.
	     */
	    public KeyIterator getKeys(SliceQuery query, StoreTransaction txh) throws BackendException;
	    // like current getKeys if column-slice is such that it queries for vertex state property

	    public String getName();

	    public void close() throws BackendException;

	}

HBaseKeyColumnValueStore的实现有点冗长，这里就不贴代码了。感兴趣的读者自己看源码，不难理解。

#### 2、HBaseStoreManager

	public class HBaseStoreManager extends DistributedStoreManager implements KeyColumnValueStoreManager {

	    /// ...配置项相关

        private static final BiMap<String, String> SHORT_CF_NAME_MAP =
        ImmutableBiMap.<String, String>builder()
                .put(INDEXSTORE_NAME, "g")
                .put(INDEXSTORE_NAME + LOCK_STORE_SUFFIX, "h")
                .put(ID_STORE_NAME, "i")
                .put(EDGESTORE_NAME, "e")
                .put(EDGESTORE_NAME + LOCK_STORE_SUFFIX, "f")
                .put(SYSTEM_PROPERTIES_STORE_NAME, "s")
                .put(SYSTEM_PROPERTIES_STORE_NAME + LOCK_STORE_SUFFIX, "t")
                .put(SYSTEM_MGMT_LOG_NAME, "m")
                .put(SYSTEM_TX_LOG_NAME, "l")
                .build();

	    // Immutable instance fields
	    private final String tableName;
	    private final String compression;
	    private final int regionCount;
	    private final int regionsPerServer;
	    private final ConnectionMask cnx;
	    private final org.apache.hadoop.conf.Configuration hconf;
	    private final boolean shortCfNames;
	    private final boolean skipSchemaCheck;
	    private final String compatClass;
	    private final HBaseCompat compat;

	    private static final ConcurrentHashMap<HBaseStoreManager, Throwable> openManagers =
	            new ConcurrentHashMap<HBaseStoreManager, Throwable>();

	    // Mutable instance state
	    private final ConcurrentMap<String, HBaseKeyColumnValueStore> openStores;

	    public HBaseStoreManager(com.thinkaurelius.titan.diskstorage.configuration.Configuration config) throws BackendException {
	        /// .. 初始化
	    }

	    @Override
	    public Deployment getDeployment() {
	    	/// ...
	    }

	    @Override
	    public String toString() {
	        return "hbase[" + tableName + "@" + super.toString() + "]";
	    }

	    public void dumpOpenManagers() {
	        /// ...
	    }

	    @Override
	    public void close() {
	        ///...
	    }

	    @Override
	    public StoreFeatures getFeatures() {
	        /// ...
	    }

	    @Override
	    public void mutateMany(Map<String, Map<StaticBuffer, KCVMutation>> mutations, StoreTransaction txh) throws BackendException {
	        final MaskedTimestamp commitTime = new MaskedTimestamp(txh);
	        // In case of an addition and deletion with identical timestamps, the
	        // deletion tombstone wins.
	        // http://hbase.apache.org/book/versions.html#d244e4250
	        Map<StaticBuffer, Pair<Put, Delete>> commandsPerKey =
	                convertToCommands(
	                        mutations,
	                        commitTime.getAdditionTime(times),
	                        commitTime.getDeletionTime(times));

	        List<Row> batch = new ArrayList<Row>(commandsPerKey.size()); // actual batch operation

	        // convert sorted commands into representation required for 'batch' operation
	        for (Pair<Put, Delete> commands : commandsPerKey.values()) {
	            if (commands.getFirst() != null)
	                batch.add(commands.getFirst());

	            if (commands.getSecond() != null)
	                batch.add(commands.getSecond());
	        }

	        try {
	            TableMask table = null;

	            try {
	                table = cnx.getTable(tableName);
	                table.batch(batch, new Object[batch.size()]);
	            } finally {
	                IOUtils.closeQuietly(table);
	            }
	        } catch (IOException e) {
	            throw new TemporaryBackendException(e);
	        } catch (InterruptedException e) {
	            throw new TemporaryBackendException(e);
	        }

	        sleepAfterWrite(txh, commitTime);
	    }

	    @Override
	    public KeyColumnValueStore openDatabase(String longName, StoreMetaData.Container metaData) throws BackendException {

	        HBaseKeyColumnValueStore store = openStores.get(longName);

	        if (store == null) {
	            final String cfName = shortCfNames ? shortenCfName(longName) : longName;

	            HBaseKeyColumnValueStore newStore = new HBaseKeyColumnValueStore(this, cnx, tableName, cfName, longName);

	            store = openStores.putIfAbsent(longName, newStore); // nothing bad happens if we loose to other thread

	            if (store == null) {
	                if (!skipSchemaCheck) {
	                    int cfTTLInSeconds = -1;
	                    if (metaData.contains(StoreMetaData.TTL)) {
	                        cfTTLInSeconds = metaData.get(StoreMetaData.TTL);
	                    }
	                    ensureColumnFamilyExists(tableName, cfName, cfTTLInSeconds);
	                }

	                store = newStore;
	            }
	        }

	        return store;
	    }

	    @Override
	    public StoreTransaction beginTransaction(final BaseTransactionConfig config) throws BackendException {
	        return new HBaseTransaction(config);
	    }

	    @Override
	    public String getName() {
	        return tableName;
	    }

	    /**
	     * Deletes the specified table with all its columns.
	     * ATTENTION: Invoking this method will delete the table if it exists and therefore causes data loss.
	     */
	    @Override
	    public void clearStorage() throws BackendException {
	        /// ...
	    }

	    @Override
	    public List<KeyRange> getLocalKeyPartition() throws BackendException {

	        /// ...
	    }


	    /**
	     * This method generates the second argument to
	     * {@link HBaseAdmin#createTable(HTableDescriptor, byte[], byte[], int)}.
	     * <p/>
	     * From the {@code createTable} javadoc:
	     * "The start key specified will become the end key of the first region of
	     * the table, and the end key specified will become the start key of the
	     * last region of the table (the first region has a null start key and
	     * the last region has a null end key)"
	     * <p/>
	     * To summarize, the {@code createTable} argument called "startKey" is
	     * actually the end key of the first region.
	     */
	    private byte[] getStartKey(int regionCount) {
	        ByteBuffer regionWidth = ByteBuffer.allocate(4);
	        regionWidth.putInt((int)(((1L << 32) - 1L) / regionCount)).flip();
	        return StaticArrayBuffer.of(regionWidth).getBytes(0, 4);
	    }

	    /**
	     * Companion to {@link #getStartKey(int)}. See its javadoc for details.
	     */
	    private byte[] getEndKey(int regionCount) {
	        ByteBuffer regionWidth = ByteBuffer.allocate(4);
	        regionWidth.putInt((int)(((1L << 32) - 1L) / regionCount * (regionCount - 1))).flip();
	        return StaticArrayBuffer.of(regionWidth).getBytes(0, 4);
	    }

	    /**
	     * Convert Titan internal Mutation representation into HBase native commands.
	     *
	     * @param mutations    Mutations to convert into HBase commands.
	     * @param putTimestamp The timestamp to use for Put commands.
	     * @param delTimestamp The timestamp to use for Delete commands.
	     * @return Commands sorted by key converted from Titan internal representation.
	     * @throws com.thinkaurelius.titan.diskstorage.PermanentBackendException
	     */
	    private Map<StaticBuffer, Pair<Put, Delete>> convertToCommands(Map<String, Map<StaticBuffer, KCVMutation>> mutations,
	                                                                   final long putTimestamp,
	                                                                   final long delTimestamp) throws PermanentBackendException {
	        Map<StaticBuffer, Pair<Put, Delete>> commandsPerKey = new HashMap<StaticBuffer, Pair<Put, Delete>>();

	        for (Map.Entry<String, Map<StaticBuffer, KCVMutation>> entry : mutations.entrySet()) {

	            String cfString = getCfNameForStoreName(entry.getKey());
	            byte[] cfName = cfString.getBytes();

	            for (Map.Entry<StaticBuffer, KCVMutation> m : entry.getValue().entrySet()) {
	                byte[] key = m.getKey().as(StaticBuffer.ARRAY_FACTORY);
	                KCVMutation mutation = m.getValue();

	                Pair<Put, Delete> commands = commandsPerKey.get(m.getKey());

	                if (commands == null) {
	                    commands = new Pair<Put, Delete>();
	                    commandsPerKey.put(m.getKey(), commands);
	                }

	                if (mutation.hasDeletions()) {
	                    if (commands.getSecond() == null) {
	                        Delete d = new Delete(key);
	                        compat.setTimestamp(d, delTimestamp);
	                        commands.setSecond(d);
	                    }

	                    for (StaticBuffer b : mutation.getDeletions()) {
	                        commands.getSecond().deleteColumns(cfName, b.as(StaticBuffer.ARRAY_FACTORY), delTimestamp);
	                    }
	                }

	                if (mutation.hasAdditions()) {
	                    if (commands.getFirst() == null) {
	                        Put p = new Put(key, putTimestamp);
	                        commands.setFirst(p);
	                    }

	                    for (Entry e : mutation.getAdditions()) {
	                        commands.getFirst().add(cfName,
	                                e.getColumnAs(StaticBuffer.ARRAY_FACTORY),
	                                putTimestamp,
	                                e.getValueAs(StaticBuffer.ARRAY_FACTORY));
	                    }
	                }
	            }
	        }

	        return commandsPerKey;
	    }

	}

首先值得注意的是HBase的table和Column family定义： 

	private static final BiMap<String, String> SHORT_CF_NAME_MAP =
            ImmutableBiMap.<String, String>builder()
                    .put(INDEXSTORE_NAME, "g")
                    .put(INDEXSTORE_NAME + LOCK_STORE_SUFFIX, "h")
                    .put(ID_STORE_NAME, "i")
                    .put(EDGESTORE_NAME, "e")
                    .put(EDGESTORE_NAME + LOCK_STORE_SUFFIX, "f")
                    .put(SYSTEM_PROPERTIES_STORE_NAME, "s")
                    .put(SYSTEM_PROPERTIES_STORE_NAME + LOCK_STORE_SUFFIX, "t")
                    .put(SYSTEM_MGMT_LOG_NAME, "m")
                    .put(SYSTEM_TX_LOG_NAME, "l")
                    .build();

把常量展开得到long column family和short column family的对应关系如下：

* graphindex <=> g
* graphindex_lock_ <=> h
* titan_ids <=> i
* edgestore <=> e
* edgestore_lock_ <=> f
* system_properties <=> s
* system_properties_lock_ <=> t
* systemlog <=> m
* txlog <=> l

一共有9个Column family。然后在openDatabase时调用`ensureColumnFamilyExists(tableName, cfName, cfTTLInSeconds)`创建表结构：

    @Override
    public KeyColumnValueStore openDatabase(String longName, StoreMetaData.Container metaData) throws BackendException {

        HBaseKeyColumnValueStore store = openStores.get(longName);

        if (store == null) {
            final String cfName = shortCfNames ? shortenCfName(longName) : longName;

            HBaseKeyColumnValueStore newStore = new HBaseKeyColumnValueStore(this, cnx, tableName, cfName, longName);

            store = openStores.putIfAbsent(longName, newStore); // nothing bad happens if we loose to other thread

            if (store == null) {
                if (!skipSchemaCheck) {
                    int cfTTLInSeconds = -1;
                    if (metaData.contains(StoreMetaData.TTL)) {
                        cfTTLInSeconds = metaData.get(StoreMetaData.TTL);
                    }
                    ensureColumnFamilyExists(tableName, cfName, cfTTLInSeconds);
                }

                store = newStore;
            }
        }

        return store;
    }

    private void ensureColumnFamilyExists(String tableName, String columnFamily, int ttlInSeconds) throws BackendException {
        AdminMask adm = null;
        try {
            adm = getAdminInterface();
            HTableDescriptor desc = ensureTableExists(tableName, columnFamily, ttlInSeconds);

            Preconditions.checkNotNull(desc);

            HColumnDescriptor cf = desc.getFamily(columnFamily.getBytes());

            // Create our column family, if necessary
            if (cf == null) {
                try {
                    if (!adm.isTableDisabled(tableName)) {
                        adm.disableTable(tableName);
                    }
                } catch (){
                	...
                }

                try {
                    HColumnDescriptor cdesc = new HColumnDescriptor(columnFamily);

                    setCFOptions(cdesc, ttlInSeconds);

                    adm.addColumn(tableName, cdesc);

                    try {
                        logger.debug("Added HBase ColumnFamily {}, waiting for 1 sec. to propogate.", columnFamily);
                        Thread.sleep(1000L);
                    } catch (InterruptedException ie) {
                        throw new TemporaryBackendException(ie);
                    }

                    adm.enableTable(tableName);
                } catch (...){
                	...
                }
            }
        } finally {
            IOUtils.closeQuietly(adm);
        }
    }

注意：不同于BerkeleyDB，这里是一个Column family一个 HBaseKeyColumnValueStore 实例：

    HBaseKeyColumnValueStore newStore = new HBaseKeyColumnValueStore(this, cnx, tableName, cfName, longName);

实际上通过HBase shell，我们可以得到底层HBase的表结构信息：

	hbase(main):001:0> list
	TABLE
	titan
	1 row(s) in 0.3370 seconds

	=> ["titan"]
	hbase(main):002:0> describe 'titan'
	Table titan is ENABLED
	titan
	COLUMN FAMILIES DESCRIPTION
	{NAME => 'e', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'false', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', COMPRESSION => 'GZ', MIN_V
	ERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}
	{NAME => 'f', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'false', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', COMPRESSION => 'GZ', MIN_V
	ERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}
	{NAME => 'g', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'false', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', COMPRESSION => 'GZ', MIN_V
	ERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}
	{NAME => 'h', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'false', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', COMPRESSION => 'GZ', MIN_V
	ERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}
	{NAME => 'i', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'false', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', COMPRESSION => 'GZ', MIN_V
	ERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}
	{NAME => 'l', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'false', DATA_BLOCK_ENCODING => 'NONE', TTL => '604800 SECONDS (7 DAYS)', COMPRESSIO
	N => 'GZ', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}
	{NAME => 'm', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'false', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', COMPRESSION => 'GZ', MIN_V
	ERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}
	{NAME => 's', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'false', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', COMPRESSION => 'GZ', MIN_V
	ERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}
	{NAME => 't', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'false', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', COMPRESSION => 'GZ', MIN_V
	ERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}
	9 row(s) in 0.2050 seconds

可以看到只有一张表——titan，表中有9个Column Family。看一下里面的记录：

	hbase(main):002:0> scan 'titan', {limit => 20}

	ROW                                            COLUMN+CELL
	\x00\x00\x00\x00\x00\x00\x00\x00\x00\xE3/P    column=m:\x00\x05J \x85-BP\xA0ac12ba6b7181-BDSZYF0000b1683\xB1\x00\x00\x00\x00\x00\x00\x00\x01, timestamp=1488878417435001, value=\x80
	                                           \x81\x81\x04\x89
	\x00\x00\x00\x00\x00\x00\x00\x00\x00\xE3/P    column=m:\x00\x05J \x85}\x0C\x18\xA0ac12ba6b7181-BDSZYF0000b1683\xB1\x00\x00\x00\x00\x00\x00\x00\x02, timestamp=1488878422781001, valu
	                                           e=\x81\xA0ac12ba6b7181-BDSZYF0000b1683\xB1\x81
	\x00\x00\x00\x00\x00\x00\x00\x03              column=i:\xFF\xFF\xFF\xFF\xFF\xFE\xC7\x7F\x00\x05J \x85\x17-\xF8ac12ba6b7181-BDSZYF0000b16831, timestamp=1488878415851001, value=
	\x00\x00\x00\x00\x00\x00\x00\x04              column=i:\xFF\xFF\xFF\xFF\xFF\xFF\xFF\x9B\x00\x05J \x85\x12\x01\xC0ac12ba6b7181-BDSZYF0000b16831, timestamp=1488878415512001, value=
	\x00\x00\x00\x00\x00\x00\x00\x04              column=i:\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xCD\x00\x05J \x85\x0C\x97\x08ac12ba6b7181-BDSZYF0000b16831, timestamp=1488878415158001, value=
	\x00\x00\x00\x00\x00\x00\x02\x09              column=e:\x02, timestamp=1488878417188001, value=\x00\x018\x80
	\x00\x00\x00\x00\x00\x00\x02\x09              column=e:\x10\xC0, timestamp=1488878417188001, value=\xA0gi\x1Enam\xE54\x80
	\x00\x00\x00\x00\x00\x00\x02\x09              column=e:\x10\xC2\x80H\x00, timestamp=1488878417188001, value=\xB3\x82\x01\x8E\x00\x87\x80
	\x00\x00\x00\x00\x00\x00\x02\x09              column=e:\x10\xC2\x80L\x00, timestamp=1488878417188001, value=\xB1\x80\x01\x8E\x00\x88\x80
	\x00\x00\x00\x00\x00\x00\x02\x09              column=e:\x10\xC2\x80P\x00, timestamp=1488878417188001, value=\xAF\x80\x01\x8E\x00\x89\x80
	configuration                                 column=s:cache.db-cache, timestamp=1488878296959001, value=\x8F\x01
	configuration                                 column=s:cache.db-cache-clean-wait, timestamp=1488878296944001, value=\x8C\xA8
	configuration                                 column=s:cache.db-cache-size, timestamp=1488878296956001, value=\x94?\xE0\x00\x00\x00\x00\x00\x00
	configuration                                 column=s:cache.db-cache-time, timestamp=1488878296948001, value=\x8D\x80\x00\x00\x00\x00\x02\xBF
	configuration                                 column=s:graph.timestamps, timestamp=1488878297238001, value=\xB6\x81
	configuration                                 column=s:graph.titan-version, timestamp=1488878296968001, value=\x92\xA01.0.\xB0
	configuration                                 column=s:hidden.frozen, timestamp=1488878297242001, value=\x8F\x01
	configuration                                 column=s:index.search.backend, timestamp=1488878296870001, value=\x92\xA0elasticsearc\xE8
	configuration                                 column=s:index.search.elasticsearch.client-only, timestamp=1488878296951001, value=\x8F\x01
	configuration                                 column=s:index.search.hostname, timestamp=1488878296954001, value=\x9E\x82\xA0127.0.0.\xB1
	...   
	
这里数据被编码了，所以看起来不是很直观。不过数据格式是确定的：

左边是rowKey，右边的 COLUMN+CELL 字段格式如下：column=cf(column family):cq(column qualifier), a timestamp that is automatically created by HBase, and the value。

事实上HBase的写操作都是以cell为单位的，这大概就是所谓的按列存储吧。

	put - Puts a cell value at a specified column in a specified row in a particular table.
	get - Fetches the contents of row or a cell.
	delete - Deletes a cell value in a table.
	deleteall - Deletes all the cells in a given row.
	scan - Scans and returns the table data.

然后看一下最重要的几个接口：

1、开始一个事务: 

	@Override
    public StoreTransaction beginTransaction(final BaseTransactionConfig config) throws BackendException {
        return new HBaseTransaction(config);
    }

这里只是简单的创建了一个HBaseTransaction而已。待会我们会看到，HBaseTransaction只是一个签名类而已（没有内容的类）。


2、打开一个Database(这里是 HBaseKeyColumnValueStore ): 

    public KeyColumnValueStore openDatabase(String longName, StoreMetaData.Container metaData) throws BackendException;

所有的Manager的openDatabase做法都大同小异，这里还调用了`ensureColumnFamilyExists(tableName, cfName, cfTTLInSeconds)`确保表结构存在。


3、处理批量更新: `public void mutateMany(Map<String, Map<StaticBuffer, KCVMutation>> mutations, StoreTransaction txh) throws BackendException;`

首先调用 convertToCommands 方法将 mutations 转换成HBase的Put和Delete Command。additions => Put, deletes => Delete：

    /**
     * Convert Titan internal Mutation representation into HBase native commands.
     *
     * @param mutations    Mutations to convert into HBase commands.
     * @param putTimestamp The timestamp to use for Put commands.
     * @param delTimestamp The timestamp to use for Delete commands.
     * @return Commands sorted by key converted from Titan internal representation.
     * @throws com.thinkaurelius.titan.diskstorage.PermanentBackendException
     */
    private Map<StaticBuffer, Pair<Put, Delete>> convertToCommands(Map<String, Map<StaticBuffer, KCVMutation>> mutations,
                                                                   final long putTimestamp,
                                                                   final long delTimestamp) throws PermanentBackendException;


然后将Put和Delete操作添加到List<Row> batch列表中，调用`void com.thinkaurelius.titan.diskstorage.hbase.TableMask.batch(List<Row> writes, Object[] results) throws IOException, InterruptedException`方法进行批量更新。


#### 3、HBaseTransaction

	/**
	 * This class overrides and adds nothing compared with ExpectedValueCheckingTransaction; 
	 * however, it creates a transaction type specific to HBase, which lets us check for user errors 
	 * like passing a Cassandra transaction into a HBase method.
	 */
	public class HBaseTransaction extends AbstractStoreTransaction {

	    public HBaseTransaction(final BaseTransactionConfig config) {
	        super(config);
	    }
	}

正如类注释所说的，这个类并没有做什么事情，只是表明这是一个HBase事务而已。


### Cassandra backend storage

具体参见：[How Titan stores data in Cassandra](https://jaceklaskowski.gitbooks.io/titan-scala/content/titan-titans_internals.html)


Titan底层处理逻辑分析
-------------------

Titan的标准实现入口类是StandardTitanGraph:

	public class StandardTitanGraph extends TitanBlueprintsGraph {

	    static {
	        TraversalStrategies graphStrategies = TraversalStrategies.GlobalCache.getStrategies(Graph.class).clone()
	                .addStrategies(AdjacentVertexFilterOptimizerStrategy.instance(), TitanLocalQueryOptimizerStrategy.instance(), TitanGraphStepStrategy.instance());

	        //Register with cache
	        TraversalStrategies.GlobalCache.registerStrategies(StandardTitanGraph.class, graphStrategies);
	        TraversalStrategies.GlobalCache.registerStrategies(StandardTitanTx.class, graphStrategies);
	    }

	    private GraphDatabaseConfiguration config;
	    private Backend backend;
	    private IDManager idManager;
	    private VertexIDAssigner idAssigner;
	    private TimestampProvider times;

	    //Serializers
	    protected IndexSerializer indexSerializer;
	    protected EdgeSerializer edgeSerializer;
	    protected Serializer serializer;

	    //Caches
	    public SliceQuery vertexExistenceQuery;
	    private RelationQueryCache queryCache;
	    private SchemaCache schemaCache;

	    //Log
	    private ManagementLogger mgmtLogger;

	    //Shutdown hook
	    private volatile ShutdownThread shutdownHook;

	    private volatile boolean isOpen = true;
	    private AtomicLong txCounter;

	    private Set<StandardTitanTx> openTransactions;

	    ...

	}

它基础自TitanBlueprintsGraph，所有的顶点和边的操作都会委托给当前的Transaction处理（TitanBlueprintsTransaction类）：

	public abstract class TitanBlueprintsGraph implements TitanGraph {

	    // ########## TRANSACTION HANDLING ###########################

	    final GraphTransaction tinkerpopTxContainer = new GraphTransaction();

	    private ThreadLocal<TitanBlueprintsTransaction> txs = new ThreadLocal<TitanBlueprintsTransaction>() {

		private TitanBlueprintsTransaction getAutoStartTx() {
	        if (txs == null) throw new IllegalStateException("Graph has been closed");
	        tinkerpopTxContainer.readWrite();

	        TitanBlueprintsTransaction tx = txs.get();
	        Preconditions.checkState(tx!=null,"Invalid read-write behavior configured: " +
	                "Should either open transaction or throw exception.");
	        return tx;
	    }

		// ########## TRANSACTIONAL FORWARDING ###########################

	    @Override
	    public TitanVertex addVertex(Object... keyValues) {
	        return getAutoStartTx().addVertex(keyValues);
	    }

	    @Override
	    public Iterator<Edge> edges(Object... edgeIds) {
	        return getAutoStartTx().edges(edgeIds);
	    }

	    @Override
	    public TitanGraphQuery<? extends TitanGraphQuery> query() {
	        return getAutoStartTx().query();
	    }

	    @Override
	    public TitanIndexQuery indexQuery(String indexName, String query) {
	        return getAutoStartTx().indexQuery(indexName,query);
	    }

	    ...

	    //Schema

	    @Override
	    public PropertyKeyMaker makePropertyKey(String name) {
	        return getAutoStartTx().makePropertyKey(name);
	    }

	    @Override
	    public EdgeLabelMaker makeEdgeLabel(String name) {
	        return getAutoStartTx().makeEdgeLabel(name);
	    }

	    @Override
	    public VertexLabelMaker makeVertexLabel(String name) {
	        return getAutoStartTx().makeVertexLabel(name);
	    }
	    
	    ...

    }


我们看一下TitanBlueprintsTransaction的addVertex方法实现，这其是一个模板方法来的：

 	public TitanVertex addVertex(Object... keyValues) {
 		...

        final TitanVertex vertex = addVertex(null,label);
        com.thinkaurelius.titan.graphdb.util.ElementHelper.attachProperties(vertex, keyValues);
        return vertex;
    }

它回调了子类StandardTitanTx的 addVertex(Long id, VertexLabel vertexLabel) 接口：

    @Override
    public TitanVertex addVertex(Long vertexId, VertexLabel label) {
        
        ... // 各种检查

        StandardVertex vertex = new StandardVertex(this, IDManager.getTemporaryVertexID(IDManager.VertexIDType.NormalVertex, temporaryIds.nextID()), ElementLifeCycle.New);
        if (vertexId != null) {
            vertex.setId(vertexId);
        } else if (config.hasAssignIDsImmediately() || label.isPartitioned()) {
            graph.assignID(vertex,label);
        }
        addProperty(vertex, BaseKey.VertexExists, Boolean.TRUE);
        if (label!=BaseVertexLabel.DEFAULT_VERTEXLABEL) { //Add label
            Preconditions.checkArgument(label instanceof VertexLabelVertex);
            addEdge(vertex, (VertexLabelVertex) label, BaseLabel.VertexLabelEdge);
        }
        vertexCache.add(vertex, vertex.longId());
        return vertex;

    }

逻辑蛮简单的，创建一个 StandardVertex ，设置id，设置系统属性VertexExist=True，设置VertexLabel，然后把vertex加入vertexCache缓存中。

注意到，Titan把VertexLabel作为边存储的；这是因为Titan中属性和边都是关系，addProperty和addEdge最后都会调用connectRelation方法：

	private void connectRelation(InternalRelation r) {
        for (int i = 0; i < r.getLen(); i++) {
            boolean success = r.getVertex(i).addRelation(r);
            if (!success) throw new AssertionError("Could not connect relation: " + r);
        }
        addedRelations.add(r);
        for (int pos = 0; pos < r.getLen(); pos++) vertexCache.add(r.getVertex(pos), r.getVertex(pos).longId());
        if (TypeUtil.hasSimpleInternalVertexKeyIndex(r)) newVertexIndexEntries.add((TitanVertexProperty) r);
    }

其实就是把新增关系放在addedRelations缓存中。

接着回到TitanBlueprintsTransaction的addVertex方法，创建完简单的vertex之后，会调用ElementHelper.attachProperties将其他的属性（非id，非label）添加到该vertex中：

    /**
     * This is essentially an adjusted copy&paste from TinkerPop's ElementHelper class.
     * The reason for copying it is so that we can determine the cardinality of a property key based on
     * Titan's schema which is tied to this particular transaction and not the graph.
     *
     * @param vertex
     * @param propertyKeyValues
     */
    public static void attachProperties(final TitanVertex vertex, final Object... propertyKeyValues) {
        for (int i = 0; i < propertyKeyValues.length; i = i + 2) {
            if (!propertyKeyValues[i].equals(T.id) && !propertyKeyValues[i].equals(T.label))
                vertex.property((String) propertyKeyValues[i], propertyKeyValues[i + 1]);
        }
    }

其实是通过vertex本身的property方法添加属性的，这是定义在AbstractVertex类中：

    public<V> TitanVertexProperty<V> property(final String key, final V value, final Object... keyValues) {
        TitanVertexProperty<V> p = tx().addProperty(it(), tx().getOrCreatePropertyKey(key), value);
        ElementHelper.attachProperties(p,keyValues);
        return p;
    }

第三个keyValues参数其实是为了支持 property of property，一般是null。

看到其实最后也是委托给StandardTitanTx处理：

    public TitanVertexProperty addProperty(VertexProperty.Cardinality cardi, TitanVertex vertex, PropertyKey key, Object value) {
        if (key.cardinality().convert()!=cardi && cardi!=VertexProperty.Cardinality.single)
                throw new SchemaViolationException(String.format("Key is defined for %s cardinality which conflicts with specified: %s",key.cardinality(),cardi));
        verifyWriteAccess(vertex);
        Preconditions.checkArgument(!(key instanceof ImplicitKey),"Cannot create a property of implicit type: %s",key.name());
        vertex = ((InternalVertex) vertex).it();
        Preconditions.checkNotNull(key);
        final Object normalizedValue = verifyAttribute(key, value);
        Cardinality cardinality = key.cardinality();

        //Determine unique indexes
        List<IndexLockTuple> uniqueIndexTuples = new ArrayList<IndexLockTuple>();
        for (CompositeIndexType index : TypeUtil.getUniqueIndexes(key)) {
            IndexSerializer.IndexRecords matches = IndexSerializer.indexMatches(vertex, index, key, normalizedValue);
            for (Object[] match : matches.getRecordValues()) uniqueIndexTuples.add(new IndexLockTuple(index,match));
        }

        TransactionLock uniqueLock = getUniquenessLock(vertex, (InternalRelationType) key, normalizedValue);
        //Add locks for unique indexes
        for (IndexLockTuple lockTuple : uniqueIndexTuples) uniqueLock = new CombinerLock(uniqueLock,getLock(lockTuple),times);
        uniqueLock.lock(LOCK_TIMEOUT);
        try {
            //Delete properties if the cardinality is restricted
            if (cardi==VertexProperty.Cardinality.single || cardi== VertexProperty.Cardinality.set) {
                Consumer<TitanRelation> propertyRemover;
                if (cardi==VertexProperty.Cardinality.single)
                    propertyRemover = p -> p.remove();
                else
                    propertyRemover = p -> { if (((TitanVertexProperty)p).value().equals(normalizedValue)) p.remove(); };

                /* If we are simply overwriting a vertex property, then we don't have to explicitly remove it thereby saving a read operation
                   However, this only applies if
                   1) we don't lock on the property key or consistency checks are disabled and
                   2) there are no indexes for this property key
                   3) the cardinalities match (if we overwrite a set with single, we need to read all other values to delete)
                */
                ConsistencyModifier mod = ((InternalRelationType)key).getConsistencyModifier();
                boolean hasIndex = TypeUtil.hasAnyIndex(key);
                boolean cardiEqual = cardi==cardinality.convert();

                if ( (!config.hasVerifyUniqueness() || ((InternalRelationType)key).getConsistencyModifier()!=ConsistencyModifier.LOCK) &&
                        !TypeUtil.hasAnyIndex(key) && cardi==cardinality.convert()) {
                    //Only delete in-memory so as to not trigger a read from the database which isn't necessary because we will overwrite blindly
                    ((InternalVertex) vertex).getAddedRelations(p -> p.getType().equals(key)).forEach(propertyRemover);
                } else {
                    ((InternalVertex) vertex).query().types(key).properties().forEach(propertyRemover);
                }
            }

            //Check index uniqueness
            if (config.hasVerifyUniqueness()) {
                //Check all unique indexes
                for (IndexLockTuple lockTuple : uniqueIndexTuples) {
                    if (!Iterables.isEmpty(IndexHelper.getQueryResults(lockTuple.getIndex(), lockTuple.getAll(), this)))
                        throw new SchemaViolationException("Adding this property for key [%s] and value [%s] violates a uniqueness constraint [%s]", key.name(), normalizedValue, lockTuple.getIndex());
                }
            }
            StandardVertexProperty prop = new StandardVertexProperty(IDManager.getTemporaryRelationID(temporaryIds.nextID()), key, (InternalVertex) vertex, normalizedValue, ElementLifeCycle.New);
            if (config.hasAssignIDsImmediately()) graph.assignID(prop);
            connectRelation(prop);
            return prop;
        } finally {
            uniqueLock.unlock();
        }

    }
 
 代码挺长也挺复杂的样子，但是其实核心的代码就这最后几行：

    StandardVertexProperty prop = new StandardVertexProperty(IDManager.getTemporaryRelationID(temporaryIds.nextID()), key, (InternalVertex) vertex, normalizedValue, ElementLifeCycle.New);
    if (config.hasAssignIDsImmediately()) graph.assignID(prop);
    connectRelation(prop);

创建一个StandardVertexProperty prop，然后connectRelation(prop)，这个方法我们前面介绍过，因为之前Titan已经调用它连接过label和一个系统属性了。

至此我们简单总结一下：

* 所有的图操作最后都会委托给StandardTitanTx
* StandardTitanTx的addVertex接口，其实就是把创建一个StandardVertex，将其放在vertexCache中
* 属性和边都是关系（Titan中TitanEdge和TitanVertexProperty都是属于TitanRelation），addProperty和addEdge接口其实就是把property和edge放入addedRelations缓存中


那么时候会提交到后端存储呢？答案就是StandardTitanTx commit的时候，只有commit的时候才会调用后端实际存储进行持久化。

让我们看一下StandardTitanTx的commit代码:

    @Override
    public synchronized void commit() {
        
        ... // 一些检查

        try {
            if (hasModifications()) {
                graph.commit(addedRelations.getAll(), deletedRelations.values(), this);
            } else {
                txHandle.commit();
            }
            success = true;
        } catch (Exception e) {
            try {
                txHandle.rollback();
            } catch (BackendException e1) {
                throw new TitanException("Could not rollback after a failed commit", e);
            }
            throw new TitanException("Could not commit transaction due to exception during persistence", e);
        } finally {
            releaseTransaction();
            if (null != config.getGroupName() && !success) {
                MetricManager.INSTANCE.getCounter(config.getGroupName(), "tx", "commit.exceptions").inc();
            }
        }
    }

其实非常简单，基本就是委托给了StandardTitanGraph.commit()方法，这是一个想当复杂的方法，也是非常核心的方法：

    public void commit(final Collection<InternalRelation> addedRelations,
                     final Collection<InternalRelation> deletedRelations, final StandardTitanTx tx) {
        if (addedRelations.isEmpty() && deletedRelations.isEmpty()) return;
        //1. Finalize transaction
        log.debug("Saving transaction. Added {}, removed {}", addedRelations.size(), deletedRelations.size());
        if (!tx.getConfiguration().hasCommitTime()) tx.getConfiguration().setCommitTime(times.getTime());
        final Instant txTimestamp = tx.getConfiguration().getCommitTime();
        final long transactionId = txCounter.incrementAndGet();

        //2. Assign TitanVertex IDs
        if (!tx.getConfiguration().hasAssignIDsImmediately())
            idAssigner.assignIDs(addedRelations);

        //3. Commit
        BackendTransaction mutator = tx.getTxHandle();
        final boolean acquireLocks = tx.getConfiguration().hasAcquireLocks();
        final boolean hasTxIsolation = backend.getStoreFeatures().hasTxIsolation();
        final boolean logTransaction = config.hasLogTransactions() && !tx.getConfiguration().hasEnabledBatchLoading();
        final KCVSLog txLog = logTransaction?backend.getSystemTxLog():null;
        final TransactionLogHeader txLogHeader = new TransactionLogHeader(transactionId,txTimestamp, times);
        ModificationSummary commitSummary;

        try {
            //3.1 Log transaction (write-ahead log) if enabled
            if (logTransaction) {
                //[FAILURE] Inability to log transaction fails the transaction by escalation since it's likely due to unavailability of primary
                //storage backend.
                txLog.add(txLogHeader.serializeModifications(serializer, LogTxStatus.PRECOMMIT, tx, addedRelations, deletedRelations),txLogHeader.getLogKey());
            }

            //3.2 Commit schema elements and their associated relations in a separate transaction if backend does not support
            //    transactional isolation
            boolean hasSchemaElements = !Iterables.isEmpty(Iterables.filter(deletedRelations,SCHEMA_FILTER))
                    || !Iterables.isEmpty(Iterables.filter(addedRelations,SCHEMA_FILTER));
            Preconditions.checkArgument(!hasSchemaElements || (!tx.getConfiguration().hasEnabledBatchLoading() && acquireLocks),
                    "Attempting to create schema elements in inconsistent state");

            if (hasSchemaElements && !hasTxIsolation) {
                /*
                 * On storage without transactional isolation, create separate
                 * backend transaction for schema aspects to make sure that
                 * those are persisted prior to and independently of other
                 * mutations in the tx. If the storage supports transactional
                 * isolation, then don't create a separate tx.
                 */
                final BackendTransaction schemaMutator = openBackendTransaction(tx);

                try {
                    //[FAILURE] If the preparation throws an exception abort directly - nothing persisted since batch-loading cannot be enabled for schema elements
                    commitSummary = prepareCommit(addedRelations,deletedRelations, SCHEMA_FILTER, schemaMutator, tx, acquireLocks);
                    assert commitSummary.hasModifications && !commitSummary.has2iModifications;
                } catch (Throwable e) {
                    //Roll back schema tx and escalate exception
                    schemaMutator.rollback();
                    throw e;
                }

                try {
                    schemaMutator.commit();
                } catch (Throwable e) {
                    //[FAILURE] Primary persistence failed => abort and escalate exception, nothing should have been persisted
                    log.error("Could not commit transaction ["+transactionId+"] due to storage exception in system-commit",e);
                    throw e;
                }
            }

            //[FAILURE] Exceptions during preparation here cause the entire transaction to fail on transactional systems
            //or just the non-system part on others. Nothing has been persisted unless batch-loading
            commitSummary = prepareCommit(addedRelations,deletedRelations, hasTxIsolation? NO_FILTER : NO_SCHEMA_FILTER, mutator, tx, acquireLocks);
            if (commitSummary.hasModifications) {
                String logTxIdentifier = tx.getConfiguration().getLogIdentifier();
                boolean hasSecondaryPersistence = logTxIdentifier!=null || commitSummary.has2iModifications;

                //1. Commit storage - failures lead to immediate abort

                //1a. Add success message to tx log which will be committed atomically with all transactional changes so that we can recover secondary failures
                //    This should not throw an exception since the mutations are just cached. If it does, it will be escalated since its critical
                if (logTransaction) {
                    txLog.add(txLogHeader.serializePrimary(serializer,
                                        hasSecondaryPersistence?LogTxStatus.PRIMARY_SUCCESS:LogTxStatus.COMPLETE_SUCCESS),
                            txLogHeader.getLogKey(),mutator.getTxLogPersistor());
                }

                try {
                    mutator.commitStorage();
                } catch (Throwable e) {
                    //[FAILURE] If primary storage persistence fails abort directly (only schema could have been persisted)
                    log.error("Could not commit transaction ["+transactionId+"] due to storage exception in commit",e);
                    throw e;
                }

                if (hasSecondaryPersistence) {
                    LogTxStatus status = LogTxStatus.SECONDARY_SUCCESS;
                    Map<String,Throwable> indexFailures = ImmutableMap.of();
                    boolean userlogSuccess = true;

                    try {
                        //2. Commit indexes - [FAILURE] all exceptions are collected and logged but nothing is aborted
                        indexFailures = mutator.commitIndexes();
                        if (!indexFailures.isEmpty()) {
                            status = LogTxStatus.SECONDARY_FAILURE;
                            for (Map.Entry<String,Throwable> entry : indexFailures.entrySet()) {
                                log.error("Error while commiting index mutations for transaction ["+transactionId+"] on index: " +entry.getKey(),entry.getValue());
                            }
                        }
                        //3. Log transaction if configured - [FAILURE] is recorded but does not cause exception
                        if (logTxIdentifier!=null) {
                            try {
                                userlogSuccess = false;
                                final Log userLog = backend.getUserLog(logTxIdentifier);
                                Future<Message> env = userLog.add(txLogHeader.serializeModifications(serializer, LogTxStatus.USER_LOG, tx, addedRelations, deletedRelations));
                                if (env.isDone()) {
                                    try {
                                        env.get();
                                    } catch (ExecutionException ex) {
                                        throw ex.getCause();
                                    }
                                }
                                userlogSuccess=true;
                            } catch (Throwable e) {
                                status = LogTxStatus.SECONDARY_FAILURE;
                                log.error("Could not user-log committed transaction ["+transactionId+"] to " + logTxIdentifier, e);
                            }
                        }
                    } finally {
                        if (logTransaction) {
                            //[FAILURE] An exception here will be logged and not escalated; tx considered success and
                            // needs to be cleaned up later
                            try {
                                txLog.add(txLogHeader.serializeSecondary(serializer,status,indexFailures,userlogSuccess),txLogHeader.getLogKey());
                            } catch (Throwable e) {
                                log.error("Could not tx-log secondary persistence status on transaction ["+transactionId+"]",e);
                            }
                        }
                    }
                } else {
                    //This just closes the transaction since there are no modifications
                    mutator.commitIndexes();
                }
            } else { //Just commit everything at once
                //[FAILURE] This case only happens when there are no non-system mutations in which case all changes
                //are already flushed. Hence, an exception here is unlikely and should abort
                mutator.commit();
            }
        } catch (Throwable e) {
            log.error("Could not commit transaction ["+transactionId+"] due to exception",e);
            try {
                //Clean up any left-over transaction handles
                mutator.rollback();
            } catch (Throwable e2) {
                log.error("Could not roll-back transaction ["+transactionId+"] after failure due to exception",e2);
            }
            if (e instanceof RuntimeException) throw (RuntimeException)e;
            else throw new TitanException("Unexpected exception",e);
        }
    }

不过它的注释相对也比较详细，大概是几个步骤：

1. Finalize transaction
	1.1 设置事务的commitTime
	1.2 得到当前的事务id
2. Assign TitanVertex IDs：分配VertexID
3. Commit
	3.1 Log transaction (write-ahead log) if enabled
	3.2 Commit schema elements and their associated relations in a separate transaction if backend does not support transactional isolation
		3.2.1 prepareCommit for schema
		3.2.2 schemaMutator.commit();
	3.3 Commit data 
		3.3.1 prepareCommit for data
		3.3.2 mutator.commitStorage();		
	3.4 Commit indexes
		3.4.1 mutator.commitIndexes();	
	3.5 Log transaction if configured 


注意到每次提交之前都会先调用prepareCommit方法，这个方法也挺长的，不过同样有比较详细的注释：

    public ModificationSummary prepareCommit(final Collection<InternalRelation> addedRelations,
                                     final Collection<InternalRelation> deletedRelations,
                                     final Predicate<InternalRelation> filter,
                                     final BackendTransaction mutator, final StandardTitanTx tx,
                                     final boolean acquireLocks) throws BackendException {

        ListMultimap<Long, InternalRelation> mutations = ArrayListMultimap.create();
        ListMultimap<InternalVertex, InternalRelation> mutatedProperties = ArrayListMultimap.create();
        List<IndexSerializer.IndexUpdate> indexUpdates = Lists.newArrayList();
        //1) Collect deleted edges and their index updates and acquire edge locks
        for (InternalRelation del : Iterables.filter(deletedRelations,filter)) {
            Preconditions.checkArgument(del.isRemoved());
            for (int pos = 0; pos < del.getLen(); pos++) {
                InternalVertex vertex = del.getVertex(pos);
                if (pos == 0 || !del.isLoop()) {
                    if (del.isProperty()) mutatedProperties.put(vertex,del);
                    mutations.put(vertex.longId(), del);
                }
                if (acquireLock(del,pos,acquireLocks)) {
                    Entry entry = edgeSerializer.writeRelation(del, pos, tx);
                    mutator.acquireEdgeLock(idManager.getKey(vertex.longId()), entry);
                }
            }
            indexUpdates.addAll(indexSerializer.getIndexUpdates(del));
        }

        //2) Collect added edges and their index updates and acquire edge locks
        for (InternalRelation add : Iterables.filter(addedRelations,filter)) {
            Preconditions.checkArgument(add.isNew());

            for (int pos = 0; pos < add.getLen(); pos++) {
                InternalVertex vertex = add.getVertex(pos);
                if (pos == 0 || !add.isLoop()) {
                    if (add.isProperty()) mutatedProperties.put(vertex,add);
                    mutations.put(vertex.longId(), add);
                }
                if (!vertex.isNew() && acquireLock(add,pos,acquireLocks)) {
                    Entry entry = edgeSerializer.writeRelation(add, pos, tx);
                    mutator.acquireEdgeLock(idManager.getKey(vertex.longId()), entry.getColumn());
                }
            }
            indexUpdates.addAll(indexSerializer.getIndexUpdates(add));
        }

        //3) Collect all index update for vertices
        for (InternalVertex v : mutatedProperties.keySet()) {
            indexUpdates.addAll(indexSerializer.getIndexUpdates(v,mutatedProperties.get(v)));
        }
        //4) Acquire index locks (deletions first)
        for (IndexSerializer.IndexUpdate update : indexUpdates) {
            if (!update.isCompositeIndex() || !update.isDeletion()) continue;
            CompositeIndexType iIndex = (CompositeIndexType) update.getIndex();
            if (acquireLock(iIndex,acquireLocks)) {
                mutator.acquireIndexLock((StaticBuffer)update.getKey(), (Entry)update.getEntry());
            }
        }
        for (IndexSerializer.IndexUpdate update : indexUpdates) {
            if (!update.isCompositeIndex() || !update.isAddition()) continue;
            CompositeIndexType iIndex = (CompositeIndexType) update.getIndex();
            if (acquireLock(iIndex,acquireLocks)) {
                mutator.acquireIndexLock((StaticBuffer)update.getKey(), ((Entry)update.getEntry()).getColumn());
            }
        }

        //5) Add relation mutations
        for (Long vertexid : mutations.keySet()) {
            Preconditions.checkArgument(vertexid > 0, "Vertex has no id: %s", vertexid);
            List<InternalRelation> edges = mutations.get(vertexid);
            List<Entry> additions = new ArrayList<Entry>(edges.size());
            List<Entry> deletions = new ArrayList<Entry>(Math.max(10, edges.size() / 10));
            for (InternalRelation edge : edges) {
                InternalRelationType baseType = (InternalRelationType) edge.getType();
                assert baseType.getBaseType()==null;

                for (InternalRelationType type : baseType.getRelationIndexes()) {
                    if (type.getStatus()== SchemaStatus.DISABLED) continue;
                    for (int pos = 0; pos < edge.getArity(); pos++) {
                        if (!type.isUnidirected(Direction.BOTH) && !type.isUnidirected(EdgeDirection.fromPosition(pos)))
                            continue; //Directionality is not covered
                        if (edge.getVertex(pos).longId()==vertexid) {
                            StaticArrayEntry entry = edgeSerializer.writeRelation(edge, type, pos, tx);
                            if (edge.isRemoved()) {
                                deletions.add(entry);
                            } else {
                                Preconditions.checkArgument(edge.isNew());
                                int ttl = getTTL(edge);
                                if (ttl > 0) {
                                    entry.setMetaData(EntryMetaData.TTL, ttl);
                                }
                                additions.add(entry);
                            }
                        }
                    }
                }
            }

            StaticBuffer vertexKey = idManager.getKey(vertexid);
            mutator.mutateEdges(vertexKey, additions, deletions);
        }

        //6) Add index updates
        boolean has2iMods = false;
        for (IndexSerializer.IndexUpdate indexUpdate : indexUpdates) {
            assert indexUpdate.isAddition() || indexUpdate.isDeletion();
            if (indexUpdate.isCompositeIndex()) {
                IndexSerializer.IndexUpdate<StaticBuffer,Entry> update = indexUpdate;
                if (update.isAddition())
                    mutator.mutateIndex(update.getKey(), Lists.newArrayList(update.getEntry()), KCVSCache.NO_DELETIONS);
                else
                    mutator.mutateIndex(update.getKey(), KeyColumnValueStore.NO_ADDITIONS, Lists.newArrayList(update.getEntry()));
            } else {
                IndexSerializer.IndexUpdate<String,IndexEntry> update = indexUpdate;
                has2iMods = true;
                IndexTransaction itx = mutator.getIndexTransaction(update.getIndex().getBackingIndexName());
                String indexStore = ((MixedIndexType)update.getIndex()).getStoreName();
                if (update.isAddition())
                    itx.add(indexStore, update.getKey(), update.getEntry(), update.getElement().isNew());
                else
                    itx.delete(indexStore,update.getKey(),update.getEntry().field,update.getEntry().value,update.getElement().isRemoved());
            }
        }
        return new ModificationSummary(!mutations.isEmpty(),has2iMods);
    }

首先介绍三个数据结构：

1. ListMultimap<Long, InternalRelation> mutations: 每个顶点id对应的变更的 属性(StandardVertexProperty)/边(StandardEdge) 列表
2. ListMultimap<InternalVertex, InternalRelation> mutatedProperties: 每个vertex对应的变更属性(StandardVertexProperty)列表
3. List<IndexSerializer.IndexUpdate> indexUpdates: 检查 顶点属性(StandardVertexProperty)/边(StandardEdge)的每个properties是否有索引变更。
	* IndexUpdate<K,E>:
		* IndexType index: 通过TitanManagement buildIndex方法指定的索引信息，如`{STATUS=ENABLED, ELEMENT_CATEGORY=EDGE, INDEX_CARDINALITY=LIST, INTERNAL_INDEX=false, BACKING_INDEX=search, INDEXSTORE_NAME=edges}`
			* CompositeIndex
			* MixedIndex
        * Type mutationType
        	* ADD
        	* DELETE
        * K key: TitanElement.id()，如EdgeId
        * E entry: IndexEntry，要构建索引的Property
        	* field: property key
        	* value: property value
        * TitanElement element: 索引相关联的元素（顶点属性或者边）

用Titan的Gods图谱单测断点调试：

	public static void load(final TitanGraph graph, String mixedIndexName, boolean uniqueNameCompositeIndex) {

        //Create Schema
        TitanManagement mgmt = graph.openManagement();
        final PropertyKey name = mgmt.makePropertyKey("name").dataType(String.class).make();
        TitanManagement.IndexBuilder nameIndexBuilder = mgmt.buildIndex("name", Vertex.class).addKey(name);
        if (uniqueNameCompositeIndex)
            nameIndexBuilder.unique();
        TitanGraphIndex namei = nameIndexBuilder.buildCompositeIndex();
        mgmt.setConsistency(namei, ConsistencyModifier.LOCK);
        final PropertyKey age = mgmt.makePropertyKey("age").dataType(Integer.class).make();
        if (null != mixedIndexName)
            mgmt.buildIndex("vertices", Vertex.class).addKey(age).buildMixedIndex(mixedIndexName);

        final PropertyKey time = mgmt.makePropertyKey("time").dataType(Integer.class).make();
        final PropertyKey reason = mgmt.makePropertyKey("reason").dataType(String.class).make();
        final PropertyKey place = mgmt.makePropertyKey("place").dataType(Geoshape.class).make();
        if (null != mixedIndexName)
            mgmt.buildIndex("edges", Edge.class).addKey(reason).addKey(place).buildMixedIndex(mixedIndexName);

        mgmt.makeEdgeLabel("father").multiplicity(Multiplicity.MANY2ONE).make();
        mgmt.makeEdgeLabel("mother").multiplicity(Multiplicity.MANY2ONE).make();
        EdgeLabel battled = mgmt.makeEdgeLabel("battled").signature(time).make();
        mgmt.buildEdgeIndex(battled, "battlesByTime", Direction.BOTH, Order.decr, time);
        mgmt.makeEdgeLabel("lives").signature(reason).make();
        mgmt.makeEdgeLabel("pet").make();
        mgmt.makeEdgeLabel("brother").make();

        mgmt.makeVertexLabel("titan").make();
        mgmt.makeVertexLabel("location").make();
        mgmt.makeVertexLabel("god").make();
        mgmt.makeVertexLabel("demigod").make();
        mgmt.makeVertexLabel("human").make();
        mgmt.makeVertexLabel("monster").make();

        mgmt.commit();

        TitanTransaction tx = graph.newTransaction();
        // vertices

        Vertex saturn = tx.addVertex(T.label, "titan", "name", "saturn", "age", 10000);

        Vertex sky = tx.addVertex(T.label, "location", "name", "sky");

        Vertex sea = tx.addVertex(T.label, "location", "name", "sea");

        Vertex jupiter = tx.addVertex(T.label, "god", "name", "jupiter", "age", 5000);

        Vertex neptune = tx.addVertex(T.label, "god", "name", "neptune", "age", 4500);

        Vertex hercules = tx.addVertex(T.label, "demigod", "name", "hercules", "age", 30);

        Vertex alcmene = tx.addVertex(T.label, "human", "name", "alcmene", "age", 45);

        Vertex pluto = tx.addVertex(T.label, "god", "name", "pluto", "age", 4000);

        Vertex nemean = tx.addVertex(T.label, "monster", "name", "nemean");

        Vertex hydra = tx.addVertex(T.label, "monster", "name", "hydra");

        Vertex cerberus = tx.addVertex(T.label, "monster", "name", "cerberus");

        Vertex tartarus = tx.addVertex(T.label, "location", "name", "tartarus");

        // edges

        jupiter.addEdge("father", saturn);
        jupiter.addEdge("lives", sky, "reason", "loves fresh breezes");
        jupiter.addEdge("brother", neptune);
        jupiter.addEdge("brother", pluto);

        neptune.addEdge("lives", sea).property("reason", "loves waves");
        neptune.addEdge("brother", jupiter);
        neptune.addEdge("brother", pluto);

        hercules.addEdge("father", jupiter);
        hercules.addEdge("mother", alcmene);
        hercules.addEdge("battled", nemean, "time", 1, "place", Geoshape.point(38.1f, 23.7f));
        hercules.addEdge("battled", hydra, "time", 2, "place", Geoshape.point(37.7f, 23.9f));
        hercules.addEdge("battled", cerberus, "time", 12, "place", Geoshape.point(39f, 22f));

        pluto.addEdge("brother", jupiter);
        pluto.addEdge("brother", neptune);
        pluto.addEdge("lives", tartarus, "reason", "no fear of death");
        pluto.addEdge("pet", cerberus);

        cerberus.addEdge("lives", tartarus);

        // commit the transaction to disk
        tx.commit();
    }

会在ES创建如下mapping:

	2017-04-18 17:41:40,318 (elasticsearch[Hindsight Lad][clusterService#updateTask][T#1]) [ INFO] (MetaDataMappingService.java:566) - [Hindsight Lad] [titan] create_mapping [vertices]
	2017-04-18 17:41:40,372 (elasticsearch[Hindsight Lad][clusterService#updateTask][T#1]) [ INFO] (MetaDataMappingService.java:566) - [Hindsight Lad] [titan] create_mapping [edges]
	2017-04-18 17:41:40,385 (elasticsearch[Hindsight Lad][clusterService#updateTask][T#1]) [ INFO] (MetaDataMappingService.java:558) - [Hindsight Lad] [titan] update_mapping [edges]

可以看到这三个数据结构的样子：

mutations: 

	{4240=[vp[~T$VertexExists->true], e[sy-39s-51-6q5][4240-~T$vertexlabel->8717], vp[name->saturn], vp[age->10000]], 8336=[vp[~T$VertexExists->true], e[2du-6fk-51-74d][8336-~T$vertexlabel->9229], vp[name->sky]], 4336=[vp[~T$VertexExists->true], e[ta-3cg-51-7il][4336-~T$vertexlabel->9741], vp[name->jupiter], vp[age->5000]], 4144=[vp[~T$VertexExists->true], e[sm-374-51-7il][4144-~T$vertexlabel->9741], vp[name->neptune], vp[age->4500]], 8432=[vp[~T$VertexExists->true], e[2e6-6i8-51-7il][8432-~T$vertexlabel->9741], vp[name->pluto], vp[age->4000]], 12528=[vp[~T$VertexExists->true], e[3z2-9o0-51-8p9][12528-~T$vertexlabel->11277], vp[name->hydra]], 4312=[vp[~T$VertexExists->true], e[t7-3bs-51-74d][4312-~T$vertexlabel->9229], vp[name->sea]], 4104=[vp[~T$VertexExists->true], e[sh-360-51-7wt][4104-~T$vertexlabel->10253], vp[name->hercules], vp[age->30]], 8408=[vp[~T$VertexExists->true], e[1zv-6hk-51-8b1][8408-~T$vertexlabel->10765], vp[name->alcmene], vp[age->45]], 8200=[vp[~T$VertexExists->true], e[2dd-6bs-51-8p9][8200-~T$vertexlabel->11277], vp[name->nemean]]}

细心的读者可能注意到Titan的顶点数据是顶点属性和边一起存放的，这就是所谓的 Adjacency-free list。具体可以参考文档 [Titan Documentation > Titan Internals > Titan Data Model](http://s3.thinkaurelius.com/docs/titan/1.0.0/data-model.html):

> Titan stores graphs in adjacency list format which means that a graph is stored as a collection of vertices with their adjacency list. The adjacency list of a vertex contains all of the vertex’s incident edges (and properties).

> Titan stores each adjacency list as a row in the underlying storage backend. The (64 bit) vertex id (which Titan uniquely assigns to every vertex) is the key which points to the row containing the vertex’s adjacency list. Each edge and property is stored as an individual cell in the row which allows for efficient insertions and deletions. 

另外，从另一方便来说，Titan确实也没有区分边和属性，统一实现InternalRelation接口。

mutatedProperties:

	{v[4240]=[vp[~T$VertexExists->true], vp[name->saturn], vp[age->10000]], v[8336]=[vp[~T$VertexExists->true], vp[name->sky]], v[4336]=[vp[~T$VertexExists->true], vp[name->jupiter], vp[age->5000]], v[4144]=[vp[~T$VertexExists->true], vp[name->neptune], vp[age->4500]], v[8432]=[vp[~T$VertexExists->true], vp[name->pluto], vp[age->4000]], v[12528]=[vp[~T$VertexExists->true], vp[name->hydra]], v[4312]=[vp[~T$VertexExists->true], vp[name->sea]], v[4104]=[vp[~T$VertexExists->true], vp[name->hercules], vp[age->30]], v[8408]=[vp[~T$VertexExists->true], vp[name->alcmene], vp[age->45]], v[8200]=[vp[~T$VertexExists->true], vp[name->nemean]]}

indexUpdates:

	element=e[3kc-6e8-b2t-38g][8288-lives->4192], index=edges, indexType=MixedIndexTypeWrapper, key=3kc-6e8-b2t-38g, entry.field=reason, entry.value=loves fresh breezes
	element=e[3kh-39k-b2t-36o][4232-lives->4128], index=edges, indexType=MixedIndexTypeWrapper, key=3kh-39k-b2t-36o, entry.field=reason, entry.value=loves waves
	element=e[3yx-3bc-9hx-368][4296-battled->4112], index=edges, indexType=MixedIndexTypeWrapper, key=3yx-3bc-9hx-368, entry.field=place, entry.value=point[38.1,23.7]
	element=e[4d5-3bc-9hx-9n4][4296-battled->12496], index=edges, indexType=MixedIndexTypeWrapper, key=4d5-3bc-9hx-9n4, entry.field=place, entry.value=point[37.7,23.9]
	element=e[4rd-3bc-9hx-6h4][4296-battled->8392], index=edges, indexType=MixedIndexTypeWrapper, key=4rd-3bc-9hx-6h4, entry.field=place, entry.value=point[39.0,22.0]
	element=e[5jl-6fc-b2t-6c0][8328-lives->8208], index=edges, indexType=MixedIndexTypeWrapper, key=5jl-6fc-b2t-6c0, entry.field=reason, entry.value=no fear of death
	element=v[4304], index=name, indexType=CompositeIndexTypeWrapper, key=  4-137-160-115- 97-116-117-114-238, entry=  0-> 33-208
	element=v[4304], index=vertices, indexType=MixedIndexTypeWrapper, key=3bk, entry.field=age, entry.value=10000
	element=v[4192], index=name, indexType=CompositeIndexTypeWrapper, key=  4-137-160-115-107-249, entry=  0-> 32-224
	element=v[4128], index=name, indexType=CompositeIndexTypeWrapper, key=  4-137-160-115-101-225, entry=  0-> 32-160
	element=v[8288], index=name, indexType=CompositeIndexTypeWrapper, key=  4-137-160-106-117-112-105-116-101-242, entry=  0-> 64-224
	element=v[8288], index=vertices, indexType=MixedIndexTypeWrapper, key=6e8, entry.field=age, entry.value=5000
	element=v[8400], index=name, indexType=CompositeIndexTypeWrapper, key=  4-137-160- 97-108- 99-109-101-110-229, entry=  0-> 65-208
	element=v[8400], index=vertices, indexType=MixedIndexTypeWrapper, key=6hc, entry.field=age, entry.value=45
	element=v[4112], index=name, indexType=CompositeIndexTypeWrapper, key=  4-137-160-110-101-109-101- 97-238, entry=  0-> 32-144
	element=v[12496], index=name, indexType=CompositeIndexTypeWrapper, key=  4-137-160-104-121-100-114-225, entry=  0-> 97-208
	element=v[8208], index=name, indexType=CompositeIndexTypeWrapper, key=  4-137-160-116- 97-114-116- 97-114-117-243, entry=  0-> 64-144
	element=v[4232], index=vertices, indexType=MixedIndexTypeWrapper, key=39k, entry.field=age, entry.value=4500
	element=v[4232], index=name, indexType=CompositeIndexTypeWrapper, key=  4-137-160-110-101-112-116-117-110-229, entry=  0-> 33-136
	element=v[4296], index=name, indexType=CompositeIndexTypeWrapper, key=  4-137-160-104-101-114- 99-117-108-101-243, entry=  0-> 33-200
	element=v[4296], index=vertices, indexType=MixedIndexTypeWrapper, key=3bc, entry.field=age, entry.value=30
	element=v[8328], index=name, indexType=CompositeIndexTypeWrapper, key=  4-137-160-112-108-117-116-239, entry=  0-> 65-136
	element=v[8328], index=vertices, indexType=MixedIndexTypeWrapper, key=6fc, entry.field=age, entry.value=4000
	element=v[8392], index=name, indexType=CompositeIndexTypeWrapper, key=  4-137-160- 99-101-114- 98-101-114-117-243, entry=  0-> 65-200

其中index=edges表示我们之前创建了一个edges的index：

    mgmt.buildIndex("edges", Edge.class).addKey(reason).addKey(place).buildMixedIndex(mixedIndexName);

同理其他的index(name, vertices)：

    TitanManagement.IndexBuilder nameIndexBuilder = mgmt.buildIndex("name", Vertex.class).addKey(name);
    mgmt.buildIndex("vertices", Vertex.class).addKey(age).buildMixedIndex(mixedIndexName);

注意到不同的索引类型(Composited和Mixed)，entry和key的类型也不同。所以打印出来的内容也是各不相同的。

然后看一下流程：

1. Collect deleted edges and their index updates and acquire edge locks
	1. Collect deleted edges to mutatedProperties(`ListMultimap<InternalVertex, InternalRelation>`) and mutations(`ListMultimap<Long, InternalRelation>`)
	2. collect their index to indexUpdates(`List<IndexSerializer.IndexUpdate>`)
	3. acquire edge locks: `mutator.acquireEdgeLock(idManager.getKey(vertex.longId()), entry);`
2. Collect added edges and their index updates and acquire edge locks
	1. Collect added edges to mutatedProperties(`ListMultimap<InternalVertex, InternalRelation>`) and mutations(`ListMultimap<Long, InternalRelation>`)
	2. collect their index to indexUpdates(`List<IndexSerializer.IndexUpdate>`)：这里只是收集边的indexUpdates
	3. acquire edge locks: `mutator.acquireEdgeLock(idManager.getKey(vertex.longId()), entry.getColumn());`
3. Collect all index update for vertices: 收集顶点的indexUpdates
	1. `indexUpdates.addAll(indexSerializer.getIndexUpdates(v,mutatedProperties.get(v)));`
4. Acquire index locks (deletions first): 
	1. mutator.acquireIndexLock((StaticBuffer)update.getKey(), (Entry)update.getEntry());
	2. mutator.acquireIndexLock((StaticBuffer)update.getKey(), ((Entry)update.getEntry()).getColumn());
5. Add relation mutations：
	1. 将 mutations 的修改转换成 additions 和 deletions (`List<Entry>`)
	2. 调用底层mutator进行变更：`mutator.mutateEdges(vertexKey, additions, deletions);` // 详见下面说明
6. Add index updates
	1. 如果是CompositeIndex(元素类型为`IndexSerializer.IndexUpdate<StaticBuffer,Entry>`)，调用底层mutator进行索引变更：
		1. isAddition: `mutator.mutateIndex(update.getKey(), Lists.newArrayList(update.getEntry()), KCVSCache.NO_DELETIONS);`
		2. else: `mutator.mutateIndex(update.getKey(), KeyColumnValueStore.NO_ADDITIONS, Lists.newArrayList(update.getEntry()));`
	2. 否则(是MixedIndex，元素类型为`IndexSerializer.IndexUpdate<String,IndexEntry>`)
		1. has2iMods = true;
		2. 得到IndexTransaction itx和indexStore名称，调用itx进行索引变更：
			1. isAddition: `itx.add(indexStore, update.getKey(), update.getEntry(), update.getElement().isNew());`
			2. else: `itx.delete(indexStore,update.getKey(),update.getEntry().field,update.getEntry().value,update.getElement().isRemoved());`

根据addedRelations和deletedRelations判断是否有数据更新(hasModifications)和索引更新(has2iModifications)。

prepareCommit方法还会调用BackendTransaction的mutateEdges方法应用修改，而后者会调用edgeStore的mutateEntries方法，而edgeStore则最后委托给StoreTransaction（CacheTransaction）的mutate方法，这个方法会将修改缓存起来。

同样修改的索引，prepareCommit方法也会调用BackendTransaction的mutateIndex方法，最后缓存在StoreTransaction（CacheTransaction）的mutations变量中。

调用完prepareCommit方法后，commit方法然后调用BackendTransaction的commitStorage()方法，提交数据存储。这个方法会调用storeTx.commit()

后者就是调用了CacheTransaction的commit()方法:

    @Override
    public void commit() throws BackendException {
        flushInternal();
        tx.commit();
    }

flushInternal()方法正如其名，就是把mutations变量中缓存的更新持久化persist

	private int persist(final Map<String, Map<StaticBuffer, KCVMutation>> subMutations) {
        BackendOperation.execute(new Callable<Boolean>() {
            @Override
            public Boolean call() throws Exception {
                manager.mutateMany(subMutations, tx);
                return true;
            }

            @Override
            public String toString() {
                return "CacheMutation";
            }
        }, maxWriteTime);
        subMutations.clear();
        return 0;
    }

这里看到persist方法其实就是调用manager的mutateMany接口，这个就是我们要实现的接口之一。

tx.commit(); 也会调用到我们实现StoreTransaction的commit方法，如CassandraTransaction，这就回调到具体的后端存储的实现了。


分析完写入过程，我们来看一下查询过程。

在StandardTitanTx中两个QueryExecutor专门处理查询：

	1. QueryExecutor<VertexCentricQuery, TitanRelation, SliceQuery> edgeProcessor ：处理边和属性查询
	2. QueryExecutor<GraphCentricQuery, TitanElement, JointIndexQuery> elementProcessor：处理顶点查询

查询的关键在Query类，因为查询的条件比较复杂，而且后端存储支持的查询也不同，所以query类多而复杂：

![图片](http://bos.nj.bpc.baidu.com/v1/agroup/aed88a406a8d0ad7f795385736f5bff22de80b66)

简单来说所有的Query基本都是通过 Condition 类来表达查询:

	/**
	 * A logical condition which evaluates against a provided element to true or false.
	 * </p>
	 * A condition can be nested to form complex logical expressions with AND, OR and NOT.
	 * A condition is either a literal, a negation of a condition, or a logical combination of conditions (AND, OR).
	 * If a condition has sub-conditions we consider those to be children.
	 *
	 * @author Matthias Broecheler (me@matthiasb.com)
	 */
	public interface Condition<E extends TitanElement> {
	
	    public enum Type { AND, OR, NOT, LITERAL}
	
	    public Type getType();
	
	    public Iterable<Condition<E>> getChildren();
	
	    public boolean hasChildren();
	
	    public int numChildren();
	
	    public boolean evaluate(E element);
	
	    public int hashCode();
	
	    public boolean equals(Object other);
	
	    public String toString();
	
	}

Condition类也拥有一个庞大的家族：

![图片](http://bos.nj.bpc.baidu.com/v1/agroup/78802f3601bd63cf8addba27ae9605654dce3453)

其中最重要的是PredicateCondition：

	public class PredicateCondition<K, E extends TitanElement> extends Literal<E> {

	    private final K key;
	    private final TitanPredicate predicate;
	    private final Object value;

	    ...
	}

它引入了另一个重要的类——TitanPredicate及其家族：

	* interface BiPredicate<T, U>
		* enum Compare
			* eq
			* neq
			* gt
			* gte
			* lt
			* lte
		* enum Contains
			* within
			* without
		* class AndBiPredicate
		* class OrBiPredicate
		* interface TitanPredicate
			* enum Cmp:
				* EQUAL
				* NOT_EQUAL
				* LESS_THAN
				* LESS_THAN_EQUAL
				* GREATER_THAN
				* GREATER_THAN_EQUAL
			* enum Contain
				* IN
				* NOT_IN
			* enum Geo
				* INTERSECT
				* DISJOINT
				* WITHIN
			* Text
				* CONTAINS
				* CONTAINS_PREFIX
				* CONTAINS_REGEX
				* PREFIX
				* REGEX

最后不管是edge查询还是vertex查询都会调用 BackendTransaction 的查询方法：  

1. EdgeQuery => public EntryList edgeStoreQuery(final KeySliceQuery query) ;
2. VertexQuery => public EntryList indexQuery(final KeySliceQuery query) ;

吐槽一下，其中vertex查询居然是通过 IndexSerializer的query方法跳过去了。。这代码逻辑够混乱。。

BackendTransaction类是一个非常重要的类，看一下类说明：

> Bundles all storage/index transactions and provides a proxy for some of their methods for convenience. Also increases robustness of read call by attempting read calls multiple times on failure.

还有类庞大和重要的类成员：

	public class BackendTransaction implements LoggableTransaction {

	    private static final Logger log =
	            LoggerFactory.getLogger(BackendTransaction.class);

	    public static final int MIN_TASKS_TO_PARALLELIZE = 2;

	    //Assumes 64 bit key length as specified in IDManager
	    public static final StaticBuffer EDGESTORE_MIN_KEY = BufferUtil.zeroBuffer(8);
	    public static final StaticBuffer EDGESTORE_MAX_KEY = BufferUtil.oneBuffer(8);

	    private final CacheTransaction storeTx;
	    private final BaseTransactionConfig txConfig;
	    private final StoreFeatures storeFeatures;

	    private final KCVSCache edgeStore;
	    private final KCVSCache indexStore;
	    private final KCVSCache txLogStore;

	    private final Duration maxReadTime;

	    private final Executor threadPool;

	    private final Map<String, IndexTransaction> indexTx;

	    private boolean acquiredLock = false;
	    private boolean cacheEnabled = true;
	    
	    。。。

	}
	