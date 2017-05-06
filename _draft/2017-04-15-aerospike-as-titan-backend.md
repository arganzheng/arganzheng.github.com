---
title: Aerospike如何作为Titan的Storage Backend
layout: post
catalog: true
---

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

实现其实蛮简单的，因为OrderedKeyValueStore的接口本身就很简单。基本都是简单的K-V增删改查，除了 getSlice ，用到了Cursor。


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

1、开始一个事务: public BerkeleyJETx beginTransaction(final BaseTransactionConfig txCfg) throws BackendException;

2、打开一个Database(这里是BerkeleyJEKeyValueStore): public BerkeleyJEKeyValueStore openDatabase(String name) throws BackendException;

3、处理批量更新: public void mutateMany(Map<String, KVMutation> mutations, StoreTransaction txh) throws BackendException;

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


