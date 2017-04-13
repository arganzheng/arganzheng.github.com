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

1、BerkeleyJEKeyValueStore

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
	            OperationStatus status = cursor.getSearchKeyRange(foundKey, foundData, getLockMode(txh));
	            //Iterate until given condition is satisfied or end of records
	            while (status == OperationStatus.SUCCESS) {
	                StaticBuffer key = getBuffer(foundKey);

	                if (key.compareTo(keyEnd) >= 0)
	                    break;

	                if (selector.include(key)) {
	                    result.add(new KeyValueEntry(key, getBuffer(foundData)));
	                }

	                if (selector.reachedLimit())
	                    break;

	                status = cursor.getNext(foundKey, foundData, getLockMode(txh));
	            }
	            log.trace("db={}, op=getSlice, tx={}, resultcount={}", name, txh, result.size());
	//            log.trace("db={}, op=getSlice, tx={}, resultcount={}", name, txh, result.size(), new Throwable("getSlice trace"));

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


2、BerkeleyJEStoreManager

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
然后就是实现接口的方法。


3、BerkeleyJETx

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

让我们看一下StandardTitanTx的commit代码，其实非常简单：

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

首先会调用prepareCommit，根据addedRelations和deletedRelations判断是否有数据更新(hasModifications)和索引更新(has2iModifications)

prepareCommit方法还会调用BackendTransaction的mutateEdges方法应用修改，而后者会调用edgeStore的mutateEntries方法，而edgeStore则最后委托给StoreTransaction（CacheTransaction）的mutate方法，这个方法会将修改缓存起来。

同样修改的索引，prepareCommit方法也会调用BackendTransaction的mutateIndex方法，最后缓存在StoreTransaction（CacheTransaction）的mutations变量中。

然后调用BackendTransaction的commitStorage()方法，提交数据存储。这个方法会调用storeTx.commit()

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
	