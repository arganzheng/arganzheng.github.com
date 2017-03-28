---
title: ArangoDB的索引学习
layout: post
---


ArangoDB索引概览
---------------

ArangoDB内建了很多索引结构，用于解决不同的应用场景（个人感觉有点过多了。。）

* Primary Index:
	* 主键索引: `_id`, `_key`
	* unsorted hash index
	* 内存索引
* Edge Index
	* 边索引：`_from`, `_to`
	* hash index
	* 内存索引
* Hash Index: 精确查询
	* unique hash index
	* unique, sparse hash index
	* non unique hash index
	* non unique, sparse hash index
	* 内存索引
* Skiplist Index: 范围查询
	* unique skiplist index
	* unique, sparse skiplist index
	* non-unique skiplist index
	* non-unique, sparse skiplist index
	* 内存索引
* Fulltext Index
	* word tokenization is done using the word boundary analysis provided by libicu
	* words are indexed in their lower-cased form
	* supports complete match queries (full words) and prefix queries, plus basic logical operations such as and, or and not for combining partial results.
	* 每个Collection只支持对一个属性构建fulltext index！
	* 只支持长度 <= 40个单词 的文档属性
	* 使用libicu对该属性进行分词构建索引
	* 支持对嵌套属性和数组元素进行索引
* Geo Index
	* [Using Hilbert curves and Polyhedrons for Geo-Indexing](https://www.arangodb.com/2012/03/31/using-hilbert-curves-and-polyhedrons-for-geo-indexing)
* Persistent Index
	* a sorted index with persistence
	* implementation is using the RocksDB engine
	* is used as a secondary indexes like MySQL
	* O(log n) to get the primary key + O(1) to fetch the actural document
	* sorted index, can be used for point lookups, range queries and sorting operations

**TIPS & NOTES**

1、ArangoDB的数据文件使用的是内存映射文件(memory mapped files)，索引除了Persistent Index，其他的都是纯内存(normal memory)索引，因此内存占用率会比较高；另外，重启的时候所有内存索引需要重新构建（或者打开某个Collection，该Collection相关的索引也会构建），导致启动非常耗时而且耗CPU。这也是后来引入Persitent Index的原因。[Suggestion: Have indexes also on disk besides "only memory indexes" #209](https://github.com/arangodb/arangodb/issues/209#issuecomment-193838232)。

另外，即使是内存索引，如果 size of indexes > size of ram，那么也会被操作系统换出。[The kernel will swap only to avoid an out of memory condition](https://en.wikipedia.org/wiki/Swappiness)。另一方面，ArangoDB的数据是以内存映射文件(Memory-Mapped Files)的方式加载的，数据量稍微一大，必然会发生换页。


2、相对于其他的图存储，ArangoDB很友好的支持组合索引、数组索引和嵌套属性索引。

* Combined index over multiple fields
	* db.posts.ensureIndex({ type: "hash", fields: [ "name", "age" ] })
* Indexing array values
	* db.posts.ensureIndex({ type: "hash", fields: [ "tags[*]" ] });
* Indexing attributes and sub-attributes
	* db.posts.ensureIndex({ type: "hash", fields: [ "name" ] })
	* db.posts.ensureIndex({ type: "hash", fields: [ "name.last" ] })	 

3、ArangoDB还引入[Vertex-Centric Indexes](https://docs.arangodb.com/3.1/Manual/Indexing/VertexCentric.html)解决图遍历过程中的supernode问题

A vertex-centric can be either of the following types:

* Hash Index
* Skiplist Index
* Persistent Index

4、其实从功能上出发，可以更好的理解为什么需要这些索引类型：

* 精确匹配 => Hash(Unsorted)
* 范围查询 => Sorted => B+ Tree or Skiplist
* 模糊匹配 => Fulltext 
* 地理位置查询 => Geo

这些索引理论上来说应该是可以内存，也可以是持久化的。是否持久化应该是一个选项(flag)，而不是一种索引类型。不过正如，ArangoDB官方所说的，直接新增一种持久化索引比将原来内存的索引变成持久化更简单一些，他们原来是有想在shutdown接口将内存索引持久化到磁盘的。


ArangoDB索引类层次和基本概念
-------------------------

ArangoDB的索引是有层级关系的，如下所示：

**ArangDB索引的类层级结构**

* Index
  	* MMFilesPrimaryIndex: uint8_t => MMFilesSimpleIndexElement
  	* MMFilesEdgeIndex: arangodb::velocypack::Slice => MMFilesSimpleIndexElement
	* MMFilesFulltextIndex
	* MMFilesGeoIndex
	* MMFilesPathBasedIndex
    	* MMFilesHashIndex: arangodb::velocypack::Slice => MMFilesHashIndexElement*
	    * MMFilesSkiplistIndex: MMFilesSkiplistIndexElement
	    * MMFilesPersistentIndex: MMFilesSkiplistIndexElement

Index基类，定义在 arangodb/arangod/Indexes/Index.h 文件：

    class Index {
      public:
    
        ...

	  	Index(TRI_idx_iid_t, LogicalCollection*,
	          std::vector<std::vector<arangodb::basics::AttributeName>> const&,
	          bool unique, bool sparse);

	    Index(TRI_idx_iid_t, LogicalCollection*, arangodb::velocypack::Slice const&);

	    explicit Index(arangodb::velocypack::Slice const&);

	  public:
	    /// @brief index types
	    enum IndexType {
	      TRI_IDX_TYPE_UNKNOWN = 0,
	      TRI_IDX_TYPE_PRIMARY_INDEX,
	      TRI_IDX_TYPE_GEO1_INDEX,
	      TRI_IDX_TYPE_GEO2_INDEX,
	      TRI_IDX_TYPE_HASH_INDEX,
	      TRI_IDX_TYPE_EDGE_INDEX,
	      TRI_IDX_TYPE_FULLTEXT_INDEX,
	      TRI_IDX_TYPE_SKIPLIST_INDEX,
	      TRI_IDX_TYPE_ROCKSDB_INDEX
	    };

	    ...

	    virtual IndexType type() const = 0;

	  private:
	    /// @brief set fields from slice
	    void setFields(VPackSlice const& slice, bool allowExpansion);

	  protected:
	    /// the index id
	    TRI_idx_iid_t const _iid;

	    /// the underlying collection
	    LogicalCollection* _collection;

	    /// the index fields
	    std::vector<std::vector<arangodb::basics::AttributeName>> _fields;

	    mutable bool _unique;

	    mutable bool _sparse;
	};

Index类很清晰明了，其实就是表示某个Collection的哪些 _fields 需要构建索引，索引类型是什么，索引id是什么，是不是 _unique ，是不是 _sparse ，例如：

	db.posts.ensureIndex({ type: "hash", fields: [ "name", "name.last", "age", "tag[*]" ], sparse: true })

然后每个具体的索引类型(hash, skiplist, fulltext, persistent等)由具体的子类实现。

这里有一个地方需要注意一下，就是_fields的数据类型：

	std::vector<std::vector<arangodb::basics::AttributeName>> _fields;

AttributeName是要索引字段(属性)的名称：

	struct AttributeName {
	    std::string name;  // 属性名称，如name，name.last
	    bool shouldExpand; // if the attribute was followed by [*] which means it should be expanded. Only works on arrays.
	    ...
	}  

_fields结构体是在通过解析用户的ensureIndex语句得到的，如上面的这条对posts Collection构建索引的语句：

    db.posts.ensureIndex({ type: "hash", fields: [ "name", "name.last", "age", "tag[*]" ], sparse: true })

会通过Index.cpp中的validateFields进行解析：

	/// @brief validate fields from slice
	void Index::validateFields(VPackSlice const& slice) {
	  bool const allowExpansion =
	      Index::allowExpansion(Index::type(slice.get("type").copyString()));

	  VPackSlice fields = slice.get("fields");

	  if (!fields.isArray()) {
	    THROW_ARANGO_EXCEPTION_MESSAGE(TRI_ERROR_ARANGO_ATTRIBUTE_PARSER_FAILED,
	                                   "invalid index description");
	  }

	  for (auto const& name : VPackArrayIterator(fields)) {
	    if (!name.isString()) {
	      THROW_ARANGO_EXCEPTION_MESSAGE(TRI_ERROR_ARANGO_ATTRIBUTE_PARSER_FAILED,
	                                     "invalid index description");
	    }

	    std::vector<arangodb::basics::AttributeName> parsedAttributes;
	    TRI_ParseAttributeString(name.copyString(), parsedAttributes,
	                             allowExpansion);
	  }
	}

fields属性中的每个元素会通过TRI_ParseAttributeString方法进行解析。注意，解析得到的是一个vector<AttributeName>，而不是简单的一个AttributeName结构，这是因为ArangoDB支持嵌套属性如 "name.last" ，得到的是一个 size=2 的vector，第一个元素是"name"，第二个是"last"。具体可以参考 arangodb/tests/Basics/AttributeNameParserTest.cpp 测试文件，里面有相应的测试用例。

另外，为了支持数组属性索引，ArangoDB引入了 [Array expansion](https://docs.arangodb.com/3.1/AQL/Advanced/ArrayOperators.html#array-expansion) 的概念，使用的符号是 `"[*]"`。所以，对于 `"foo.bar[*]"` 的输入，会得到这样的解析结果：
  
	std::vector<AttributeName> result;

	TRI_ParseAttributeString(input, result, true);
	  
	CHECK(result.size() == static_cast<size_t>(2));
	CHECK(result[0].name == "foo");
	CHECK(result[0].shouldExpand == false);
	CHECK(result[1].name == "bar");
	CHECK(result[1].shouldExpand == true);

相信大家现在应该知道为什么 _fields 是一个二维数组(vector of vector)，而不是简单的一维数组(vector)了。

这里也引出来ArangoDB一个非常重要的概念：path based index，这是ArangoDB支持嵌套属性索引的关键。

path based index其实全称应该是attribute path based index。对于一级属性(top level attribute)，attribute path就是attrbute name；但是对于嵌套属性(nested attrubutes/sub-attrbutes)，则attrbute path是"."号连接的路径，如 "name.last"。

path based index 和上面介绍的Array expansion 结合起来，可以达到很灵活的查询需求，如下面语句指定返回用户的名字和他所有朋友的名字：

    FOR u IN users
      RETURN { name: u.name, friends: u.friends[*].name }

这个概念的实现类是MMFilesPathBasedIndex，定义在 arangodb/arangod/MMFiles/MMFilesPathBasedIndex.h，继承自上面的Index基类，从这个类开始引入paths的概念：

	class MMFilesPathBasedIndex : public Index {
	    ...
	   
	  public:

	    /// @brief return the attribute paths
	    std::vector<std::vector<std::string>> const& paths() const {
	      return _paths;
	    }

	    /// @brief return the attribute paths, a -1 entry means none is expanding,
	    /// otherwise the non-negative number is the index of the expanding one.
	    std::vector<int> const& expanding() const {
	      return _expanding;
	    }

	  protected:
	    /// @brief helper function to insert a document into any index type
	    template<typename T>
	    int fillElement(std::vector<T*>& elements, 
	            TRI_voc_rid_t revisionId, arangodb::velocypack::Slice const&);

	  private:

	    /// @brief helper function to transform AttributeNames into string lists
	    void fillPaths(std::vector<std::vector<std::string>>& paths,
	                   std::vector<int>& expanding);

	    /// @brief helper function to create a set of index combinations to insert
	    std::vector<std::pair<VPackSlice, uint32_t>> buildIndexValue(VPackSlice const documentSlice);

	    /// @brief helper function to create a set of index combinations to insert
	    void buildIndexValues(VPackSlice const document, size_t level,
	                          std::vector<std::vector<std::pair<VPackSlice, uint32_t>>>& toInsert,
	                          std::vector<std::pair<VPackSlice, uint32_t>>& sliceStack);

	  protected:

	    /// @brief the attribute paths
	    std::vector<std::vector<std::string>> _paths;

	    /// @brief ... and which of them expands
	    std::vector<int> _expanding;

	    /// @brief whether or not at least one attribute is expanded
	    bool _useExpansion;

	    /// @brief whether or not partial indexing is allowed
	    bool _allowPartialIndex;
	};
  

我们看看对应的实现 MMFilesPathBasedIndex.cpp。其中最重要的方法是fillElement方法，它把文档转换成任何索引结构T：

	/// @brief helper function to insert a document into any index type
	template<typename T>
	int MMFilesPathBasedIndex::fillElement(std::vector<T*>& elements, /// 返回的结果
	                                TRI_voc_rid_t revisionId, 
	                                VPackSlice const& doc) {

	  size_t const n = _paths.size();

	  if (!_useExpansion) { /// 如果文档中有数组字段，则_useExpansion为true
	    // fast path for inserts... no array elements used
	    auto slices = buildIndexValue(doc);

	    if (slices.size() == n) {
	      // if slices.size() != n, then the value is not inserted into the index
	      // because of index sparsity!
	      T* element = static_cast<T*>(_allocator->allocate());
	      TRI_ASSERT(element != nullptr);
	      element = T::initialize(element, revisionId, slices);
	      ...
	      try {
	        elements.emplace_back(element);
	      } catch (...) {
	        _allocator->deallocate(element);
	        return TRI_ERROR_OUT_OF_MEMORY;
	      }
	    }
	  } else { /// 有数组字段需要构建索引
	    // other path for handling array elements, too
	    std::vector<std::vector<std::pair<VPackSlice, uint32_t>>> toInsert;
	    std::vector<std::pair<VPackSlice, uint32_t>> sliceStack;

	    buildIndexValues(doc, 0, toInsert, sliceStack);

	    if (!toInsert.empty()) {
	      elements.reserve(toInsert.size());

	      for (auto& info : toInsert) {
	        TRI_ASSERT(info.size() == n);
	        T* element = static_cast<T*>(_allocator->allocate());
	        TRI_ASSERT(element != nullptr);
	        element = T::initialize(element, revisionId, info);
	        ...
	        try {
	          elements.emplace_back(element);
	        } catch (...) {
	          _allocator->deallocate(element);
	          return TRI_ERROR_OUT_OF_MEMORY;
	        }
	      }
	    }
	  }

	  return TRI_ERROR_NO_ERROR;
	}

其中 _paths 属性是通过 fillPaths 方法将 _fields 属性转成得到的一个 `std::vector<std::vector<std::string>>` 结构。

逻辑蛮简单的：如果没有数组字段需要索引，那么就调用 `auto slices = buildIndexValue(doc);` 构建索引；否则调用 `buildIndexValues(doc, 0, toInsert, sliceStack);`。

	/// @brief helper function to create the sole index value insert
	std::vector<std::pair<VPackSlice, uint32_t>> MMFilesPathBasedIndex::buildIndexValue(
	    VPackSlice const documentSlice) {
	  size_t const n = _paths.size();

	  std::vector<std::pair<VPackSlice, uint32_t>> result;
	  for (size_t i = 0; i < n; ++i) {
	    VPackSlice slice = documentSlice.get(_paths[i]);
	    if (slice.isNone() || slice.isNull()) { // attribute not found
	      if (_sparse) {
	        // if sparse we do not have to index, this is indicated by result being shorter than n
	        result.clear();
	        break;
	      }
	      // null, note that this will be copied later!
	      result.emplace_back(arangodb::basics::VelocyPackHelper::NullValue(), 0); // fake offset 0
	    } else {
	      result.emplace_back(slice, static_cast<uint32_t>(slice.start() - documentSlice.start()));
	    }
	  }
	  return result;
	}

IndexValue是pair<VPackSlice, uint32_t>对象。其中 VPackSlice 是该属性对应的文档内容： 

	VPackSlice slice = documentSlice.get(_paths[i]);

而uint32_t 则是它相对于文档的偏移量：

	static_cast<uint32_t>(slice.start() - documentSlice.start());

buildIndexValues则要复杂的多，是一个递归函数来的，有点长，这里不展开了。

**TIPS & NOTES** 

1、`VPackSlice::get(std::vector<std::string>({ "name", "last" })))` 拿到的是subattribute "name.last"对应的值。具体参见：[velocypack](https://github.com/arangodb/velocypack/blob/master/examples/API.md)。所以IndexValue其实key就是属性值，value是属性值在整个document中的偏移量。

2、MMFilesFulltextIndex和MMFilesGeoIndex并继承自MMFilesPathBasedIndex，所以它们并不支持嵌套属性（MMFilesFulltextIndex会默认对属性的子属性也构建索引，但是不是一个概念）。

再回到fillElement函数。buildIndexValue将document构建出vector<IndexValue>，这些indexValues会转换成std::vector<T*>& elements。从代码看来，T应该是 MMFilesIndexElement.h中定义的 XXXIndexElement 结构体:

    auto slices = buildIndexValue(doc);
    ...
	T* element = static_cast<T*>(_allocator->allocate());
	TRI_ASSERT(element != nullptr);
	element = T::initialize(element, revisionId, slices);
	...
    elements.emplace_back(element);
	  
可以看到T必须有initialize函数定义，这些xxxIndexElement都定义在arangod/MMFiles/MMFilesIndexElement.h中。

MMFilesHashIndexElement：

	/// @brief hash index element. Do not directly construct it.
	struct MMFilesHashIndexElement {
	  // Do not use new for this struct, use create()!
	 private:
	  MMFilesHashIndexElement(TRI_voc_rid_t revisionId, std::vector<std::pair<VPackSlice, uint32_t>> const& values);

	 public:
	  ...
	  /// @brief base memory usage of an index element
	  static constexpr size_t baseMemoryUsage(size_t numSubs) {
	    return sizeof(TRI_voc_rid_t) + sizeof(uint32_t) + (sizeof(MMFilesIndexElementValue) * numSubs);
	  }
	  
	  inline MMFilesIndexElementValue const* subObject(size_t position) const {
	    char const* p = reinterpret_cast<char const*>(this) + baseMemoryUsage(position);
	    return reinterpret_cast<MMFilesIndexElementValue const*>(p);
	  }
	  /// @brief allocate a new index element from a vector of slices
	  static MMFilesHashIndexElement* initialize(MMFilesHashIndexElement* memory, 
	                                      TRI_voc_rid_t revisionId, 
	                                      std::vector<std::pair<arangodb::velocypack::Slice, uint32_t>> const& values);
	  ...
	 private:
	  TRI_voc_rid_t _revisionId;
	  uint32_t _hash;
	};

MMFilesSkiplistIndexElement：

	/// @brief skiplist index element. Do not directly construct it.
	struct MMFilesSkiplistIndexElement {
	  // Do not use new for this struct, use create()!
	 private:
	  MMFilesSkiplistIndexElement(TRI_voc_rid_t revisionId, std::vector<std::pair<VPackSlice, uint32_t>> const& values);
	 public:
	  /// @brief base memory usage of an index element
	  static constexpr size_t baseMemoryUsage(size_t numSubs) {
	    return sizeof(TRI_voc_rid_t) + (sizeof(MMFilesIndexElementValue) * numSubs);
	  }
	  
	  inline MMFilesIndexElementValue const* subObject(size_t position) const {
	    char const* p = reinterpret_cast<char const*>(this) + baseMemoryUsage(position);
	    return reinterpret_cast<MMFilesIndexElementValue const*>(p);
	  }

	  ...
	  /// @brief allocate a new index element from a vector of slices
	  static MMFilesSkiplistIndexElement* initialize(MMFilesSkiplistIndexElement* element,
	                                          TRI_voc_rid_t revisionId, 
	                                          std::vector<std::pair<arangodb::velocypack::Slice, uint32_t>> const& values);
	 private:
	  TRI_voc_rid_t _revisionId;
	};

MMFilesSimpleIndexElement比较特殊，它没有initialize，所以比较simple:

	struct MMFilesSimpleIndexElement {
	 public:
	  MMFilesSimpleIndexElement(TRI_voc_rid_t revisionId, arangodb::velocypack::Slice const& value, uint32_t offset); 
	  MMFilesSimpleIndexElement(MMFilesSimpleIndexElement const& other) : _revisionId(other._revisionId), _hashAndOffset(other._hashAndOffset) {}
	  MMFilesSimpleIndexElement& operator=(MMFilesSimpleIndexElement const& other) noexcept {
	    _revisionId = other._revisionId;
	    _hashAndOffset = other._hashAndOffset;
	    return *this;
	  }

	  /// @brief get the revision id of the document
	  inline TRI_voc_rid_t revisionId() const { return _revisionId; }
	  inline uint64_t hash() const { return _hashAndOffset & 0xFFFFFFFFULL; }
	  inline uint32_t offset() const { return static_cast<uint32_t>((_hashAndOffset & 0xFFFFFFFF00000000ULL) >> 32); }
	  arangodb::velocypack::Slice slice(IndexLookupContext*) const;
	  
	  inline operator bool() const { return _revisionId != 0; }
	  inline bool operator==(MMFilesSimpleIndexElement const& other) const {
	    return _revisionId == other._revisionId && _hashAndOffset == other._hashAndOffset;
	  }
	  inline bool operator<(MMFilesSimpleIndexElement const& other) const {
	    return _revisionId < other._revisionId;
	  }
	  
	  static uint64_t hash(arangodb::velocypack::Slice const& value);
	  inline void updateRevisionId(TRI_voc_rid_t revisionId, uint32_t offset) { 
	    _revisionId = revisionId; 
	    _hashAndOffset &= 0xFFFFFFFFULL; 
	    _hashAndOffset |= (static_cast<uint64_t>(offset) << 32);
	  }
	  
	 private:
	  TRI_voc_rid_t _revisionId;
	  uint64_t _hashAndOffset;
	};

这里看一下 MMFilesHashIndexElement 的 initialize 实现（其实 MMFilesSkiplistIndexElement 的 initialize 实现基本一样）：

	MMFilesHashIndexElement* MMFilesHashIndexElement::initialize(MMFilesHashIndexElement* element, 
	                                               TRI_voc_rid_t revisionId, 
	                                               std::vector<std::pair<arangodb::velocypack::Slice, uint32_t>> const& values) {
	  TRI_ASSERT(!values.empty());
	  return new (element) MMFilesHashIndexElement(revisionId, values);
	}

	MMFilesHashIndexElement::MMFilesHashIndexElement(TRI_voc_rid_t revisionId, std::vector<std::pair<VPackSlice, uint32_t>> const& values) 
	    : _revisionId(revisionId), _hash(static_cast<uint32_t>(hash(values))) {
	   
	  for (size_t i = 0; i < values.size(); ++i) {
	    subObject(i)->fill(values[i].first, values[i].second);
	  }
	}

最后是调用了subObject（MMFilesIndexElementValue）的fill方法，参数就是我们前面通过 buildIndexValue 得到的 `pair<arangodb::velocypack::Slice, uint32_t>` 。

subObject的类型是 MMFilesIndexElementValue ，定义蛮简单的：

	/// @brief velocypack sub-object (for indexes, as part of IndexElement, 
	/// if the last byte in data[] is 0, then the VelocyPack data is managed 
	/// by the datafile the element is in. If the last byte in data[] is 1, then 
	/// value.data contains the actual VelocyPack data in place.
	struct MMFilesIndexElementValue {
	  friend struct MMFilesHashIndexElement;
	  friend struct MMFilesSkiplistIndexElement;

	 public:
	  ...

	  /// @brief fill an MMFilesIndexElementValue structure with a subvalue
	  void fill(VPackSlice const value, uint32_t offset) {
	    VPackValueLength len = value.byteSize();
	    if (len <= maxValueLength()) {
	      setInline(value.start(), static_cast<size_t>(len));
	    } else {
	      setOffset(offset);
	    }
	  }

	  /// @brief velocypack sub-object (for indexes, as part of IndexElement, 
	  /// if offset is non-zero, then it is an offset into the VelocyPack data in
	  /// the data or WAL file. If offset is 0, then data contains the actual data
	  /// in place.
	  arangodb::velocypack::Slice slice(IndexLookupContext* context) const;
	  
	  inline bool isOffset() const noexcept {
	    return !isInline();
	  }

	  inline bool isInline() const noexcept {
	    return value.data[maxValueLength()] == 1;
	  }

	 private:
	  void setOffset(uint32_t offset) {
	    value.offset = offset;
	    value.data[maxValueLength()] = 0; // type = offset
	  }
	    
	  void setInline(uint8_t const* data, size_t length) noexcept {
	    TRI_ASSERT(length > 0);
	    TRI_ASSERT(length <= maxValueLength());
	    memcpy(&value.data[0], data, length);
	    value.data[maxValueLength()] = 1; // type = inline
	  }

	  static constexpr size_t maxValueLength() noexcept {
	    return sizeof(value.data) - 1;
	  }
	 
	 private:
	  union {
	    uint8_t data[12];
	    uint32_t offset;
	  } value;
	};

fill方法逻辑很简单，如果属性值value的长度小于 maxValueLength（即 sizeof(value.data) - 1），那么就直接存储属性值(setInline)，否则，只存储value在document中的offset。


ArangoDB索引实现源码剖析
----------------------

### 1、Primary Index、Edge Index和Hash Index

第一眼看到ArangoDB的索引类型介绍，总以为是文档写错了。因为这个分类确实太诡异了。Primary Index和Edge Index看起来是按照功能（或者使用场景）来划分，Hash Index、Skiplist Index等则是按照索引使用的数据结构来划分，而最诡异的是Persistent Index，看起来是按照是否索引是内存索引还是持久化索引来划分。。简直让人看了一头雾水。

正常来说，Primary Index和Edge Index都是属于精确查找，所以使用Hash Index来实现是最直接高效的方式，而且Hash Index应该可以是内存索引，也可以是持久化索引，取决于用户的配置才对。

深入代码研究发现，ArangoDB确实是使用了Hash Index实现Primary Index和Edge Index：

	arangod/MMFiles/MMFilesPrimaryIndex.h

	typedef arangodb::basics::AssocUnique<uint8_t, MMFilesSimpleIndexElement> MMFilesPrimaryIndexImpl;

	class MMFilesPrimaryIndex final : public Index {

		...

	private:
	  	/// @brief the actual index
	  	MMFilesPrimaryIndexImpl* _primaryIndex;
	};

AssocUnique类就是一个key必须唯一的hash array（内部使用的是可以动态扩容的数组来实现哈希：std::vector<Bucket> _buckets;）。
而EdgeIndex也是使用了Hash Index实现：

	arangod/MMFiles/MMFilesEdgeIndex.h

	typedef arangodb::basics::AssocMulti<arangodb::velocypack::Slice, MMFilesSimpleIndexElement, 
                                         uint32_t, false> TRI_MMFilesEdgeIndexHash_t;

	private:
	/// @brief the hash table for _from
	TRI_MMFilesEdgeIndexHash_t* _edgesFrom;

	/// @brief the hash table for _to
	TRI_MMFilesEdgeIndexHash_t* _edgesTo;

相对于Primary Index主键是唯一的，Edge Index的key是可以重复的，所以底层使用的是AssocMulti类（associative array of pointers, tolerating repeated keys。内部使用的是可以动态扩容的数组来实现哈希：std::vector<Bucket> _buckets;）。

而Hash Index是由类MMFilesHashIndex来实现，这个类其实就是对单值key和多值key的一个封装而已：

	class MMFilesHashIndex final : public MMFilesPathBasedIndex {

	public:

	  ...

	  int insertUnique(transaction::Methods*, TRI_voc_rid_t,
	                   arangodb::velocypack::Slice const&, bool isRollback);
	  int insertMulti(transaction::Methods*, TRI_voc_rid_t,
	                   arangodb::velocypack::Slice const&, bool isRollback);

	private:
	  /// @brief the actual hash index (unique type)
	  typedef arangodb::basics::AssocUnique<arangodb::velocypack::Slice,
	                                        MMFilesHashIndexElement*> TRI_HashArray_t;

	  struct UniqueArray {
	    UniqueArray() = delete;
	    UniqueArray(size_t numPaths, TRI_HashArray_t*, HashElementFunc*,
	                IsEqualElementElementByKey*);

	    ~UniqueArray();

	    TRI_HashArray_t* _hashArray;    // the hash array itself, unique values
	    HashElementFunc* _hashElement;  // hash function for elements
	    IsEqualElementElementByKey* _isEqualElElByKey;  // comparison func
	    size_t _numPaths;
	  };

	  /// @brief the actual hash index (multi type)
	  typedef arangodb::basics::AssocMulti<arangodb::velocypack::Slice,
	                                       MMFilesHashIndexElement*, uint32_t,
	                                       false> TRI_HashArrayMulti_t;

	  struct MultiArray {
	    MultiArray() = delete;
	    MultiArray(size_t numPaths, TRI_HashArrayMulti_t*, HashElementFunc*,
	               IsEqualElementElementByKey*);
	    ~MultiArray();

	    TRI_HashArrayMulti_t*
	        _hashArray;                 // the hash array itself, non-unique values
	    HashElementFunc* _hashElement;  // hash function for elements
	    IsEqualElementElementByKey* _isEqualElElByKey;  // comparison func
	    size_t _numPaths;
	  };

	  union {
	    UniqueArray* _uniqueArray;
	    MultiArray* _multiArray;
	  };
	};
      
其实我不大明白为什么Primary Index和Edge Index不直接使用MMFileHashIndex，而使用底层的AssocUnique和AssocMulti。


### 2、Skiplist Index

Skiplist索引跟Hash Index一样，也是一个典型的数据结构实现(arangod/MMFiles/MMFilesSkiplist.h):

常量定义。最多48个层级：

 	// We will probably never see more than 2^48 documents in a skip list
 	#define TRI_SKIPLIST_MAX_HEIGHT 48

首先是节点的定义，其实就是一个有层级的双向链表：

	////////////////////////////////////////////////////////////////////////////////
	/// @brief type of a skiplist node
	////////////////////////////////////////////////////////////////////////////////

	template <class Key, class Element>
	class MMFilesSkiplist;

	template <class Key, class Element>
	class MMFilesSkiplistNode {
	  friend class MMFilesSkiplist<Key, Element>;
	  MMFilesSkiplistNode<Key, Element>** _next;
	  MMFilesSkiplistNode<Key, Element>* _prev;
	  Element* _doc;
	  int _height;

	  ...

	}

节点有两种比较方式，以函数指针的形式提供：

	////////////////////////////////////////////////////////////////////////////////
	/// @brief two possibilities for comparison, preorder & total ordre
	////////////////////////////////////////////////////////////////////////////////

	enum MMFilesSkiplistCmpType { SKIPLIST_CMP_PREORDER, SKIPLIST_CMP_TOTORDER };

最后是整个Skiplist的定义：

	////////////////////////////////////////////////////////////////////////////////
	/// @brief type of a skiplist
	/// _end always points to the last node in the skiplist, this can be the
	/// same as the _start node. If a node does not have a successor on a certain
	/// level, then the corresponding _next pointer is a nullptr.
	////////////////////////////////////////////////////////////////////////////////

	template <class Key, class Element>
	class MMFilesSkiplist {
	  typedef MMFilesSkiplistNode<Key, Element> Node;

	private:
	  Node* _start;
	  Node* _end;
	  CmpElmElmFuncType _cmp_elm_elm;
	  CmpKeyElmFuncType _cmp_key_elm;
	  FreeElementFuncType _free;
	  bool _unique;  // indicates whether multiple entries that
	                 // are equal in the preorder are allowed in
	  uint64_t _nrUsed;
	  bool _isArray;  // indicates whether this index is used to
	                  // index arrays.
	  size_t _memoryUsed;

	public:

		...

	}

Skiplist是一种比B tree简单的数据结构，具体细节可以参考整个PPT介绍: [SkipLists](https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/skiplists.pdf)


### 3、Fulltext Index

这个类型的索引主要是用来实现全文检索的，也是搜索引擎的核心，数据结构一般是用倒排索引。开源的搜索引擎内核是lucene，不过是Java版本的，C版本的lucene非常老旧。

不过ArangoDB这里并没有使用任何外部的倒排索引库，而是自己“简单”的实现了一个。

相对于前面介绍的索引，fulltext index涉及到的文件比较多，有13个：

	MMFilesFulltextIndex.cpp
	MMFilesFulltextIndex.h
	mmfiles-fulltext-common.h
	mmfiles-fulltext-handles.cpp
	mmfiles-fulltext-handles.h
	mmfiles-fulltext-index.cpp
	mmfiles-fulltext-index.h
	mmfiles-fulltext-list.cpp
	mmfiles-fulltext-list.h
	mmfiles-fulltext-query.cpp
	mmfiles-fulltext-query.h
	mmfiles-fulltext-result.cpp
	mmfiles-fulltext-result.h

先看一下 mmfiles-fulltext-index.h 文件，这个文件定义了Fulltext index的增删改查。

首先声明了三个数据结构：	

	struct TRI_fulltext_query_s;
	struct TRI_fulltext_result_s;
	struct TRI_fulltext_wordlist_s;

TRI_fulltext_query_s定义在 arangod/MMFiles/mmfiles-fulltext-query.h，表示全文查询语句：

	/// @brief maximum number of search words in a query
	/// 最多支持长度为32个单词的查询语句
	#define TRI_FULLTEXT_SEARCH_MAX_WORDS 32 

	/// @brief fulltext query match options
	/// 查询匹配类型：全匹配，前缀匹配，包含匹配(未实现)
	typedef enum {
	  TRI_FULLTEXT_COMPLETE,
	  TRI_FULLTEXT_PREFIX,
	  TRI_FULLTEXT_SUBSTRING  // currently not implemented, maybe later
	} TRI_fulltext_query_match_e;

	/// @brief fulltext query logical operators
	/// 单词之间的逻辑关系：AND, OR, EXCLUDE(NOT)
	typedef enum {
	  TRI_FULLTEXT_AND,
	  TRI_FULLTEXT_OR,
	  TRI_FULLTEXT_EXCLUDE
	} TRI_fulltext_query_operation_e;

	/// @brief fulltext query
	/// fulltext查询结构，每个字段都很直观。
	typedef struct TRI_fulltext_query_s {
	  size_t _numWords;
	  char** _words;
	  TRI_fulltext_query_match_e* _matches;
	  TRI_fulltext_query_operation_e* _operations;
	  size_t _maxResults;
	} TRI_fulltext_query_t;

	...


TRI_fulltext_result_s 定义在 arangod/MMFiles/mmfiles-fulltext-result.h，表示全文查询结构：

	// Forward declarations
	namespace arangodb {
	  struct DocumentIdentifierToken;
	}

	/// @brief typedef for a fulltext result list
	typedef struct TRI_fulltext_result_s {
	  uint32_t _numDocuments;
	  arangodb::DocumentIdentifierToken* _documents;
	} TRI_fulltext_result_t;

	...

其中引用到了 DocumentIdentifierToken 这个类，这个类是一个基础类，定义在 arangod/StorageEngine/DocumentIdentifierToken.h:

	// @brief This token is handed out by Indexes and
	// is used by StorageEngines to return a document
	// Only specializations of this token can be created.

	struct DocumentIdentifierToken {

		...

	 public:
	  uint64_t _data;
	};

其实就是一个uint64_t的句柄，可以是指针，也可以是ID(如果ID是长整数)。所以返回结构其实就是指向结果文档所在位置的数组。

第三个数据结构 TRI_fulltext_wordlist_s 没找到定义也没有找到使用方，应该是无用代码。不过字面意思很好理解，就是对查询语句/文档的切词结果。

让我们继续回到 mmfiles-fulltext-index.h 文件。接下来是定义了支持的单词长度范围：2~40个字符（每个字符最多4个字节，UTF8编码）：

	/// @brief maximum length of an indexed word in characters
	/// a character may consist of up to 4 bytes
	#define TRI_FULLTEXT_MAX_WORD_LENGTH 40

	/// @brief default minimum word length for a fulltext index
	#define TRI_FULLTEXT_MIN_WORD_LENGTH_DEFAULT 2

然后就是定义了倒排索引最常见也是最重要的操作接口：

	/// @brief create a fulltext index
	TRI_fts_index_t* TRI_CreateFtsIndex(uint32_t, uint32_t, uint32_t);

	/// @brief free a fulltext index
	void TRI_FreeFtsIndex(TRI_fts_index_t*);

	void TRI_TruncateMMFilesFulltextIndex(TRI_fts_index_t*);

	/// @brief delete a document from the index
	void TRI_DeleteDocumentMMFilesFulltextIndex(TRI_fts_index_t* const,
	                                            const TRI_voc_rid_t);

	/// @brief insert a list of words to the index
	bool TRI_InsertWordsMMFilesFulltextIndex(TRI_fts_index_t* const,
	                                         const TRI_voc_rid_t,
	                                         std::vector<std::string>&);

	/// @brief find all documents that contain a word (exact match)
	#if 0
	struct TRI_fulltext_result_s* TRI_FindExactMMFilesFulltextIndex (TRI_fts_index_t* const,
	                                                          char const* const,
	                                                          size_t const);
	#endif

	/// @brief find all documents that contain a word (prefix match)
	#if 0
	struct TRI_fulltext_result_s* TRI_FindPrefixMMFilesFulltextIndex (TRI_fts_index_t* const,
	                                                           char const*,
	                                                           size_t const);
	#endif

	/// @brief execute a query on the fulltext index
	/// note: this will free the query
	struct TRI_fulltext_result_s* TRI_QueryMMFilesFulltextIndex(
	    TRI_fts_index_t* const, struct TRI_fulltext_query_s*);

	/// @brief dump index tree
	#if TRI_FULLTEXT_DEBUG
	void TRI_DumpTreeFtsIndex(const TRI_fts_index_t* const);
	#endif

	/// @brief return the total memory used by the index
	size_t TRI_MemoryMMFilesFulltextIndex(const TRI_fts_index_t* const);

	/// @brief compact the fulltext index
	bool TRI_CompactMMFilesFulltextIndex(TRI_fts_index_t* const);

我们来看一下对应的具体实现 arangod/MMFiles/mmfiles-fulltext-index.cpp，这个文件有点长(1610行)：

fulltext-index在实现上是一颗树，定义如下：

	/// @brief the actual fulltext index
	typedef struct {
	  node_t* _root;  // root node of the index

	  TRI_fulltext_handles_t* _handles;  // handles management instance

	  TRI_read_write_lock_t _lock;

	  size_t _memoryAllocated;  // total memory used by index

	  uint32_t _nodeChunkSize;       // how many sub-nodes to allocate per chunk
	  uint32_t _initialNodeHandles;  // how many handles to allocate per node
	} index__t;

定义蛮简单的，主要数据结构就是一个根节点(node_t)和关联的handles(TRI_fulltext_handles_t)。

我们首先看一下节点定义：

	/// @brief maximum length of an indexed word in bytes
	/// a UTF-8 character can contain up to 4 bytes
	#define MAX_WORD_BYTES ((TRI_FULLTEXT_MAX_WORD_LENGTH)*4)

	/// @brief the type of characters indexed. should be one byte long
	typedef uint8_t node_char_t;

	/// @brief typedef for follower nodes. this is just void because for the
	/// compiler it is a sequence of binary data with no internal structure
	typedef void followers_t;

	/// @brief a node in the fulltext index
	///
	/// the _followers property is a pointer to dynamic memory. If it is NULL, then
	/// the node does not have any followers/sub-nodes. if the _followers property
	/// is non-NULL, it contains a byte stream consisting of the following values:
	/// - uint8_t numAllocated: number of sub-nodes we have allocated memory for
	/// - uint8_t numFollowers: the actual number of sub-nodes for the node
	/// - node_char_t* keys: keys of sub-nodes, sorted binary
	/// - node_t** sub-nodes: pointers to sub-nodes, in the same order as keys
	/// this structure is fiddly, but saves a lot of memory and malloc calls when
	/// compared to a "proper" structure.
	/// As the "properties" inside _followers are just binary data for the compiler,
	/// it is not wise to access them directly, but use the access functions this
	/// file provides. There is no need to calculate the offsets of the different
	/// sub-properties directly, as this is all done by special functions which
	/// provide the offsets at relatively low costs.
	///
	/// The _handles property is a pointer to dynamic memory, too. If it is NULL,
	/// then the node does not have any handles attached. If it is non-NULL, it
	/// contains a byte stream consisting of the following values:
	/// - uint32_t numAllocated: number of handles allocated for the node
	/// - unit32_t numEntries: number of handles currently in use
	/// - TRI_fulltext_handle_t* handles: all the handle values subsequently
	/// Note that the highest bit of the numAllocated value contains a flag whether
	/// the handles list is sorted or not. It is therefore not safe to access the
	/// properties directly, but instead always the special functions provided in
	/// fulltext-list.c must be used. These provide access to the individual values
	/// at relatively low cost
	typedef struct node_s {
	  followers_t* _followers;
	  TRI_fulltext_list_t* _handles;
	} node_t;

一个节点有多个followers（子节点），同时也可能attached一些handles。followers_t和TRI_fulltext_list_t其实都是void类型(typedef void)。不过注释上有大概解释他们的数据结构：

* followers_t
	- uint8_t numAllocated: number of sub-nodes we have allocated memory for
	- uint8_t numFollowers: the actual number of sub-nodes for the node
	- node_char_t* keys: keys of sub-nodes, sorted binary
	- node_t** sub-nodes: pointers to sub-nodes, in the same order as keys
* TRI_fulltext_list_t
    - uint32_t numAllocated: number of handles allocated for the node. Note that the highest bit of the numAllocated value contains a flag whether the handles list is sorted or not.
    - unit32_t numEntries: number of handles currently in use
    - TRI_fulltext_handle_t* handles: all the handle values subsequently，其实就是一个简单的uint32_t类型。

然后我们再看看 TRI_fulltext_handles_t* _handles; 的定义，定义在 arangod/MMFiles/mmfiles-fulltext-handles.h 中，这个文件有比较详细的注释：

	namespace arangodb {
	  struct DocumentIdentifierToken;
	}

	/// @brief a slot containing _numUsed handles and has some statistics about
	/// itself
	///
	/// the fulltext index will not store document ids in its nodes, because that
	/// will be complicated in the case of deleting a document. in this case, all
	/// nodes would need to be traversed to find where the document was referenced.
	/// this would be too slow. instead of storing document ids, a node stores
	/// handles. handles are increasing integer numbers that are each mapped to a
	/// specific document. when a document is deleted from the index, its handle is
	/// marked as deleted, but the handle value may remain stored in one or many
	/// index nodes. handles of deleted documents are removed from result sets at
	/// the end of each index query on-the-fly, so query results are still correct.
	/// To finally get rid of handles of deleted documents, the index can perform
	/// a compaction. The compaction rewrites a new, dense handle list consisting
	/// with only handles that point to existing documents. The old handles used in
	/// nodes become invalid by this, so the handles stores in the nodes have to
	/// be rewritten. When the rewrite is done, the old handle list is freed and
	/// the new one is put in place.
	///
	/// Inserting a new document will simply allocate a new handle, and the handle
	/// will be stored for the node. We simply assign the next handle number for
	/// the document. After that, we can quickly look up the document id for a
	/// handle value. It's more tricky the other way around, because there is no
	/// simple mapping from document ids to handles. To find the handle for a
	/// document id, we have to check all handles already used.
	/// As this would mean traversing over all handles used and comparing their
	/// document values with the sought document id, there is some optimisation:
	/// handles are stored in slots of fixed sizes. Each slot has some statistics
	/// about the number of used and deleted documents/handles in it, as well as
	/// its min and max document values.
	/// When looking for a specific document id in all handles in the case of
	/// deletion, the slot statistics are used to early prune non-relevant slots
	/// from the further search. The simple min/max document id check implemented is
	/// sufficient because normally document memory is contiguous so the pointers
	/// to documents are just adjacent (second pointer is higher than first pointer).
	/// This is only true for documents that are created on the same memory page
	/// but this should be the common case to optimize for.
	typedef struct TRI_fulltext_handle_slot_s {
	  uint32_t _numUsed;               // number of handles used in slot
	  uint32_t _numDeleted;            // number of deleted handles in slot
	  TRI_voc_rid_t _min;              // minimum handle value in slot
	  TRI_voc_rid_t _max;              // maximum handle value in slot
	  TRI_voc_rid_t* _documents;       // document ids for the slots
	  uint8_t* _deleted;               // deleted flags for the slots
	} TRI_fulltext_handle_slot_t;

	/// @brief typedef for a fulltext handles instance
	typedef struct TRI_fulltext_handles_s {
	  TRI_fulltext_handle_t _next;          // next handle to use
	  uint32_t _numSlots;                   // current number of slots
	  TRI_fulltext_handle_slot_t** _slots;  // pointers to slots
	  uint32_t _slotSize;                   // the size of each slot
	  uint32_t _numDeleted;                 // total number of deleted documents
	  TRI_fulltext_handle_t* _map;  // a temporary map for remapping existing
	                                // handles to new handles during compaction
	} TRI_fulltext_handles_t;

	...

	/// @brief get the document id for a handle
	arangodb::DocumentIdentifierToken TRI_GetDocumentMMFilesFulltextIndex(
	    const TRI_fulltext_handles_t* const, const TRI_fulltext_handle_t);

结构看起来蛮简单的，就是一个一维数组（TRI_fulltext_handle_slot_t ** _slots;），数组的每个元素就是一个slot（TRI_fulltext_handle_slot_t），slot存放相关联的文档id（以handles的方式存放）。

**说明** TRI_voc_rid_t定义在arangod/VocBase/voc-types.h中，就是一个简单的uint64_t：

	/// @brief revision identifier type
	typedef uint64_t TRI_voc_rid_t;

并且定义了从string转成TRI_voc_rid_t的方法。

最后，让我们看一下入口定义：arangod/MMFiles/MMFilesFulltextIndex.h：

	struct DocumentIdentifierToken;

	class MMFilesFulltextIndex final : public Index {

	 public:

	  	...

	  TRI_fts_index_t* internals() { return _fulltextIndex; }

	  static TRI_voc_rid_t fromDocumentIdentifierToken(
	      DocumentIdentifierToken const& token);
	  static DocumentIdentifierToken toDocumentIdentifierToken(
	      TRI_voc_rid_t revisionId);

	 private:
	  std::vector<std::string> wordlist(arangodb::velocypack::Slice const&);

	 private:
	  /// @brief the indexed attribute (path)
	  std::vector<std::string> _attr;

	  /// @brief the fulltext index
	  TRI_fts_index_t* _fulltextIndex;

	  /// @brief minimum word length
	  int _minWordLength;
	};
	
结构很直观，也蛮简单的：

* std::vector<std::string> wordlist(arangodb::velocypack::Slice const&): 切词后的单词列表
* std::vector<std::string> _attr: 需要建立fulltext索引的属性，实际上目前只支持一个属性
* TRI_fts_index_t* _fulltextIndex: fulltext index结构

其中最重要的数据结构：TRI_fts_index_t 定义在arangod/MMFiles/mmfiles-fulltext-common.h文件中：

	/// @brief typedef for the fulltext index
	/// this is just a void* for users, and the actual index definitions is only
	/// used internally in file fulltext-index.c
	/// the reason is the index does some binary stuff its users should not bother with
	typedef void TRI_fts_index_t;

结果令人大跌眼镜——居然是个void类型！！不过注释也说了，真正的定义在fulltext-index.c，就是我们前面分析过的arangod/MMFiles/mmfiles-fulltext-index.cpp，其实就是`index__t`。看一下arangod/MMFiles/MMFilesFulltextIndex.cpp中具体的操作验证一下我们的理解。首先看一下insert操作：

	int MMFilesFulltextIndex::insert(transaction::Methods*, TRI_voc_rid_t revisionId,
	                          VPackSlice const& doc, bool isRollback) {
	  int res = TRI_ERROR_NO_ERROR;

	  std::vector<std::string> words = wordlist(doc);
	   
	  if (words.empty()) {
	    // TODO: distinguish the cases "empty wordlist" and "out of memory"
	    // LOG_TOPIC(WARN, arangodb::Logger::FIXME) << "could not build wordlist";
	    return res;
	  }

	  // TODO: use status codes
	  if (!TRI_InsertWordsMMFilesFulltextIndex(_fulltextIndex, revisionId, words)) {
	    LOG_TOPIC(ERR, arangodb::Logger::FIXME) << "adding document to fulltext index failed";
	    res = TRI_ERROR_INTERNAL;
	  }
	  return res;
	}

VPackSlice是ArangoDB自己定义的一种序列化方式，可以快速的转换成JSON格式。[velocypack](https://github.com/arangodb/velocypack)。

先对文档属性进行分词:

	/// @brief callback function called by the fulltext index to determine the
	/// words to index for a specific document
	std::vector<std::string> MMFilesFulltextIndex::wordlist(VPackSlice const& doc) {
	  std::vector<std::string> words;
	  try {
	    VPackSlice const value = doc.get(_attr);

	    if (!value.isString() && !value.isArray() && !value.isObject()) {
	      // Invalid Input
	      return words;
	    }

	    ExtractWords(words, value, _minWordLength, 0);
	  } catch (...) {
	    // Backwards compatibility
	    // The pre-vpack impl. did just ignore all errors and returned nulltpr
	    return words;
	  }
	  return words;
	}

	/// @brief walk over the attribute. Also Extract sub-attributes and elements in list.
	static void ExtractWords(std::vector<std::string>& words,
	                         VPackSlice const value,
	                         size_t minWordLength,
	                         int level) {
	  if (value.isString()) {
	    // extract the string value for the indexed attribute
	    std::string text = value.copyString();

	    // parse the document text
	    arangodb::basics::Utf8Helper::DefaultUtf8Helper.getWords(
	        words, text, minWordLength, TRI_FULLTEXT_MAX_WORD_LENGTH, true);
	    // We don't care for the result. If the result is false, words stays
	    // unchanged and is not indexed
	  } else if (value.isArray() && level == 0) {
	    for (auto const& v : VPackArrayIterator(value)) {
	      ExtractWords(words, v, minWordLength, level + 1);
	    }
	  } else if (value.isObject() && level == 0) {
	    for (auto const& v : VPackObjectIterator(value)) {
	      ExtractWords(words, v.value, minWordLength, level + 1);
	    }
	  }
	}

然后调用 TRI_InsertWordsMMFilesFulltextIndex(_fulltextIndex, revisionId, words)); 将单词插入到`_fulltextIndex`中：

	/// @brief insert a list of words into the index
	/// calling this function requires a wordlist that has word with the correct lengths. 
	/// especially, words in the wordlist must not be longer than MAX_WORD_BYTES. 
	/// the caller must check this before calling this function
	///
	/// The function will sort the wordlist in place to
	/// - filter out duplicates on insertion
	/// - save redundant lookups of prefix nodes for adjacent words with shared prefixes
	bool TRI_InsertWordsMMFilesFulltextIndex(TRI_fts_index_t* const ftx,
	                                         const TRI_voc_rid_t document,
	                                         std::vector<std::string>& wordlist) {
	  index__t* idx;
	  TRI_fulltext_handle_t handle;
	  node_t* paths[MAX_WORD_BYTES + 4];
	  size_t lastLength;

	  if (wordlist.empty()) {
	    return true;
	  }

	  // initialize to satisfy scan-build
	  paths[0] = nullptr;
	  paths[MAX_WORD_BYTES] = nullptr;

	  // the words must be sorted so we can avoid duplicate words and use an optimization
	  // for words with common prefixes (which will be adjacent in the sorted list of words)
	  // The default comparator (<) is exactly what we need here
	  std::sort(wordlist.begin(), wordlist.end());

	  idx = (index__t*)ftx;

	  TRI_WriteLockReadWriteLock(&idx->_lock);

	  // get a new handle for the document
	  handle = TRI_InsertHandleMMFilesFulltextIndex(idx->_handles, document);
	  if (handle == 0) {
	    TRI_WriteUnlockReadWriteLock(&idx->_lock);
	    return false;
	  }

	  // if words are all different, we must start from the root node. the root node is also the
	  // start for the 1st word inserted
	  paths[0] = idx->_root;
	  lastLength = 0;

	  size_t w = 0;
	  size_t numWords = wordlist.size();
	  while (w < numWords) {
	    node_t* node;
	    char const* p;
	    size_t start;
	    size_t i;

	    // LOG_TOPIC(DEBUG, arangodb::Logger::FIXME) << "checking word " << wordlist->_words[w];

	    if (w > 0) {
	      std::string tmp = wordlist[w];
	      // check if current word has a shared/common prefix with the previous word inserted
	      // in case this is true, we can use an optimisation and do not need to traverse the
	      // tree from the root again. instead, we just start at the node at the end of the
	      // shared/common prefix. this will save us a lot of tree lookups
	      start = CommonPrefixLength(wordlist[w - 1], tmp);
	      if (start > MAX_WORD_BYTES) {
	        start = MAX_WORD_BYTES;
	      }

	      // check if current word is the same as the last word. we do not want to insert the
	      // same word multiple times for the same document
	      if (start > 0 && start == lastLength &&
	          start == tmp.size()) {
	        // duplicate word, skip it and continue with next word
	        w++;
	        continue;
	      }
	    } else {
	      start = 0;
	    }

	    // for words with common prefixes, use the most appropriate start node 
	    // we do not need to traverse the tree from the root again
	    node = paths[start];
	#if TRI_FULLTEXT_DEBUG
	    TRI_ASSERT(node != nullptr);
	#endif

	    // now insert into the tree, starting at the next character after the common prefix
	    std::string tmp = wordlist[w++].substr(start);
	    p = tmp.c_str();

	    for (i = start; *p && i <= MAX_WORD_BYTES; ++i) {
	      node_char_t c = (node_char_t) * (p++);

	#if TRI_FULLTEXT_DEBUG
	      TRI_ASSERT(node != nullptr);
	#endif

	      node = EnsureSubNode(idx, node, c);
	      if (node == nullptr) {
	        TRI_WriteUnlockReadWriteLock(&idx->_lock);
	        return false;
	      }

	#if TRI_FULLTEXT_DEBUG
	      TRI_ASSERT(node != nullptr);
	#endif

	      paths[i + 1] = node;
	    }

	    if (!InsertHandle(idx, node, handle)) {
	      // document was added at least once, mark it as deleted
	      TRI_DeleteDocumentHandleMMFilesFulltextIndex(idx->_handles, document);
	      TRI_WriteUnlockReadWriteLock(&idx->_lock);
	      return false;
	    }

	    // store length of word just inserted
	    // we'll use that to compare with the next word for duplicate removal
	    lastLength = i;
	  }

	  TRI_WriteUnlockReadWriteLock(&idx->_lock);

	  return true;
	}

代码有点复杂，主要是缺少数据结构图。其实不外乎就是对着数据结构的一些操作。

查找的代码：

	/// @brief execute a query on the fulltext index
	/// note: this will free the query
	TRI_fulltext_result_t* TRI_QueryMMFilesFulltextIndex(TRI_fts_index_t* const ftx,
	                                              TRI_fulltext_query_t* query) {
	  index__t* idx;
	  TRI_fulltext_list_t* result;
	  size_t i;

	  /// ..省略一些检查..

	  auto maxResults = query->_maxResults;

	  idx = (index__t*)ftx;

	  TRI_ReadLockReadWriteLock(&idx->_lock);

	  // initial result is empty
	  result = nullptr;

	  // iterate over all words in query
	  for (i = 0; i < query->_numWords; ++i) {
	    char* word;
	    TRI_fulltext_query_match_e match;
	    TRI_fulltext_query_operation_e operation;
	    TRI_fulltext_list_t* list;
	    node_t* node;

	    word = query->_words[i];
	    if (word == nullptr) {
	      break;
	    }

	    match = query->_matches[i];
	    operation = query->_operations[i];

	    LOG_TOPIC(DEBUG, arangodb::Logger::FIXME) << "searching for word: '" << word << "'";

	    if ((operation == TRI_FULLTEXT_AND || operation == TRI_FULLTEXT_EXCLUDE) &&
	        i > 0 && TRI_NumEntriesListMMFilesFulltextIndex(result) == 0) {
	      // current result set is empty so logical AND or EXCLUDE will not have any
	      // result either
	      continue;
	    }

	    list = nullptr;
	    node = FindNode(idx, word, strlen(word));
	    if (node != nullptr) {
	      if (match == TRI_FULLTEXT_COMPLETE) {
	        // complete matching
	        list = GetDirectNodeHandles(node);
	      } else if (match == TRI_FULLTEXT_PREFIX) {
	        // prefix matching
	        list = GetSubNodeHandles(node);
	      } else {
	        LOG_TOPIC(WARN, arangodb::Logger::FIXME) << "invalid matching option for fulltext index query";
	        list = TRI_CreateListMMFilesFulltextIndex(0);
	      }
	    } else {
	      list = TRI_CreateListMMFilesFulltextIndex(0);
	    }

	    if (operation == TRI_FULLTEXT_AND) {
	      // perform a logical AND of current and previous result (if any)
	      result = TRI_IntersectListMMFilesFulltextIndex(result, list);
	    } else if (operation == TRI_FULLTEXT_OR) {
	      // perform a logical OR of current and previous result (if any)
	      result = TRI_UnioniseListMMFilesFulltextIndex(result, list);
	    } else if (operation == TRI_FULLTEXT_EXCLUDE) {
	      // perform a logical exclusion of current from previous result (if any)
	      result = TRI_ExcludeListMMFilesFulltextIndex(result, list);
	    }

	    if (result == nullptr) {
	      // out of memory
	      break;
	    }
	  }

	  TRI_FreeQueryMMFilesFulltextIndex(query);

	  if (result == nullptr) {
	    // if we haven't found anything...
	    TRI_ReadUnlockReadWriteLock(&idx->_lock);
	    return TRI_CreateResultMMFilesFulltextIndex(0);
	  }

	  // now convert the handle list into a result (this will also filter out
	  // deleted documents)
	  TRI_fulltext_result_t* r = MakeListResult(idx, result, maxResults);
	  TRI_ReadUnlockReadWriteLock(&idx->_lock);

	  return r;
	}

对每个词查找对应的节点：

	/// @brief find a node by its key, starting from the index root
	static node_t* FindNode(const index__t* idx, char const* const key,
	                        size_t const keyLength) {
	  node_t* node;
	  node_char_t* p;
	  size_t i;

	  node = (node_t*)idx->_root;
	#if TRI_FULLTEXT_DEBUG
	  TRI_ASSERT(node != nullptr);
	#endif
	  p = (node_char_t*)key;

	  for (i = 0; i < keyLength; ++i) {
	    node = FindDirectSubNode(node, *(p++));
	    if (node == nullptr) {
	      return nullptr;
	    }
	  }

	  return node;
	}

	/// @brief find a node's sub-node, identified by its start character
	static inline node_t* FindDirectSubNode(const node_t* const node,
	                                        const node_char_t c) {
	  uint32_t numFollowers;

	#if TRI_FULLTEXT_DEBUG
	  TRI_ASSERT(node != nullptr);
	#endif

	  numFollowers = NodeNumFollowers(node);

	  if (numFollowers >= 8) {
	    return FindDirectSubNodeBinary(node, c);
	  } else if (numFollowers > 1) {
	    return FindDirectSubNodeLinear(node, c);
	  } else if (numFollowers == 1) {
	    return FindDirectSubNodeSingle(node, c);
	  }

	  return nullptr;
	}

个人感觉ArangoDB的fulltext索引搞得有点复杂了，而且代码有点混乱，不容易看懂。

### 4、Geo Index

Geo Index的实现主要有四个文件:

	arangod/MMFiles/MMFilesGeoIndex.cpp
	arangod/MMFiles/MMFilesGeoIndex.h
	arangod/MMFiles/mmfiles-geo-index.cpp
	arangod/MMFiles/mmfiles-geo-index.h


	class MMFilesGeoIndex final : public Index {

	 public:
	  /// @brief geo index variants
	  enum IndexVariant {
	    INDEX_GEO_NONE = 0,
	    INDEX_GEO_INDIVIDUAL_LAT_LON,
	    INDEX_GEO_COMBINED_LAT_LON,
	    INDEX_GEO_COMBINED_LON_LAT
	  };

	 public:
	  IndexType type() const override {
	    if (_variant == INDEX_GEO_COMBINED_LAT_LON ||
	        _variant == INDEX_GEO_COMBINED_LON_LAT) {
	      return TRI_IDX_TYPE_GEO1_INDEX;
	    }

	    return TRI_IDX_TYPE_GEO2_INDEX;
	  }
	  
	  char const* typeName() const override { 
	    if (_variant == INDEX_GEO_COMBINED_LAT_LON ||
	        _variant == INDEX_GEO_COMBINED_LON_LAT) {
	      return "geo1";
	    }
	    return "geo2";
	  }
	  
	  ...

	  /// @brief looks up all points within a given radius
	  GeoCoordinates* withinQuery(transaction::Methods*, double, double,
	                              double) const;

	  /// @brief looks up the nearest points
	  GeoCoordinates* nearQuery(transaction::Methods*, double, double,
	                            size_t) const;

	  ...

	 private:

	  /// @brief attribute paths
	  std::vector<std::string>  _location;
	  std::vector<std::string>  _latitude;
	  std::vector<std::string>  _longitude;

	  /// @brief the geo index variant (geo1 or geo2)
	  IndexVariant _variant;

	  /// @brief whether the index is a geoJson index (latitude / longitude
	  /// reversed)
	  bool _geoJson;

	  /// @brief the actual geo index
	  GeoIdx* _geoIndex;
	};

最重要的索引结构GeoIdx	定义在 arangod/MMFiles/mmfiles-geo-index.h中，其实就是void类型。。

	typedef void GeoIdx;  /* to keep the structure private  */

mmfiles-geo-index.cpp有非常详细的注释，这里就不展开了。可以参考这篇文章 [Using Hilbert curves and Polyhedrons for Geo-Indexing
](https://www.arangodb.com/2012/03/using-hilbert-curves-and-polyhedrons-for-geo-indexing/) 理解。


### 5、Persistent Index

Persistent Index是[ArangoDB 3.0](https://docs.arangodb.com/3.0/Manual/ReleaseNotes/NewFeatures30.html#persistent-indexes)才作为实验性feature提供的功能。

主要是为了解决原来纯内存索引带来的一些问题：

1. 机器成本：内存成本显而易见，但是速度确实要比磁盘要快很多，这是一个trade off。
2. 加载速度：每次加载Collection都需要重新构建索引

这一次ArangoDB终于没有从零自己实现了，而在[RocksDB](http://rocksdb.org/)存储引擎的基础上实现了Persistent Index。RocksDB是Facebook开源的一个嵌入式持久化的K-V存储引擎，是LevelDB的一个改进版。相对于传统的K-V而言，RocksDB的key是sorted，所以它还可以支持sorted index，实现范围查询和排序操作。

Persistent Index相关的文件有6个：

	arangod/MMFiles/MMFilesPersistentIndex.cpp
	arangod/MMFiles/MMFilesPersistentIndex.h
	arangod/MMFiles/MMFilesPersistentIndexFeature.cpp
	arangod/MMFiles/MMFilesPersistentIndexFeature.h
	arangod/MMFiles/MMFilesPersistentIndexKeyComparator.cpp
	arangod/MMFiles/MMFilesPersistentIndexKeyComparator.h

其中 MMFilesPersistentIndexKeyComparator 实现了 rocksdb::Comparator 接口，因为RocksDB是sorted KV；MMFilesPersistentIndexFeature 则是定义了一些配置信息。关键实现还是在 MMFilesPersistentIndex 类。

PersistentIndex类含有RocksDB的OptimisticTransactionDB的引用：

	/// @brief create the index
	PersistentIndex::PersistentIndex(TRI_idx_iid_t iid,
	                           arangodb::LogicalCollection* collection,
                           	   arangodb::velocypack::Slice const& info)
	    : MMFilesPathBasedIndex(iid, collection, info, 0, true),
	      _db(RocksDBFeature::instance()->db()) {}

我们还是主要分析一下索引的插入和查找过程。先看一下索引插入过程：

	/// @brief inserts a document into the index
	int PersistentIndex::insert(arangodb::Transaction* trx, TRI_voc_rid_t revisionId,
	                         VPackSlice const& doc, bool isRollback) {
	  auto comparator = RocksDBFeature::instance()->comparator();
	  std::vector<MMFilesSkiplistIndexElement*> elements;

	  int res;
	  try {
	  	/// 1. fillElement的职责是将doc转换成MMFilesIndexElement对象，这里是转成了MMFilesSkiplistIndexElement
	    res = fillElement(elements, revisionId, doc);
	  } catch (...) {
	    res = TRI_ERROR_OUT_OF_MEMORY;
	  }

	  ...
	  
	  ManagedDocumentResult result; 
	  IndexLookupContext context(trx, _collection, &result, numPaths()); 

	  VPackSlice const key = Transaction::extractKeyFromDocument(doc);
	  std::string const prefix =
	      buildPrefix(trx->vocbase()->id(), _collection->cid(), _iid);

	  VPackBuilder builder;
	  std::vector<std::string> values;
	  values.reserve(elements.size());

	  // lower and upper bounds, only required if the index is unique
	  std::vector<std::pair<std::string, std::string>> bounds;
	  if (_unique) {
	    bounds.reserve(elements.size());
	  }

 	  /// 2. 对elements的每一个元素进行序列化
	  for (auto const& it : elements) {
	    builder.clear();
	    builder.openArray();
	    /// 对每一个field进行序列化
	    for (size_t i = 0; i < _fields.size(); ++i) {
	      builder.add(it->slice(&context, i));
	    }
	    builder.add(key); // always append _key value to the end of the array
	    builder.close();

	    VPackSlice const s = builder.slice();
	    std::string value;
	    value.reserve(keyPrefixSize() + s.byteSize());
	    value += prefix;
	    /// value的存储格式是 prefix + data + key
	    value.append(s.startAs<char const>(), s.byteSize());
	    values.emplace_back(std::move(value));

	    if (_unique) {
	      builder.clear();
	      builder.openArray();
	      for (size_t i = 0; i < _fields.size(); ++i) {
	        builder.add(it->slice(&context, i));
	      }
	      builder.add(VPackSlice::minKeySlice());
	      builder.close();
	    
	      VPackSlice s = builder.slice();
	      std::string value;
	      value.reserve(keyPrefixSize() + s.byteSize());
	      value += prefix;
	      value.append(s.startAs<char const>(), s.byteSize());
	      
	      std::pair<std::string, std::string> p;
	      p.first = value;
	      
	      builder.clear();
	      builder.openArray();
	      for (size_t i = 0; i < _fields.size(); ++i) {
	        builder.add(it->slice(&context, i));
	      }
	      builder.add(VPackSlice::maxKeySlice());
	      builder.close();
	    
	      s = builder.slice();
	      value.clear();
	      value += prefix;
	      value.append(s.startAs<char const>(), s.byteSize());
	      
	      p.second = value;
	      bounds.emplace_back(std::move(p));
	    }
	  }

	  auto rocksTransaction = trx->rocksTransaction();
	  TRI_ASSERT(rocksTransaction != nullptr);

	  rocksdb::ReadOptions readOptions;

	  size_t const count = elements.size();
	  for (size_t i = 0; i < count; ++i) {
	    if (_unique) {
	      bool uniqueConstraintViolated = false;
	      auto iterator = rocksTransaction->GetIterator(readOptions);

	      if (iterator != nullptr) {
	        auto& bound = bounds[i];
	        iterator->Seek(rocksdb::Slice(bound.first.c_str(), bound.first.size()));

	        while (iterator->Valid()) {
	          int res = comparator->Compare(iterator->key(), rocksdb::Slice(bound.second.c_str(), bound.second.size()));

	          if (res > 0) {
	            break;
	          }

	          uniqueConstraintViolated = true;
	          break;
	        }

	        delete iterator;
	      }

	      if (uniqueConstraintViolated) {
	        // duplicate key
	        res = TRI_ERROR_ARANGO_UNIQUE_CONSTRAINT_VIOLATED;
	        if (!_collection->useSecondaryIndexes()) {
	          // suppress the error during recovery
	          res = TRI_ERROR_NO_ERROR;
	        }
	      }
	    }

	    /// 3. 将前面序列化得到的vector<string> values逐个put到RocksDB中
	    if (res == TRI_ERROR_NO_ERROR) {
	      auto status = rocksTransaction->Put(values[i], std::string());
	      
	      if (! status.ok()) {
	        res = TRI_ERROR_INTERNAL;
	      }
	    }

	    if (res != TRI_ERROR_NO_ERROR) {
	      for (size_t j = 0; j < i; ++j) {
	        rocksTransaction->Delete(values[i]);
	      }
	    
	      if (res == TRI_ERROR_ARANGO_UNIQUE_CONSTRAINT_VIOLATED && !_unique) {
	        // We ignore unique_constraint violated if we are not unique
	        res = TRI_ERROR_NO_ERROR;
	      }
	      break;
	    }
	  }

	  return res;
	}

可以看到基本就是三个操作：

1. 通过fillElement将doc转换成MMFilesIndexElement对象，这里是转成了MMFilesSkiplistIndexElement
2. 对elements的每一个元素进行序列化
3. 将前面序列化得到的vector<string> values逐个put到RocksDB中

fillElement方法在前面已经介绍过了，这里不赘述。接着对elements进行序列化，得到vector<string> values，再调用rocksTransaction.Put方法插入到RocksDB。rocksTransaction是 rocksdb::Transaction，定义在 3rdParty/rocksdb/v5.1.4/include/rocksdb/utilities/transaction.h，实现类是 TransactionBaseImpl，定义在3rdParty/rocksdb/v5.1.4/utilities/transactions/transaction_base.h :

	Status Put(ColumnFamilyHandle* column_family, const Slice& key,
	           const Slice& value) override;
	Status Put(const Slice& key, const Slice& value) override {
	  return Put(nullptr, key, value);
	}

	Status Put(ColumnFamilyHandle* column_family, const SliceParts& key,
	           const SliceParts& value) override;
	Status Put(const SliceParts& key, const SliceParts& value) override {
	  return Put(nullptr, key, value);
	}	

实现类在 arangodb/3rdParty/rocksdb/v5.1.4/utilities/transactions/transaction_base.cc 中：

	Status TransactionBaseImpl::Put(ColumnFamilyHandle* column_family,
                                const Slice& key, const Slice& value) {
	  Status s =
	      TryLock(column_family, key, false /* read_only */, true /* exclusive */);

	  if (s.ok()) {
	    GetBatchForWrite()->Put(column_family, key, value);
	    num_puts_++;
	  }

	  return s;
	}  

	Status TransactionBaseImpl::Put(ColumnFamilyHandle* column_family,
	                                const SliceParts& key,
	                                const SliceParts& value) {
	  Status s =
	      TryLock(column_family, key, false /* read_only */, true /* exclusive */);

	  if (s.ok()) {
	    GetBatchForWrite()->Put(column_family, key, value);
	    num_puts_++;
	  }

	  return s;
	}

其中：GetBatchForWrite() 返回的是 WriteBatchBase 对象，有两个具体的实现：

* WriteBatch: 不添加索引
* WrithBatchWithIndex：会同时对key创建索引

根据indexing_enabled_判断返回哪个：

	// Gets the write batch that should be used for Put/Merge/Deletes.
	//
	// Returns either a WriteBatch or WriteBatchWithIndex depending on whether
	// DisableIndexing() has been called.
	WriteBatchBase* TransactionBaseImpl::GetBatchForWrite() {
	  if (indexing_enabled_) {
	    // Use WriteBatchWithIndex
	    return &write_batch_;
	  } else {
	    // Don't use WriteBatchWithIndex. Return base WriteBatch.
	    return write_batch_.GetWriteBatch();
	  }
	}

indexing_enabled_默认是true，所以这里是构建索引的 WriteBatchWithIndex 类，定义在3rdParty/rocksdb/v5.1.4/utilities/write_batch_with_index/write_batch_with_index.cc：

	void WriteBatchWithIndex::Put(ColumnFamilyHandle* column_family,
	                              const Slice& key, const Slice& value) {
	  rep->SetLastEntryOffset(); /// 记录当前的位置
	  rep->write_batch.Put(column_family, key, value); /// 调用无索引的WriteBatch类插入数据
	  rep->AddOrUpdateIndex(column_family, key); /// 对key进行索引构建
	}

	void WriteBatchWithIndex::Put(const Slice& key, const Slice& value) {
	  rep->SetLastEntryOffset();
	  rep->write_batch.Put(key, value);
	  rep->AddOrUpdateIndex(key);
	}

所有的操作都委派给类成员变量 `struct Rep *rep` 去了，这个类的定义就在同一个文件中：

	struct WriteBatchWithIndex::Rep {
	  Rep(const Comparator* index_comparator, size_t reserved_bytes = 0,
	      bool _overwrite_key = false)
	      : write_batch(reserved_bytes),
	        comparator(index_comparator, &write_batch),
	        skip_list(comparator, &arena),
	        overwrite_key(_overwrite_key),
	        last_entry_offset(0) {}
	  ReadableWriteBatch write_batch;
	  WriteBatchEntryComparator comparator;
	  Arena arena;
	  WriteBatchEntrySkipList skip_list;
	  bool overwrite_key;
	  size_t last_entry_offset;

	  // Remember current offset of internal write batch, which is used as
	  // the starting offset of the next record.
	  void SetLastEntryOffset() { last_entry_offset = write_batch.GetDataSize(); }

	  // In overwrite mode, find the existing entry for the same key and update it
	  // to point to the current entry.
	  // Return true if the key is found and updated.
	  bool UpdateExistingEntry(ColumnFamilyHandle* column_family, const Slice& key);
	  bool UpdateExistingEntryWithCfId(uint32_t column_family_id, const Slice& key);

	  // Add the recent entry to the update.
	  // In overwrite mode, if key already exists in the index, update it.
	  void AddOrUpdateIndex(ColumnFamilyHandle* column_family, const Slice& key);
	  void AddOrUpdateIndex(const Slice& key);

	  // Allocate an index entry pointing to the last entry in the write batch and
	  // put it to skip list.
	  void AddNewEntry(uint32_t column_family_id);

	  // Clear all updates buffered in this batch.
	  void Clear();
	  void ClearIndex();

	  // Rebuild index by reading all records from the batch.
	  // Returns non-ok status on corruption.
	  Status ReBuildIndex();
	};

过程非常简单明了：

	/// 记录当前的位置
	rep->SetLastEntryOffset();
	/// 调用无索引的WriteBatch类插入数据
    rep->write_batch.Put(column_family, key, value); 
    /// 对key进行索引构建
    rep->AddOrUpdateIndex(column_family, key); 

我们看一下索引是怎么构建的：

	void WriteBatchWithIndex::Rep::AddOrUpdateIndex(
	    ColumnFamilyHandle* column_family, const Slice& key) {
	  if (!UpdateExistingEntry(column_family, key)) {
	    uint32_t cf_id = GetColumnFamilyID(column_family);
	    const auto* cf_cmp = GetColumnFamilyUserComparator(column_family);
	    if (cf_cmp != nullptr) {
	      comparator.SetComparatorForCF(cf_id, cf_cmp);
	    }
	    AddNewEntry(cf_id);
	  }
	}

	void WriteBatchWithIndex::Rep::AddOrUpdateIndex(const Slice& key) {
	  if (!UpdateExistingEntryWithCfId(0, key)) {
	    AddNewEntry(0);
	  }
	}

	// Allocate an index entry pointing to the last entry in the write batch and
    // put it to skip list.
	void WriteBatchWithIndex::Rep::AddNewEntry(uint32_t column_family_id) {
	  const std::string& wb_data = write_batch.Data();
	  Slice entry_ptr = Slice(wb_data.data() + last_entry_offset,
	                          wb_data.size() - last_entry_offset);
	  // Extract key
	  Slice key;
	  bool success __attribute__((__unused__)) =
	      ReadKeyFromWriteBatchEntry(&entry_ptr, &key, column_family_id != 0);
	  assert(success);

	    auto* mem = arena.Allocate(sizeof(WriteBatchIndexEntry));
	    auto* index_entry =
	        new (mem) WriteBatchIndexEntry(last_entry_offset, column_family_id,
	                                       key.data() - wb_data.data(), key.size());
	    skip_list.Insert(index_entry);
	  }

看起来也蛮简单的，就是生产index_entry，插入到skip_list中。WriteBatchIndexEntry 是 skip_list的key，定义在 3rdParty/rocksdb/v5.1.4/utilities/write_batch_with_index/write_batch_with_index_internal.h：

	// Key used by skip list, as the binary searchable index of WriteBatchWithIndex.
	struct WriteBatchIndexEntry {
	  WriteBatchIndexEntry(size_t o, uint32_t c, size_t ko, size_t ksz)
	      : offset(o),
	        column_family(c),
	        key_offset(ko),
	        key_size(ksz),
	        search_key(nullptr) {}
	  WriteBatchIndexEntry(const Slice* sk, uint32_t c)
	      : offset(0),
	        column_family(c),
	        key_offset(0),
	        key_size(0),
	        search_key(sk) {}

	  // If this flag appears in the offset, it indicates a key that is smaller
	  // than any other entry for the same column family
	  static const size_t kFlagMin = port::kMaxSizet;

	  size_t offset;           // offset of an entry in write batch's string buffer.
	  uint32_t column_family;  // column family of the entry.
	  size_t key_offset;       // offset of the key in write batch's string buffer.
	  size_t key_size;         // size of the key.

	  const Slice* search_key;  // if not null, instead of reading keys from
	                            // write batch, use it to compare. This is used
	                            // for lookup key.
	};

而skip_list是Rep的一个成员变量，类型是WriteBatchEntrySkipList：

	typedef SkipList<WriteBatchIndexEntry*, const WriteBatchEntryComparator&>
		    WriteBatchEntrySkipList;

SkipList类定义在 arangodb/3rdParty/rocksdb/v5.1.4/db/skiplist.h 中：

	namespace rocksdb {

	template<typename Key, class Comparator>
	class SkipList {
	 private:
	  struct Node;
	}
	
	...

	}  // namespace rocksdb

这是一个标准的数据结构，这里就不赘述了，感兴趣的同学自己可以研究。

细心的读者可能会发现这段代码逻辑跟前面介绍的 MMFilesSkiplistIndex 的insert操作非常的类似：

	/// @brief inserts a document into a skiplist index
	int MMFilesSkiplistIndex::insert(transaction::Methods* trx, TRI_voc_rid_t revisionId, 
	                          VPackSlice const& doc, bool isRollback);

两者都把document转换成 vector<MMFilesSkiplistIndexElement>，然后插入到Skiplist中。
不同的地方在于 Persistent Index 先把 MMFilesSkiplistIndexElement 序列化成 [arangodb/velocypack](https://github.com/arangodb/velocypack/blob/master/include/velocypack/Slice.h) 数据格式，再调用rocksdb::Transaction的Put方法，将数据插入到RocksDB中去，而索引的构建交给了RocksDB（底层索引结构恰好也是skiplist）。

不过这里没有搞明白Persistent Index要先转成MMFilesSkiplistIndexElement，Skiplist是rocksDB的底层实现，只要把要索引的key插入就可以了啊？
还有这一行看起来貌似value为空字符串？

	auto status = rocksTransaction->Put(values[i], std::string());

最后我们来看一下查找过程。

	/// @brief attempts to locate an entry in the index
	/// Warning: who ever calls this function is responsible for destroying
	/// the PersistentIndexIterator* results
	PersistentIndexIterator* PersistentIndex::lookup(arangodb::Transaction* trx,
	                                      ManagedDocumentResult* mmdr,
	                                      VPackSlice const searchValues,
	                                      bool reverse) const {
	  TRI_ASSERT(searchValues.isArray());
	  TRI_ASSERT(searchValues.length() <= _fields.size());

	  VPackBuilder leftSearch;
	  VPackBuilder rightSearch;

	  VPackSlice lastNonEq;
	  leftSearch.openArray();
	  for (auto const& it : VPackArrayIterator(searchValues)) {
	    TRI_ASSERT(it.isObject());
	    VPackSlice eq = it.get(StaticStrings::IndexEq);
	    if (eq.isNone()) {
	      lastNonEq = it;
	      break;
	    }
	    leftSearch.add(eq);
	  }

	  VPackSlice leftBorder;
	  VPackSlice rightBorder;

	  if (lastNonEq.isNone()) {
	    // We only have equality!
	    rightSearch = leftSearch;

	    leftSearch.add(VPackSlice::minKeySlice());
	    leftSearch.close();
	    
	    rightSearch.add(VPackSlice::maxKeySlice());
	    rightSearch.close();

	    leftBorder = leftSearch.slice();
	    rightBorder = rightSearch.slice();
	  } else {
	    // Copy rightSearch = leftSearch for right border
	    rightSearch = leftSearch;

	    // Define Lower-Bound 
	    VPackSlice lastLeft = lastNonEq.get(StaticStrings::IndexGe);
	    if (!lastLeft.isNone()) {
	      TRI_ASSERT(!lastNonEq.hasKey(StaticStrings::IndexGt));
	      leftSearch.add(lastLeft);
	      leftSearch.add(VPackSlice::minKeySlice());
	      leftSearch.close();
	      VPackSlice search = leftSearch.slice();
	      leftBorder = search;
	    } else {
	      lastLeft = lastNonEq.get(StaticStrings::IndexGt);
	      if (!lastLeft.isNone()) {
	        leftSearch.add(lastLeft);
	        leftSearch.add(VPackSlice::maxKeySlice());
	        leftSearch.close();
	        VPackSlice search = leftSearch.slice();
	        leftBorder = search;
	      } else {
	        // No lower bound set default to (null <= x)
	        leftSearch.add(VPackSlice::minKeySlice());
	        leftSearch.close();
	        VPackSlice search = leftSearch.slice();
	        leftBorder = search;
	      }
	    }

	    // Define upper-bound
	    VPackSlice lastRight = lastNonEq.get(StaticStrings::IndexLe);
	    if (!lastRight.isNone()) {
	      TRI_ASSERT(!lastNonEq.hasKey(StaticStrings::IndexLt));
	      rightSearch.add(lastRight);
	      rightSearch.add(VPackSlice::maxKeySlice());
	      rightSearch.close();
	      VPackSlice search = rightSearch.slice();
	      rightBorder = search;
	    } else {
	      lastRight = lastNonEq.get(StaticStrings::IndexLt);
	      if (!lastRight.isNone()) {
	        rightSearch.add(lastRight);
	        rightSearch.add(VPackSlice::minKeySlice());
	        rightSearch.close();
	        VPackSlice search = rightSearch.slice();
	        rightBorder = search;
	      } else {
	        // No upper bound set default to (x <= INFINITY)
	        rightSearch.add(VPackSlice::maxKeySlice());
	        rightSearch.close();
	        VPackSlice search = rightSearch.slice();
	        rightBorder = search;
	      }
	    }
	  }

	  // Secured by trx. The shared_ptr index stays valid in
	  // _collection at least as long as trx is running.
	  // Same for the iterator
	  auto idx = _collection->primaryIndex();
	  return new PersistentIndexIterator(_collection, trx, mmdr, this, idx, _db, reverse, leftBorder, rightBorder);
	}

什么鬼。。完全看不懂。。


参考文章
-------

1. [Handling Indexes](https://docs.arangodb.com/3.1/Manual/Indexing/)
2. [Suggestion: Have indexes also on disk besides "only memory indexes" #209](https://github.com/arangodb/arangodb/issues/209#issuecomment-193838232)
3. [Frequently asked questions](https://www.arangodb.com/documentation/faq/#what-are-the-server-requirements-for-arangodb)
