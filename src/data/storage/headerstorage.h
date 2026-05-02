#ifndef HEADERSTORAGE_H
#define HEADERSTORAGE_H

#include <QObject>
#include <QVariant>
#include <QAnyStringView>

#include <data/bcglobals.h>


/*!
 * Namespace for HeaderStorage keys
 */
namespace BC::Store {
}

/*!
 * \brief Base class for objects that contribute fields to an experiment's
 * CSV header file.
 *
 * Every Blackchirp experiment is described on disk by a six-column,
 * semicolon-delimited \c header.csv file:
 *
 * | Column     | Meaning                                                       |
 * |------------|---------------------------------------------------------------|
 * | Object Key | Identifier of the producing HeaderStorage object              |
 * | Array Key  | Name of the array this row belongs to (empty if not an array) |
 * | Array Idx  | Index within the array (empty if not an array)                |
 * | Key        | The setting's key                                             |
 * | Value      | Stored value, formatted as a string                           |
 * | Unit       | Unit of the value (empty if dimensionless)                    |
 *
 * HeaderStorage subclasses populate a small in-memory cache of these
 * rows on the way out (write) and consume rows from it on the way in
 * (read). Rows are dispatched between objects by the Object Key — each
 * subclass picks a key in its constructor (typically a constant from
 * the BC::Store namespace, or, for hardware-instance objects, the
 * hardware key like \c "PulseGenerator.main") and only rows whose first
 * column matches that key end up in this object's cache.
 *
 * # The two virtuals you must implement
 *
 * Subclasses override two pure virtuals:
 *
 * - storeValues() runs just before the header is written. Inside it,
 *   call store() once per scalar field and storeArrayValue() once per
 *   cell of any array fields. Each row is buffered in this object's
 *   cache.
 * - retrieveValues() runs after the header has been parsed and all
 *   matching rows have been routed to this object. Inside it, call
 *   retrieve() and retrieveArrayValue() to extract the cached values
 *   into your own members.
 *
 * Each call to retrieve() or retrieveArrayValue() *removes* the row
 * from the cache. Asking for the same key twice yields the default the
 * second time. arrayStoreSize() reports how many entries an array has
 * before you start consuming it.
 *
 * # Composing a tree
 *
 * A HeaderStorage may own children: Experiment is the root, with
 * FtmwConfig, validators, hardware configs, the LIF config, etc. as
 * children, each of which may add its own grandchildren. Children are
 * declared by overriding prepareChildren() and calling addChild() once
 * per child:
 *
 *     void Experiment::prepareChildren()
 *     {
 *         addChild(ps_ftmwConfig.get());
 *         addChild(ps_validator.get());
 *         for(auto &p : d_hwCfgs)
 *             addChild(p.get());
 *         addChild(ps_lifCfg.get());
 *     }
 *
 * The framework calls prepareChildren() at the start of every read or
 * write pass, after wiping any prior child list, so it always reflects
 * the current shape of the tree. Children themselves do not call
 * addChild on their parent — the parent owns the relationship in
 * prepareChildren(). The framework recurses automatically: each
 * child's prepareChildren() is called next, allowing arbitrary
 * nesting.
 *
 * removeChild() exists for the rare case of a parent that needs to
 * detach a child mid-flight (Experiment uses it when the user disables
 * an FTMW or LIF subsystem). It does not delete the child object;
 * ownership lives with whoever holds the smart pointer.
 *
 * # Write flow (subclass perspective)
 *
 * 1. A caller invokes getStrings() on the root (Experiment does this
 *    inside save() via BlackchirpCSV::writeHeader).
 * 2. Each node's prepareToStore() runs, which calls prepareChildren()
 *    and recurses into each child.
 * 3. Each node's storeValues() runs, populating its cache via store()
 *    and storeArrayValue().
 * 4. The framework converts cached entries to the six-column form,
 *    merges in each child's getStrings() output, and clears the
 *    in-memory cache.
 *
 * Therefore: never call store() outside storeValues(); the rows would
 * be cleared on the next write.
 *
 * # Read flow (subclass perspective)
 *
 * 1. The caller (Experiment's reading constructor) calls
 *    prepareToStore() on the root once, so the child tree is built.
 * 2. The caller reads the CSV file line by line and hands each row to
 *    storeLine() on the root. Rows are dispatched to whichever node's
 *    \c d_headerKey matches column 0 (children searched first if the
 *    root does not match).
 * 3. After all lines have been routed, the caller invokes
 *    readComplete() on the root, which calls retrieveValues() on
 *    every node depth-first.
 * 4. Each retrieveValues() implementation pulls rows back out via
 *    retrieve() / retrieveArrayValue() and assigns them to the
 *    object's members.
 *
 * Therefore: never call retrieve() outside retrieveValues() (or after
 * readComplete() has run) — the cache is empty by then.
 *
 * # Object-key conventions
 *
 * - Singleton-style objects (Experiment, RfConfig, ChirpConfig, etc.)
 *   pass a constant key from their \c BC::Store::* namespace.
 * - Per-instance objects (PulseGenConfig and other hardware configs)
 *   pass the hardware key for the specific instance
 *   (e.g., \c "PulseGenerator.main"). This guarantees that experiments
 *   with several instances of the same hardware type produce
 *   distinguishable header rows.
 *
 * The chosen key is stored in d_headerKey and cannot be changed
 * afterward.
 */
class HeaderStorage
{
public:
    using ValueUnit = std::pair<QVariant,QString>; /*!< Alias for storing a value and unit */
    using HeaderMap = std::map<QString,ValueUnit,std::less<>>; /*!< Alias for a map of key-value+unit pairs */
    using HeaderArray = std::vector<HeaderMap>; /*!< Alias for a list of HeaderMap values */
    using HeaderStrings = std::multimap<QString,std::tuple<QString,QString,QString,QString,QString>>; /*!< Alias for a set of strings representing header data, together with the object key */

    /*!
     * \brief Constructor. Sets the object key, which should be unique to the
     * most derived class.
     *
     * \param objKey The object key written into the first column of every
     * header line produced by this object. Used by storeLine() to dispatch
     * incoming lines to the correct HeaderStorage in the parent/child tree.
     */
    HeaderStorage(QAnyStringView objKey);
    
    /*!
     * \brief Destructor. Does nothing
     */
    virtual ~HeaderStorage() {}

    /*!
     * \brief This node's object key (column 0 of the CSV).
     * \return The key set in the constructor.
     */
    QString headerKey() const { return d_headerKey; }
    

protected:

    QString d_headerKey; /*!< Object key (column 0 of the CSV); set in the constructor and treated as immutable. */

    /*!
     * \brief Populate this object's cache with rows to be written.
     *
     * Subclasses must implement. Inside, call store() once per scalar
     * field and storeArrayValue() once per cell of any array fields.
     * Do not call store() / storeArrayValue() from anywhere else;
     * rows added outside this function are cleared at the next
     * getStrings() call.
     */
    virtual void storeValues() =0;

    /*!
     * \brief Consume cached rows back into this object's members.
     *
     * Subclasses must implement. Called once per node, depth-first,
     * after every header row has been routed by storeLine(). Inside,
     * call retrieve() / retrieveArrayValue() (each call removes the
     * row from the cache) and assign the values back to your own
     * data members. Use arrayStoreSize() to learn how many entries an
     * array contains before iterating it.
     */
    virtual void retrieveValues() =0;



    /*!
     * \brief Cache one scalar row for writing. Templated overload.
     *
     * Call from storeValues(). Equivalent to the QVariant overload but
     * accepts any type that QVariant can wrap.
     *
     * \param key Row key (column 4 of the CSV).
     * \param val Value (column 5).
     * \param unit Optional unit (column 6).
     */
    template<typename T>
    void store(QAnyStringView key, const T& val, QAnyStringView unit = {}) {
        store(key,QVariant::fromValue(val),unit);
    }

    /*!
     * \brief Cache one scalar row for writing.
     *
     * Call from storeValues(). The row is added to this object's
     * cache and emitted by the next getStrings() call. Repeated calls
     * with the same key overwrite.
     *
     * \param key Row key (column 4 of the CSV).
     * \param val Value (column 5).
     * \param unit Optional unit (column 6).
     */
    void store(QAnyStringView key, const QVariant val, QAnyStringView unit = {});

    /*!
     * \brief Cache one cell of an array for writing. Templated overload.
     *
     * Equivalent to the QVariant overload but accepts any type that
     * QVariant can wrap.
     *
     * \param arrayKey Array name (column 2 of the CSV).
     * \param index Row index within the array (column 3).
     * \param key Cell key (column 4).
     * \param val Cell value (column 5).
     * \param unit Optional unit (column 6).
     */
    template<typename T>
    void storeArrayValue(QAnyStringView arrayKey, std::size_t index, QAnyStringView key, const T& val, QAnyStringView unit = {}) {
        storeArrayValue(arrayKey,index,key,QVariant::fromValue(val),unit);
    }

    /*!
     * \brief Cache one cell of an array for writing.
     *
     * Call from storeValues() once per cell. Use array storage for any
     * list-like collection (e.g. one row per pulse-generator channel).
     * If \a arrayKey is new, the array is created; if \a index exceeds
     * the current size, the array grows to fit. A typical loop:
     *
     *     for(std::size_t i = 0; i < d_channels.size(); ++i) {
     *         const auto &c = d_channels[i];
     *         storeArrayValue(channelArr, i, name,    c.name);
     *         storeArrayValue(channelArr, i, delay,   c.delay,   "us"_L1);
     *         storeArrayValue(channelArr, i, enabled, c.enabled);
     *     }
     *
     * \param arrayKey Array name (column 2 of the CSV).
     * \param index Row index within the array (column 3).
     * \param key Cell key (column 4).
     * \param val Cell value (column 5).
     * \param unit Optional unit (column 6).
     */
    void storeArrayValue(QAnyStringView arrayKey, std::size_t index, QAnyStringView key, const QVariant val, QAnyStringView unit = {});

    /*!
     * \brief Pull one scalar row out of the cache.
     *
     * Call from retrieveValues(). Removes the row from the cache; a
     * second retrieve with the same key returns \a defaultValue.
     * Convert from QVariant via QVariant::value, so any default value
     * type that round-trips through QVariant is allowed.
     *
     * \param key The key matched against column 4 of the cached row.
     * \param defaultValue Returned if the key is missing or has
     * already been retrieved. Defaults to a default-constructed T.
     * \return The retrieved value, or \a defaultValue.
     */
    template<typename T>
    T retrieve(QAnyStringView key, const T& defaultValue = QVariant().value<T>()) {
        auto out = defaultValue;
        auto it = d_values.find(key);
        if(it != d_values.end())
        {
            out = it->second.first.value<T>();
            d_values.erase(it);
        }

        return out;
    }

    /*!
     * \brief Number of entries in a cached array.
     *
     * Use this from retrieveValues() to size your loop before pulling
     * values out with retrieveArrayValue().
     *
     * \param key Array name to look up.
     * \return Number of entries, or 0 if the array was not stored.
     */
    std::size_t arrayStoreSize(QAnyStringView key) const;

    /*!
     * \brief Pull one cell out of a cached array.
     *
     * Call from retrieveValues(). Removes the cell from the cache; the
     * containing entry's outer slot is preserved (so arrayStoreSize()
     * does not change), but a second retrieve of the same cell returns
     * \a defaultValue.
     *
     * \param arrayKey Array name.
     * \param index Row index within the array.
     * \param key Cell key.
     * \param defaultValue Returned if the array is missing, the index
     * is out of range, or the cell has already been retrieved.
     * \return The retrieved value, or \a defaultValue.
     */
    template<typename T>
    T retrieveArrayValue(QAnyStringView arrayKey, std::size_t index, QAnyStringView key,
                            const T& defaultValue = QVariant().value<T>()) {
        auto out = defaultValue;

        auto it = d_arrayValues.find(arrayKey);
        if(it != d_arrayValues.end())
        {
            if(index < it->second.size())
            {
                HeaderMap &m = it->second[index];
                auto it2 = m.find(key);
                if(it2 != m.end())
                {
                    out = it2->second.first.value<T>();
                    m.erase(it2);
                }
            }
        }

        return out;
    }

public:
    /*!
     * \brief Drive the write pass: build the child tree, populate caches,
     * collect every node's rows, and clear the caches.
     *
     * Called by the experiment writer (BlackchirpCSV::writeHeader) on
     * the root HeaderStorage. Internally:
     *
     * -# calls prepareToStore() to (re)build the child tree;
     * -# calls this object's storeValues() to fill its cache;
     * -# packs every cached row into the six-column form (object key,
     *    array key, array index, key, value, unit);
     * -# recursively merges every child's getStrings() result;
     * -# clears this object's cache so a subsequent write starts clean.
     *
     * Subclasses normally never call this directly; only the writer
     * does. They participate by overriding storeValues() and
     * prepareChildren().
     *
     * \return Multimap of cached rows keyed by the producing object's
     * d_headerKey. Each value is a five-string tuple
     * (array key, array index, key, value, unit), any of which may be
     * empty.
     */
    HeaderStrings getStrings();

    /*!
     * \brief Build the child tree (recursive).
     *
     * Clears this node's child list, then calls prepareChildren() so
     * the subclass can repopulate it via addChild(), then recurses
     * into every newly added child. Called automatically at the start
     * of a read pass and from getStrings() at the start of a write
     * pass.
     */
    void prepareToStore();

    /*!
     * \brief Route one parsed CSV row into the appropriate node's cache.
     *
     * Called by the experiment reader once per CSV line, after the
     * line has been parsed into six QVariants. If \a l 's first entry
     * matches d_headerKey, the row is added to this object's cache;
     * otherwise children are searched depth-first. Empty key or value
     * cells cause the row to be dropped silently.
     *
     * The caller (Experiment's reading constructor) is responsible
     * for verifying \a l has exactly six entries before invoking this
     * function — there is no length check inside.
     *
     * \param l Six-element list parsed from a CSV row:
     *   -# Object Key
     *   -# Array Key (empty for scalar rows)
     *   -# Array Index (empty for scalar rows)
     *   -# Key
     *   -# Value
     *   -# Unit (may be empty)
     * \return True if the row was claimed by this node or any of its
     * descendants; false if no matching object key was found.
     */
    bool storeLine(const QVariantList l);

    /*!
     * \brief Drive the read pass: invoke retrieveValues() depth-first.
     *
     * Called by the experiment reader on the root HeaderStorage after
     * every line has been routed via storeLine(). This object's
     * retrieveValues() runs first, then every child's readComplete()
     * is called in turn. By the time control returns to the reader,
     * each subclass has consumed its cached rows back into its own
     * members.
     */
    void readComplete();

    /*!
     * \brief Register a child for this node. Call from prepareChildren().
     *
     * Children are stored as raw pointers; ownership is not transferred.
     * The caller (typically the parent's prepareChildren()) must
     * guarantee that the child outlives the read or write pass currently
     * in progress. Passing nullptr is safe and ignored.
     *
     * \param other Pointer to the child HeaderStorage.
     */
    void addChild(HeaderStorage *other);

    /*!
     * \brief Detach a child without deleting it.
     *
     * Used by parents that disable a subsystem during the lifetime of
     * the parent (Experiment uses it when an FTMW or LIF subsystem is
     * removed mid-flight). After removal the child no longer
     * participates in subsequent read/write passes; ownership of the
     * child object is unaffected.
     *
     * \param child Child to detach.
     * \return The detached child pointer, or nullptr if not found.
     */
    HeaderStorage* removeChild(HeaderStorage *child);


    /*!
     * \brief Declare this node's children. Override in subclasses with
     * children.
     *
     * Called every time prepareToStore() runs, after the prior child
     * list has been wiped. Inside, call addChild() once per child.
     * The default implementation declares no children.
     *
     * Repopulating the list every pass (rather than once at
     * construction) means children that come and go with user choices
     * — a freshly disabled hardware subsystem, an optional validator —
     * are reflected automatically.
     */
    virtual void prepareChildren() {}

private:
    HeaderMap d_values; /*!< Map containing key-value pairs */
    std::map<QString,HeaderArray,std::less<>> d_arrayValues; /*!< Map containing lists of key-value pairs */
    std::vector<HeaderStorage*> d_children; /*!< List containing pointers to children */

};

#endif // HEADERSTORAGE_H
