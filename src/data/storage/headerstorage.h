#ifndef HEADERSTORAGE_H
#define HEADERSTORAGE_H

#include <QObject>
#include <QVariant>

/*!
 * \brief Base class for objects that read/write to an experiment header file
 *
 * This class provides a common interface to an experiment header file (which itself is in csv
 * format). The experiment header format contains 6 fields:
 * - Object Key: Identifies which object a value is associated with
 * - Array Key: Indicates which array a value is associated with (may be blank)
 * - Array Index: Indicates which array index the array value is associated with
 * - Key: Identifies the specific variable
 * - Value: The value
 * - Unit: Units associated with the value (write only; not read)
 *
 *
 *
 */

using ValueUnit = std::pair<QVariant,QString>;
using HeaderMap = std::map<QString,ValueUnit>;
using HeaderArray = std::vector<HeaderMap>;
using HeaderStrings = std::multimap<QString,std::tuple<QString,QString,QString,QString,QString>>;

class HeaderStorage
{
public:
    HeaderStorage(const QString objKey);

protected:
    /*!
     * \brief Function called before saving.
     *
     * This function is where subclasses should store all values that should be saved in the header
     * (using store and storeArrayValue as desired).
     *
     */
    virtual void prepareToSave() =0;

    const QString d_objKey; /*<! Object key used for storage */

    /*!
     * \brief Stores a value-unit pair for writing
     * \param key The key assocociated with the value
     * \param val The value to store
     * \param unit Unit associated with the value (optional)
     */
    template<typename T>
    void store(const QString key, const T& val, const QString unit = "") {
        store(key,QVariant::fromValue(val),unit);
    }

    /*!
     * \brief Stores a value-unit pair for writing
     * \param key The key assocociated with the value
     * \param val The value to store
     * \param unit Unit associated with the value (optional)
     */
    void store(const QString key, const QVariant val, const QString unit = "");

    /*!
     * \brief Stores a value-unit pair as part of an array
     *
     * An array value is useful when storing repetitive entries that are part of a list
     * (e.g., settings for the pulse generator channels). If arrayKey does not exist, it
     * will be created, and if index is larger than the current array size, it will be expanded
     * until it is large enough.
     *
     * \param arrayKey Key identifying the array
     * \param index Position in the array
     * \param key Key associated with the value
     * \param val Value to be stored
     * \param unit Unit associated with the value (optional
     */
    template<typename T>
    void storeArrayValue(const QString arrayKey, std::size_t index, const QString key, const T& val, const QString unit = "") {
        storeArrayValue(arrayKey,index,key,QVariant::fromValue(val),unit);
    }

    /*!
     * \brief Stores a value-unit pair as part of an array
     *
     * An array value is useful when storing repetitive entries that are part of a list
     * (e.g., settings for the pulse generator channels). If arrayKey does not exist, it
     * will be created, and if index is larger than the current array size, it will be expanded
     * until it is large enough.
     *
     * \param arrayKey Key identifying the array
     * \param index Position in the array
     * \param key Key associated with the value
     * \param val Value to be stored
     * \param unit Unit associated with the value (optional
     */
    void storeArrayValue(const QString arrayKey, std::size_t index, const QString key, const QVariant val, const QString unit = "");

    /*!
     * \brief Retrieves a value from storage
     *
     * When a value is retrieved, it is removed from storage. A second call to retrieve with the same
     * key will return the default value.
     *
     * \param key The key associated with a value
     * \param defaultValue Return value if key not found. By default, this will be a default-constructed value.
     * \return T The value, or default value
     */
    template<typename T>
    T retrieve(const QString key, const T& defaultValue = QVariant().value<T>()) {
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
     * \brief Returns the size of the array specified by key
     * \param key The key of the array
     * \return std::size_t The size of the array. Returns 0 if the array does not exist.
     */
    std::size_t arrayStoreSize(const QString key) const;

    /*!
     * \brief Equivalent to HeaderStorage::retrieve but for array values
     *
     * The retrieved value is removed from storage. However, the individual HeaderMaps
     * are never removed, so the HeaderList size remains constant even when the HeaderMaps
     * are empty.
     *
     * \param arrayKey The key associated with the array
     * \param index Position of the value in the array
     * \param key Key associated with the value
     * \param defaultValue Return value if any key not found or if index >= array size.
     * \return T The value, or default value
     */
    template<typename T>
    T retrieveArrayValue(const QString arrayKey, std::size_t index, const QString key,
                            const T& defaultValue = QVariant().value<T>()) {
        auto out = defaultValue;

        auto it = d_arrayValues.find(arrayKey);
        if(it != d_arrayValues.end())
        {
            if(index <= it->second.size())
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
     * \brief Converts stored values to strings, and clears them from memory.
     *
     * This function is used when writing the data to disk. Because the contents are
     * in almost all cases copies of data that already exists elsewhere, the internal
     * maps are cleared out when this operation completes.
     *
     * HeaderStrings::prepareToSave is called at the beginning of this function.
     *
     * Returned is a std::multimap that contains a tuple of 5 QStrings. The key of the
     * multimap is the object key for the class. The 5 strings are:
     * 1. Array key (may be empty)
     * 2. Array index (may be empty)
     * 3. Key
     * 4. Value
     * 5. Unit (may be empty)
     *
     * If this object has any children, their getStrings() function is also called, and the
     * returned map is merged into this one.
     *
     *
     * \return HeaderStrings A multimap containing the data in string format
     */
    HeaderStrings getStrings();

    /*!
     * \brief Stores the contents of the strings provided if object key matches
     *
     * This function is intended for parsing one line of the csv header file. The line is assumed
     * to have been preiously split into a list of 6 strings as described below (the caller should
     * verify this!). If the object key does not match, then any children are searched as well. The
     * function returns true if the value was matched, and false otherwise (or if there was an error).
     *
     * \param l List of 6 strings:
     * 1. Object Key
     * 2. Array key (optional)
     * 3. Array index (optional)
     * 4. Key
     * 5. Value
     * 6. Unit (optional)
     * \return bool whether the value was added to this object or a child.
     */
    bool storeLine(const QStringList l);

    /*!
     * \brief Adds a pointer to a child HeaderStorage object
     *
     * Child storage objects are stored in a vector as pointers; ownership is not assumed here.
     * Children will be searched when adding a value through storeLine, and their getStrings function
     * will be called from the parent's getStrings function.
     *
     * \param other Pointer to the child header storage object. If the object is destroyed, calls to
     * getStrings or storeLine will crash the program.
     */
    void addChild(HeaderStorage *other);

private:
    HeaderMap d_values;
    std::map<QString,HeaderArray> d_arrayValues;
    std::vector<HeaderStorage*> d_children;


};

#endif // HEADERSTORAGE_H
