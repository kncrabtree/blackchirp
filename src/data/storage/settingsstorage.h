#ifndef SETTINGSSTORAGE_H
#define SETTINGSSTORAGE_H

#include <QString>
#include <QVariant>
#include <QList>
#include <QSettings>
#include <QCoreApplication>
#include <QAnyStringView>
#include <QLatin1StringView>

#include <data/bcglobals.h>

/*!
 * Blackchirp global namespace
 */ 
namespace BC {
/*!
 * Global SettingsStorage keys
 */
namespace Key {
}
}

namespace BC::Key {
inline constexpr QLatin1StringView BC{"Blackchirp"};
inline constexpr QLatin1StringView exptNum{"exptNum"};
inline constexpr QLatin1StringView savePath{"savePath"};
inline constexpr QLatin1StringView appFont{"appFont"};
inline constexpr QLatin1StringView versionMajor{"versionMajor"};
inline constexpr QLatin1StringView versionMinor{"versionMinor"};
inline constexpr QLatin1StringView versionPatch{"versionPatch"};
inline constexpr QLatin1StringView versionRelease{"versionRelease"};
inline constexpr QLatin1StringView exptDir{"experiments"};
inline constexpr QLatin1StringView logDir{"log"};
inline constexpr QLatin1StringView exportDir{"textexports"};
inline constexpr QLatin1StringView exportDelimiter{"exportDelimiter"};
inline constexpr QLatin1StringView trackingDir{"rollingdata"};
}

/*!
 * \brief Owning, write-protected wrapper around a
 * <a href="https://doc.qt.io/qt-6/qsettings.html">QSettings</a> group.
 *
 * Maintains an in-memory copy of every key, array, group, and
 * registered getter under a single QSettings group, and splits its
 * API across two trust levels:
 *
 * - **public, read-only**: get(), getArray(), getGroupValue(), the
 *   contains/keys helpers. Anywhere in the program may construct a
 *   transient SettingsStorage over a group and read from it.
 * - **protected, mutating**: set(), setArray(), setGroupValue(),
 *   setDefault(), registerGetter(), purge(), and the rest. Only a
 *   subclass that *owns* the group may call these.
 *
 * The split is what lets, for example, any UI code look up a hardware
 * driver's persisted value while still guaranteeing that only the
 * owning HardwareObject (or a friend declared by it) can change it.
 *
 * Lifetime invariants: the destructor calls save(), so any pending
 * changes land in QSettings when the owner goes away. Owners that
 * bind getters to UI widgets must call clearGetters() in their
 * destructor *before* the widgets are torn down, or save() will
 * dereference deleted objects. Inheriting from SettingsStorage
 * deletes the copy and assignment operators, so owners must be
 * passed by pointer or reference — never by value. Data classes that
 * need value semantics (e.g. Experiment, FtmwConfig) read from
 * SettingsStorage rather than inheriting from it.
 *
 * The \c Type enum and the \c type constructor parameter are kept
 * for source compatibility but have no effect on the opened group;
 * always pass the full keys list. Per-mutation \c write flags select
 * between immediate QSettings writes and in-memory batching that
 * lands at the next save(); discardChanges(true) suppresses writes
 * from the destructor and from periodic save() paths and is the
 * mechanism behind the loadout/profile manager batched-write
 * pattern.
 */
class SettingsStorage
{
    friend class SettingsStorageTest; // Allow test class access to protected methods
    friend class HwSettingsWidget;    // Allow direct storage writes for hardware settings UI
public:

    using SettingsGetter = std::function<QVariant()>; /*!< Alias for a getter function */
    using SettingsMap = std::map<QString,QVariant,std::less<>>; /*!< Alias for a map of strings and variants */

    /*!
     * \brief Reserved type tag.
     *
     * Currently a no-op kept for source compatibility with older
     * call sites. The constructor accepts a \c Type argument but
     * does not use it.
     */
    enum Type {
        General,  /*!< Use the keys list as given. */
        Hardware  /*!< Identical to General; retained for compatibility only. */
    };

    /*!
     * \brief Open a QSettings group and read it into memory.
     *
     * Calls `QSettings::beginGroup` once per entry in \a keys. An
     * empty list opens the top-level \c "Blackchirp" group. All
     * values, arrays, and groups under the resulting (sub)group are
     * read into the in-memory caches.
     *
     * \param keys Group path, applied as nested `beginGroup` calls.
     * \param type Reserved; ignored (see Type).
     */
    SettingsStorage(const QStringList keys = QStringList(), Type type = General);

    /*!
     * \brief Constructor with explicit organization and application names.
     *
     * Used by unit tests and by code that needs to read settings written
     * by another QCoreApplication identity.
     *
     * \param orgName Organization name passed to QSettings.
     * \param appName Application name passed to QSettings.
     * \param keys Group path, applied as nested `beginGroup` calls.
     * \param type Reserved; ignored (see Type).
     */
    SettingsStorage(QAnyStringView orgName, QAnyStringView appName, const QStringList keys = QStringList(), Type type = General);

    /*!
     * \brief Convenience constructor for a single-element group path.
     *
     * Equivalent to passing \c {{key}} to the QStringList constructor.
     *
     * \param key Group key. If empty, opens the top-level
     * \c "Blackchirp" group.
     * \param type Reserved; ignored (see Type).
     */
    SettingsStorage(QAnyStringView key, Type type = General);

    /*!
     * \brief Destructor. Calls save() unless discardChanges(true) was set.
     */
    virtual ~SettingsStorage();

    SettingsStorage(const SettingsStorage &) = delete;
    SettingsStorage& operator= (const SettingsStorage &) = delete;

    /*!
     * \brief Returns whether key is in the storage (either as a value or getter)
     *
     * \param key The key to search for
     * \return If key is found
     */
    bool containsValue(QAnyStringView key) const;

    /*!
     * \brief Returns whether key is in the storage as an array
     *
     * \param key The key to search for
     * \return If key is found
     */
    bool containsArray(QAnyStringView key) const;

    /*!
     * \brief Gets the value of a setting
     *
     * If a getter function has been registered (see registerGetter()), then
     * that getter function will be called. The optional `defaultValue`
     * argument is returned if the key is not found.
     *
     * \param key The key associated with the value
     * \param defaultValue The value returned if key is not present (default: `QVariant()`)
     * \return The value
     */
    QVariant get(QAnyStringView key, const QVariant &defaultValue = QVariant()) const;

    /*!
     * \brief Gets the value of a settting. Overloaded function.
     *
     * Attempts to convert the value to type T using <a
     * href="https://doc.qt.io/qt-6/qvariant.html#value">QVariant::value</a>.
     * The optional `defaultValue` argument is returned if the key is not
     * found. If left blank, a default-constructed value is returned for any
     * missing keys.
     *
     * \param key The key associated with the value
     * \param defaultValue The value returned if key is not present
     * \return The value, or a default constructed value if the key is not present
     */
    template<typename T>
    inline T get(QAnyStringView key, const T &defaultValue = QVariant().value<T>()) const { return (containsValue(key) ? get(key).value<T>() : defaultValue); }

    /*!
     * \brief Gets values associated with a list of keys. Overloaded function
     *
     * If a key in the list is not found, then it is skipped. The returned map
     * may be empty. Recommended usage:
     *
     *     SettingsStorage s(keys);
     *     auto x = getMultiple( {"key1","key2","key3"} );
     *     auto key1Val = x.at("key1");
     *
     * \param keys The list of keys to search for
     * \return Map containing the keys found in the values or getter maps
     */
    SettingsMap getMultiple(const std::vector<QString> keys) const;

    /*!
     * \brief Gets an array assocated with a key
     *
     * An array is a list of maps, each of which has its own keys and values
     *
     * \param key The key associated with the array
     * \return The array. An empty vector is returned if the key is not found
     */
    std::vector<SettingsMap> getArray(QAnyStringView key) const;

    /*!
     * \brief Returns the size of the array value associated with key
     *
     * \param key The key of the array value
     * \return The number of maps in the array (0 if key does not exist);
     */
    std::size_t getArraySize(QAnyStringView key) const;

    /*!
     * \brief Returns a single map from an array associated with a key
     * 
     * \param key The key of the array
     * \param i Index of the desired map
     * \return The selected map, which will be empty if key does not exist or
     * if i is out of bounds for the array
     */
    SettingsMap getArrayMap(QAnyStringView key, std::size_t i) const;

    /*!
     * \brief Gets a single value from a map that is part of an array value
     *
     * Calls getArrayMap(), then searches for `mapKey` within the returned map
     * (which may be empty). Returns the stored `QVariant` or the defaultValue
     * argument.
     *
     * \param arrayKey The key of the array
     * \param i Index of the desired map
     * \param mapKey Key of the desired value within the map
     * \param defaultValue Value returned if `arrayKey` does not exist **or**
     * i is out of bounds **or** mapKey does not exist
     * \return Stored value or defaultValue
     */
    QVariant getArrayValue(QAnyStringView arrayKey, std::size_t i, QAnyStringView mapKey, const QVariant defaultValue = QVariant()) const;


    /*!
     * \brief Overloaded function. See getArrayValue()
     *
     * \param arrayKey The key of the array
     * \param i Index of the desired map
     * \param mapKey Key of the desired value within the map
     * \param defaultValue Value returned if `arrayKey` does not exist **or** `i`
     * is out of bounds **or** `mapKey` does not exist (dedaults to a default-constructed value)
     * \return The value or the defaultValue, as appropriate
     */
    template<typename T>
    inline T getArrayValue(QAnyStringView arrayKey, std::size_t i, QAnyStringView mapKey, T defaultValue = QVariant().value<T>()) const {
        auto m = getArrayMap(arrayKey,i);
        auto it = m.find(mapKey);
        if(it != m.end())
            return it->second.value<T>();

        return defaultValue;
    }

    /*!
     * \brief Gets a value from a group-based key-value store
     *
     * Groups allow hierarchical organization of settings. Each group contains
     * its own set of key-value pairs. This is useful for protocol-specific 
     * settings, device configurations, etc.
     *
     * \param groupKey The group identifier 
     * \param key The key within the group
     * \param defaultValue Value returned if group or key doesn't exist
     * \return The stored QVariant or defaultValue if not found
     */
    QVariant getGroupValue(QAnyStringView groupKey, QAnyStringView key, const QVariant &defaultValue = QVariant()) const;

    /*!
     * \brief Gets a value from a group-based key-value store. Overloaded function
     *
     * Template version that automatically converts to the specified type.
     *
     * \param groupKey The group identifier
     * \param key The key within the group  
     * \param defaultValue Value returned if group or key doesn't exist
     * \return The stored value converted to type T, or defaultValue if not found
     */
    template<typename T>
    inline T getGroupValue(QAnyStringView groupKey, QAnyStringView key, const T &defaultValue = QVariant().value<T>()) const {
        auto groupIt = d_groupValues.find(groupKey);
        if(groupIt != d_groupValues.end())
        {
            auto keyIt = groupIt->second.find(key);
            if(keyIt != groupIt->second.end())
                return keyIt->second.value<T>();
        }
        
        return defaultValue;
    }

    /*!
     * \brief Gets all key-value pairs within a group
     *
     * \param groupKey The group identifier
     * \return SettingsMap containing all key-value pairs in the group (empty if group doesn't exist)
     */
    SettingsMap getGroup(QAnyStringView groupKey) const;

    /*!
     * \brief Controls whether changes are written to `QSettings`
     * \param discard If true, settings are not saved.
     */
    void discardChanges(bool discard = true) { d_discard = discard; }
    
    /*!
     * \brief List of all keys for normal (non-array) values
     * 
     * \return List of keys
     */
    QStringList keys() const;
    
    /*!
     * \brief List of keys for array values
     * 
     * \return List of Keys
     */
    QStringList arrayKeys() const;
    
    /*!
     * \brief List of all group keys
     * 
     * \return List of group keys
     */
    QStringList groupKeys() const;

protected:
    /*!
     * \brief Registers a getter function for a given setting
     *
     * When a getter is associated with a key, any corresponding key is removed
     * from the values dictionary, and instead the getter function will be
     * called to access the value. The getter function will be called when
     * saving the settings to disk or when calling the get function.
     *
     * This function only works for single values, not arrays.
     *
     * This form is intended for use with member function pointers.
     *
     * \param key The key for the value to be stored
     * \param obj A pointer to the object containing the getter function
     * \param getter A member function pointer that returns a type that can be
     * implicitly converted to QVariant
     *
     * \return Whether the getter was registered
     */
    template<class T, typename Out>
    bool registerGetter(QAnyStringView key, T* obj, Out (T::*getter)() const)
    {
        //cannot register a getter for an array value
        if(d_arrayValues.find(key) != d_arrayValues.end())
            return false;

        d_edited = true;

        auto it = d_values.find(key);
        if(it != d_values.end())
            d_values.erase(it->first);

        auto f = [obj, getter](){ return (obj->*getter)(); };
        return d_getters.emplace(key.toString(),f).second;
    }

    /*!
     * \brief Registers a getter function for a given setting (overloaded
     * function)
     *
     * When a getter is associated with a key, any corresponding key is removed
     * from the values dictionary, and instead the getter function will be
     * called to access the value. The getter function will be called when
     * saving the settings to disk or when calling the get function.
     *
     * This function only works for single values, not arrays.
     *
     * This form can be used with lambda functions. For example:
     *
     *     int x = 3;
     *     auto f = std::function<int ()> { [x](){ return x + 1; }
     *     registerGetter("myInt",f);
     *     auto y = get<int>("myInt")
     *     //y == 4
     *
     */
    template<typename T>
    bool registerGetter(QAnyStringView key, std::function<T()> f)
    {
        //cannot register a getter for an array value
        if(d_arrayValues.find(key) != d_arrayValues.end())
            return false;

        d_edited = true;

        auto it = d_values.find(key);
        if(it != d_values.end())
            d_values.erase(it->first);

        return d_getters.emplace(key.toString(),f).second;
    }

    /*!
     * \brief Removes a getter function
     *
     * If key matches a previously registered getter, then the getter function
     * is removed from the storage object. The getter itself is called, and the
     * value is stored in the values map with the same key amd returned.
     *
     * An empty QVariant is returned if no getter was found.
     *
     * If the write parameter is true, the current value is written to
     * persistent storage.
     *
     * \param key The key associated with the getter
     * \param write If true, value is written to persistent storage immediately
     * \return Return value of getter call, or empty QVariant if no getter was found.
     */
    QVariant unRegisterGetter(QAnyStringView key, bool write = false);

    /*!
     * \brief Removes all getter functions
     *
     * All getters are removed and their values are transferred into the values
     * map.
     *
     * \param write If true, values are written to persistent storage
     */
    void clearGetters(bool write = false);


    /*!
     * \brief Reads a setting, and sets a default value if it does not exist.
     *
     * Searches for and returns the value associated with the indicated key. If
     * the key does not exist in the settings file, then an entry is created
     * and the default value written.
     *
     * The intention of this function is to allow a developer to expose
     * settings that a user may want to edit in the settings editor. For
     * example, a `Clock` object has "minFreqMHz" and "maxFreqMHz" settings that
     * correspond to the actual hardware limits. Those settings are read by the
     * user interface to set limits on input widgets that control the desired
     * frequency setting. The user can change these values to (presumably)
     * narrow the range of allowed values. By calling this function for each
     * setting that should be exposed, an entry will be guaranteed to be
     * created in the settings file.
     *
     * \param key The key for the value to be stored
     * \param defaultValue The desired default value written to settings if
     * the key does not exist
     * \return Value associated with the key. If the key did not previously
     * exist, this will equal defaultValue
     */
    QVariant getOrSetDefault(QAnyStringView key, const QVariant defaultValue);

    /*!
     * \brief Reads a settings, and sets a default value if it does not exist
     *
     * Templated version of getOrSetDefault
     *
     * \param key The key for the value to be stored
     * \param defaultValue The desired default value written to settings if the
     * key does not exist
     * \return Value associated with the key. If the key did not previously exist,
     * this will equal defaultValue
     *
     */
    template<typename T>
    T getOrSetDefault(QAnyStringView key, const T &defaultValue) {
        QVariant out = getOrSetDefault(key,QVariant::fromValue(defaultValue));
        return out.value<T>();
    }

    /*!
     * \brief Sets a default value if none exists
     *
     * If a value already exists corresponding to a key, no action is taken.
     *
     * \param key The key for the value
     * \param defaultValue Value to set if key is not found.
     */
    void setDefault(QAnyStringView key, const QVariant defaultValue);

    /*!
     * \brief Sets a default value if none exists. Overloaded function
     *
     * If a value already exists corresponding to a key, no action is taken.
     *
     * \param key The key for the value
     * \param defaultValue Value to set if key is not found.
     */
    template<typename T>
    void setDefault(QAnyStringView key, const T &defaultValue) {
        setDefault(key,QVariant::fromValue(defaultValue));
    }

    /*!
     * \brief Stores a key-value setting
     *
     * The value is placed into the values map and associated with the given
     * key. If key already exists, its value is overwritten; otherwise a new
     * key is created. The operation will not be completed if the key is
     * associated with a getter function or with an array value.
     *
     * The write argument controls whether the new setting is immediately
     * written to persistent storage. If write is false, then the setting will
     * not be stored until a call to save() is made.
     *
     * \param key The key associated with the value
     * \param value The value to be stored
     * \param write If true, write to persistent storage immediately
     * \return Whether or not the setting was made. If false, the key is already
     * associated with a getter or array value
     */
    bool set(QAnyStringView key, const QVariant &value, bool write = false);

    /*!
     * \brief Stores a key-value setting. Overloaded function
     *
     * See set(const QString key, const QVariant &value, bool write)
     *
     * \param key The key associated with the value
     * \param value The value to be stored
     * \param write If true, write to persistent storage immediately
     * \return Whether or not the setting was made. If false, the key is already
     * associated with a getter or array value
     */
    template<typename T>
    bool set(QAnyStringView key, const T &value, bool write = false) {
        return set(key,QVariant::fromValue(value),write);
    }

    /*!
     * \brief Sets multiple key-value settings
     *
     * Calls set() for each key-value pair in the input map. If the setting was
     * successful and write is true, the new value is stored in QSettings
     * immediately. The success of each setting is returned in a map.
     *
     * \param m Map of key-value pairs to add
     * \param write If true, write to QSettings immediately
     * \return Return value of set() for each key
     */
    std::map<QString,bool,std::less<>> setMultiple(const SettingsMap m, bool write = false);

    /*!
     * \brief Sets (or unsets) an array value
     *
     * Stores a vector of maps that will be written using <a
     * href="https://doc.qt.io/qt-6/qsettings.html#beginWriteArray">QSettings::beginWriteArray</a>.
     * Passing an empty array argument will remove the value from `QSettings`.
     * Changes to `QSettings` are made immediately if write is true, and upon
     * the next call to save() otherwise.
     *
     * \param key The key of the array value
     * \param array The new array value (may be empty)
     * \param write If true, QSettings is updated immediately
     */
    void setArray(QAnyStringView key, const std::vector<SettingsMap> &array, bool write = false);

    /*!
     * \brief Sets a single value within a map associated with an array value
     *
     * Attempts to set one key-value pair for the array value specified by
     * `arrayKey` at position `i`. The write will fail if the array does not
     * exist or if `i` is out of bounds. If the optional `write` parameter is
     * true, then the updated array will be written to `QSettings`.
     *
     * \param arrayKey Key of the array value
     * \param i Index of the map within the array
     * \param key Key for the map
     * \param value Value to be stored
     * \param write If true, write updated array to `QSettings`
     * \return Whether setting was successfully made
     */
    bool setArrayValue(QAnyStringView arrayKey, std::size_t i, QAnyStringView key, const QVariant &value, bool write = false);

    /*!
     * \brief Sets a single value within a map associated with an array value.
     * Overloaded function
     *
     * See setArrayValue(const QString arrayKey, std::size_t i, const QString
     * key, const QVariant &value, bool write)
     *
     * \param arrayKey Key of the array value
     * \param i Index of the map within the array
     * \param key Key for the map
     * \param value Value to be stored
     * \param write If true, write updated array to QSettings
     * \return Whether setting was successfully made
     */
    template<typename T>
    bool setArrayValue(QAnyStringView arrayKey, std::size_t i, QAnyStringView key, const T &value, bool write = false) {
        return setArrayValue(arrayKey,i,key,QVariant::fromValue(value),write);
    }

    /*!
     * \brief Appends a new map onto an array value
     *
     * If the key does not match an existing array variable, it is added. By
     * default, the new array is not written to settings immediately. This is
     * because QSettings essentially requires rewriting the entire array every
     * time, and this function is intended to be called as part of a loop.
     *
     * \param key The key of the array value
     * \param map The new map to append
     * \param write Whether to update `QSettings` immediately
     */
    void appendArrayMap(QAnyStringView key, const SettingsMap &map, bool write = false);

    /*!
     * \brief Sets a value within a group-based key-value store
     *
     * Groups provide hierarchical organization of settings. This method creates
     * the group if it doesn't exist and sets the specified key-value pair within it.
     *
     * \param groupKey The group identifier
     * \param key The key within the group
     * \param value The value to be stored
     * \param write If true, write to persistent storage immediately
     * \return Whether the setting was successfully made
     */
    bool setGroupValue(QAnyStringView groupKey, QAnyStringView key, const QVariant &value, bool write = false);

    /*!
     * \brief Sets a value within a group-based key-value store. Overloaded function
     *
     * Template version for type safety.
     *
     * \param groupKey The group identifier  
     * \param key The key within the group
     * \param value The value to be stored
     * \param write If true, write to persistent storage immediately
     * \return Whether the setting was successfully made
     */
    template<typename T>
    bool setGroupValue(QAnyStringView groupKey, QAnyStringView key, const T &value, bool write = false) {
        return setGroupValue(groupKey, key, QVariant::fromValue(value), write);
    }

    /*!
     * \brief Sets multiple values within a group
     *
     * \param groupKey The group identifier
     * \param values Map of key-value pairs to set within the group
     * \param write If true, write to persistent storage immediately
     * \return Map indicating success/failure for each key
     */
    std::map<QString,bool,std::less<>> setGroupValues(QAnyStringView groupKey, const SettingsMap &values, bool write = false);

    /*!
     * \brief Sets a default value within a group if none exists
     *
     * Group-aware counterpart to setDefault(): if the group already contains
     * the key, no action is taken; otherwise the value is stored via
     * setGroupValue(). A value the user has already configured is never
     * overwritten, which makes this safe to call on every construction (e.g.
     * to seed registry-provided communication defaults).
     *
     * \param groupKey The group identifier
     * \param key The key within the group
     * \param defaultValue Value to set if the group key is not found
     * \param write If true, write to persistent storage immediately
     */
    void setGroupDefault(QAnyStringView groupKey, QAnyStringView key, const QVariant &defaultValue, bool write = false);

    /*!
     * \brief Sets a default value within a group if none exists. Overloaded function
     *
     * Template version for type safety.
     *
     * \param groupKey The group identifier
     * \param key The key within the group
     * \param defaultValue Value to set if the group key is not found
     * \param write If true, write to persistent storage immediately
     */
    template<typename T>
    void setGroupDefault(QAnyStringView groupKey, QAnyStringView key, const T &defaultValue, bool write = false) {
        setGroupDefault(groupKey, key, QVariant::fromValue(defaultValue), write);
    }

    /*!
     * \brief Clears all data associated with a key and removes it from QSettings
     *
     * This clears all forms of data associated with the given key: regular values,
     * getter functions, and array values. The key is immediately removed from
     * `QSettings`. If the key is not found in any form, no action is taken.
     *
     * \param key The key to clear from all storage forms
     */
    void clearValue(QAnyStringView key);

    /*!
     * \brief Removes all settings for this object's group from persistent storage
     *
     * Clears all in-memory data (values, getters, arrays, groups) and removes
     * the entire QSettings group associated with this object. Sets d_discard so
     * the destructor will not re-write the cleared data. Intended for use when
     * a hardware profile is permanently deleted and this object is about to be
     * destroyed.
     */
    void purge();

    /*!
     * \brief Removes a QSettings group by key path without requiring a live instance
     *
     * Utility for purging settings when no SettingsStorage instance exists for the
     * target group (e.g., when deleting an inactive hardware profile that has no
     * corresponding live HardwareObject).
     *
     * \param keys The list of group/subgroup keys that identify the group to remove
     */
    static void purgeGroup(const QStringList& keys);

    /*!
     * \brief Removes all top-level QSettings groups whose name ends with
     * \c "." + suffix. Used to clean up widget settings (e.g.,
     * \c "PulseWidget.PulseGenerator.main") when a hardware profile whose
     * key matches \a suffix is deleted.
     *
     * \param suffix The hardware key to match (e.g., "PulseGenerator.main")
     */
    static void purgeGroupsBySuffix(QAnyStringView suffix);

    /*!
     * \brief Write all values to `QSettings`.
     */
    void save();
    
    
    /*!
     * \brief Reads all values from settings file (see note about getters)
     *
     * This function reads all values from the `QSettings` storage. It does so
     * by clearing out the values and arrayValues maps, and reading in all keys
     * and array groups found in the settings file.
     *
     * If a getter has been registered, that key will be skipped. First call
     * unRegisterGetter() or clearGetters() if you wish to these keys to be
     * re-read.
     *
     */
    void readAll();
    
private:
    explicit SettingsStorage(const QStringList keys, Type type, QSettings::Scope scope);
    explicit SettingsStorage(QAnyStringView orgName, QAnyStringView appName, const QStringList keys, Type type, QSettings::Scope scope);

    SettingsMap d_values; /*!< Map of key-value pairs */
    bool d_discard{false}; /*! If set to true, changes will not be stored to QSettings */
    bool d_edited{false}; /*! Set to true when a value is changed. */

    std::map<QString, SettingsGetter, std::less<>> d_getters; /*!< Map containing all registered getters */
    std::map<QString,std::vector<SettingsMap>,std::less<>> d_arrayValues; /*!< Map containing all array values */
    std::map<QString, SettingsMap, std::less<>> d_groupValues; /*!< Map containing group-based key-value pairs */

    QSettings d_settings; /*!< Handle to QSettings storage object */

    /*!
     * \brief Writes a single array to QSettings
     * \param key Key of the array to write
     */
    void writeArray(QAnyStringView key);

    /*!
     * \brief Writes a single group to QSettings
     * \param groupKey Key of the group to write
     */
    void writeGroup(QAnyStringView groupKey);


};

#endif // SETTINGSSTORAGE_H
