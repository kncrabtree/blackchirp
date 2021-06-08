#ifndef SETTINGSSTORAGE_H
#define SETTINGSSTORAGE_H

#include <QString>
#include <QVariant>
#include <QList>
#include <QSettings>

using SettingsGetter = std::function<QVariant()>;
using SettingsMap = std::map<QString,QVariant>;

/*!
 * \brief The SettingsStorage class manages persistent settings (through QSettings)
 *
 *
 *
 */
class SettingsStorage
{
private:
    explicit SettingsStorage(const QStringList keys, QSettings::Scope scope);
    explicit SettingsStorage(const QString orgName, const QString appName, const QStringList keys, QSettings::Scope scope);
public:
    SettingsStorage(const QStringList keys = QStringList(), bool systemWide = true);
    SettingsStorage(const QString orgName, const QString appName, const QStringList keys = QStringList(), bool systemWide = true);
    SettingsStorage(const SettingsStorage &) = delete;
    SettingsStorage& operator= (const SettingsStorage &) = delete;

    /*!
     * \brief Returns whether key is in the storage (either as a value or getter)
     *
     * \param key The key to search for
     * \return bool True if key is found
     */
    bool containsValue(const QString key) const;

    /*!
     * \brief Returns whether key is in the storage as an array
     *
     * \param key The key to search for
     * \return bool True if key is found
     */
    bool containsArray(const QString key) const;

    /*!
     * \brief Gets the value of a setting
     *
     * If a getter function has been registered (see SettingsStorage::registerGetter), then
     * that getter function will be called.
     *
     * \param key The key associated with the value
     * \return QVariant The value, or QVariant() if the key is not present
     */
    QVariant get(const QString key) const;

    /*!
     * \brief Gets the value of a settting. Overloaded function.
     *
     * Attempts to convert the value to type T using QVariant::value<T>(). See Qt
     * documentation for a discussion
     *
     * \param key The key associated with the value
     * \return T The value, or a default constructed value if the key is not present
     */
    template<typename T>
    inline T get(const QString key) const { return get(key).value<T>(); };

    /*!
     * \brief Gets values associated with a list of keys. Overloaded function
     *
     * If a key in the list is not found, then it is skipped. The returned map may be empty.
     * Recommended usage:
     *
     * `SettingsStorage s(mainKey,subKeys,systemWide);
     * auto x = get( {'key1','key2','key3'} );
     * auto key1Val = x.at('key1');`
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
     * \return std::vector<SettingsMap> The array value. An empty vector is returned if the key is not found
     */
    std::vector<SettingsMap> getArray(const QString key) const;

    /*!
     * \brief Returns a single map from an array associated with a key
     * \param key The key of the array
     * \param i Index of the desired map
     * \return SettingsMap The selected map, which will be empty if key does not exist or if i is out of bounds for the array
     */
    SettingsMap getArrayValue(const QString key, std::size_t i) const;


protected:
    /*!
     * \brief Registers a getter function for a given setting
     *
     * When a getter is associated with a key, any corresponding key is removed from the values
     * dictionary, and instead the getter function will be called to access the value.
     *
     * The getter function will be called when saving the settings to disk or when calling the
     * get function.
     *
     * This function only works for single values, not arrays.
     *
     * \param key The key for the value to be stored
     * \param obj A pointer to the object containing the getter function
     * \param getter A member function pointer that returns a type that can be implicitly converted to QVariant
     *
     * \return bool Whether the getter was registered
     */
    template<class T, typename Out>
    bool registerGetter(QString key, T* obj, Out (T::*getter)() const)
    {
        //cannot register a getter for an array value
        if(d_arrayValues.find(key) != d_arrayValues.end())
            return false;

        auto it = d_values.find(key);
        if(it != d_values.end())
            d_values.erase(it->first);

        auto f = [obj, getter](){ return (obj->*getter)(); };
        d_getters.emplace(key,f);

        return true;
    }


    /*!
     * \brief Reads a setting, and sets a default value if it does not exist.
     *
     * Searches for and returns the value associated with the indicated key. If the key does not
     * exist in the settings file, then an entry is created and the default value written.
     *
     * The intention of this function is to allow a developer to expose settings that a user may want
     * to edit in the settings editor. For example, a Clock object has "minFreqMHz" and "maxFreqMHz"
     * settings that correspond to the actual hardware limits. Those settings are read by the user
     * interface to set limits on input widgets that control the desired frequency setting. The user
     * can change these values to (presumably) narrow the range of allowed values. By calling this
     * function for each setting that should be exposed, an entry will be guaranteed to be created in
     * the settings file.
     *
     * \param key The key for the value to be stored
     * \param defaultValue The desired default value written to settings if the key does not exist
     * \return QVariant containing the value associated with the key. If the key did not previously exist, this will equal defaltValue
     */
    QVariant getOrSetDefault(QString key, QVariant defaultValue);

    /*!
     * \brief Stores a key-value setting
     *
     * The value is placed into the values map and associated with the given key. If key already
     * exists, its value is overwritten; otherwise a new key is created. The operation will not be
     * completed if the key is associated with a getter function or with an array value.
     *
     * The write argument controls whether the new setting is immediately written to persistent storage.
     * If write is false, then the setting will not be stored until a call to SettingsStorage::save is made.
     *
     * \param key The key associated with the value
     * \param value The value to be stored
     * \param write If true, write to persistent storage immediately
     * \return bool Returns whether or not the setting was made. If false, the key is already associated with a getter or array value
     */
    bool set(QString key, QVariant value, bool write = true);

    /*!
     * \brief Sets multiple key-value settings
     *
     * Calls SettingsStorage::set for each key-value pair in the input map. If the setting was successful
     * and write is true, the new value is stored in QSettings immediately. The success of each setting
     * is returned in a map.
     *
     * \param m Map of key-value pairs to add
     * \param write If true, write to QSettings immediately
     * \return std::map<QString,bool> Contains return value of SettingsStorage::set for each key
     */
    std::map<QString,bool> setMultiple(SettingsMap m, bool write = true);

    /*!
     * \brief Sets (or unsets) an array value
     *
     * Stores a vector of maps that will be written to QSettings using QSettings::beginWriteArray()
     * Passing an empty array argument will remove the value from QSettings.
     * Changes to QSettings are made immediately if write is true, and upon the next call to
     * SettingsStorage::save otherwise
     *
     * \param key The key of the array value
     * \param array The new array value (may be empty)
     * \param write If true, QSettings is updated immediately
     */
    void setArray(QString key, std::vector<SettingsMap> array, bool write = true);

    /*!
     * \brief Write all values to QSettings.
     */
    void save();

    void readAll();

private:
    SettingsMap d_values;

    std::map<QString, SettingsGetter> d_getters;
    std::map<QString,std::vector<SettingsMap>> d_arrayValues;

    QSettings d_settings;


};

#endif // SETTINGSSTORAGE_H
