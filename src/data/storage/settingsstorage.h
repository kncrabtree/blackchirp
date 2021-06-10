#ifndef SETTINGSSTORAGE_H
#define SETTINGSSTORAGE_H

#include <QString>
#include <QVariant>
#include <QList>
#include <QSettings>
#include <QCoreApplication>

using SettingsGetter = std::function<QVariant()>; /*!< Alias for a getter function */
using SettingsMap = std::map<QString,QVariant>; /*!< Alias for a map of strongs and variants */

/*!
 * \brief The SettingsStorage class manages persistent settings (through QSettings)
 *
 * The SettingsStorage class proides a unified interface for classes that need to read from or write to
 * persistent storage through QSettings. For read-only access to QSettings, a SettingsStorage object can
 * be created at any point in the code and initialized with the appropriate keys that refer to the group
 * that needs to be read from. All functions that modify values in QSettings are protected. Classes that
 * wish to use SettingsStorage to write to persistent storage need to inherit SettingsStorage and initialize
 * it in their constructors. SettingsStorage does not inherit any other classes, and it is suitable for use
 * in multiple inheritance with QObject-derived classes. **However:** classes that inherit from SettingsStorage
 * will have their assignment and copy constructors deleted! Do not inherit from SettingsStorage in a
 * class that needs to be passed around by value (such as data storage classes like Experiment). This class
 * is intended for use with objects that only passed by pointer (e.g., HardwareObject, UI classes, etc).
 *
 * A SettingsStorage object reads and maintains an internal copy of the QSettings keys and values associated
 * with the group/subgroup that it is initialized with. Internally, this is done through the use of two
 * assocuative containers (key-value containers): one which represents single key-value pairs,
 * and another that contains array values as structured by QSettings. An array value is a list whose items
 * each contain a map consisting of one or more key-value pairs.
 *
 * When initializing SettingsStorage, the standard constructor is
 *
 *     SettingsStorage::SettingsStorage(const QStringList keys, Type type, bool systemWide)
 *
 * QSettings::beginGroup will be called for each key in the keys list. If the list is empty, then
 * the group is set to "Blackchirp". This is done to prevent QSettings form reading in various system-wide
 * garbage on macOS. The `systemWide` argument is used to determine whether the main BlackChirp.conf file
 * is accessed (which by default is located in /home/data/CrabtreeLab on unix systems), or a user-specific one.
 * Settings that should apply to any user account (experiment number, hardware configuration, etc) should set
 * systemWide=true, and settings that are specific to a user (UI colors/preferences) should set systemWide=false.
 * If Type is set to Hardware and the length of the keys list is 1, then the program assumes the key in the list
 * corresponds to a HardwareObject, and the current subKey will be added to the keys list upon opening QSettings.
 * Alternative forms of the constructor exist that take a const char* or QString specifically for single-key operations.
 *
 *     SettingsStorage::SettingsStorage(const QString key, Type type = General, bool systemWide = true)
 *     SettingsStorage::SettingsStorage(const char* key, Type type = General, bool systemWide = true)
 *
 * There is also a form of the constructor that places the systemwide argument first for easier access to General (non-hardware)
 * settings in User scope:
 *
 *     SettingsStorage::SettingsStorage(bool systemWide, const QStringList keys = QStringList())
 *
 * To create a read-only SettingsStorage object that reads the global Blackchirp settings:
 *
 *     SettingsStorage s;
 *
 *     //for user settings: SettingsStorage s(false);
 *
 * If instead you need read-only access to the "awg/virtual" group:
 *
 *     SettingsStorage s({"awg","virtual"});
 *
 *
 * For read-only access to the settings associated with the current "awg" implementation (determined from QSettings):
 *
 *     SettingsStorage s("awg",SettingsStorage::Hardware);
 *
 * Finally, for read-only access to the "trackingPlot" group for user-based setting:
 *
 *     SettingsStorage s(false,"trackingPlot");
 *     //alternative: SettingsStorage s("trackingPlot",SettingsStorage::General,false);
 *
 * The value associated with a key can be obtained with one of the SettingsStorage::get functions. If there
 * is an integer associacted with the key "myInt", it can be accessed as:
 *
 *     //returns a QVariant containing "myInt", or QVariant() if "myKey" is not found.
 *     QVariant v = get("myInt");
 *
 *     //attempts to convert to an integer using QVariant::value.
 *     //Returns default-constructed value if unsuccessful
 *     int v2 = get<int>("myInt");
 *
 *     //in either case. a default argument can be supplied, which will be returned if the key
 *     //is not found.
 *     QVariant defaultInt = get("myInt",10);
 *     int defaultInt2 = get<int>("myInt",10);
 *
 *
 * There is also SettingsStorage::getMultiple that returns a std::map<QString,Qvariant> containing
 * all keys that match the indicated values.
 *
 * Array values can be accessed with the SettingsStorage::getArray function, which returns a const reference
 * to the array as a std::vector<SettingsMap>. The vector will be empty if the key is not found.
 * Alternatively, a reference to a particular SettingsMap within the array can be accesed by index with
 * SettingsStorage::getArrayMap. The returned map will be empty if the index is out of bounds or if the key
 * is not found. Finally, SettingsStorage::getArrayValue may be used to access one individual element in an
 * array value map by its key, using an interface similar to SettingsStorage::get.
 *
 *     //Access "arrayKey" map at index 1, return value associated with "mapKey"
 *     //If "arrayKey" is not present, 1 is out of bounds, or "mapKey" is not present, v contains defaultValue
 *     QVariant v = getArrayValue("arrayKey",1,"mapKey",defaultValue);
 *
 *     //alternative form using template function; d will contain 1.5 if lookup fails.
 *     double d = getArrayValue("arrayKey",1,"mapKey",1.5)
 *
 * The SettingsStorage::containsValue and SettingsStorage::containsArray functions can be used to check whether
 * a given key exists for a standard value or an array value, respectively.
 *
 * To obtain write access to persistent storage thogh the SettingsStorage interface, an object must inherit
 * from SettingsStorage: e.g., `class MyClass : public QObject, public SettingsStorage`, and the constructor
 * must initialise SettingsStorage with an initializer; e.g.,
 *
 *     MyClass::MyClass(QObject *parent) : QObject(parent),
 *         SettingsStorage({"MyClassKey","MyClassSubkey"})
 *     {
 *         //other initialization
 *     }
 *
 * When working with a subclass of SettingsStorage, the object has access to the SettingsStorage::set,
 * SettingsStorage::setMultiple, and SettingsStorage::setArray functions. Each of these takes an optional
 * bool argument (default true) that controls whther the new value is immedately written to settings.
 * If false, the value is just stored in memory until a call to SettingsStorage::save() is made. It is
 * recommended that SettingsStorage::save() is called in the destructor of the child class to ensure that
 * any changes are written before the object is destroyed. Because of the getter mechanism described below,
 * it is not possible to call save() in the destructor of SettingsStorage. If the key in a call to one of the
 * set functions does not exist, a new key-value pair is added.
 *
 * In addition, a subclass may call SettingsStorage::readAll at any point to reread all values from settings.
 * However, any keys associated with a getter will not be read! If this behavior is undesired, first unregister
 * any getters before calling readAll, ensuring that the optional write parameter is set to false.
 *
 * Subclasses may also use the SettingsStorage::getOrSetDefault function to add a new key to the settings. This
 * function will search for the key and return its value if it exists. If it does not exist, a new entry in the
 * QSettings file is immediately created with the provided default value. For example:
 *
 *
 *     QVariant out = getOrSetDefault("existingKey",10);
 *     //out contains value of "existingKey", which may not be 10
 *
 *     QVariant out2 = getOrSetDefault("newKey",10);
 *     //out contains 10; "newKey" added to QSettings
 *
 *
 * Finally, subclasses may call SettingsStorage::registerGetter to associate a function with a key. Any
 * subsequent references to that key will call the associated function to retrieve the value. A call to
 * SettingsStorage::save() will retrieve the value from the stored function to save. In this way, the subclass
 * does not have to call SettingsStorage::set every time a value changes. This mechanism only works for
 * single key-value settings, not for array values. Getters may be cleared individually with
 * SettingsStorage::unRegisterGetter or all at once with SettingsStorage::clearGetters. When unregistered
 * or cleared, the getter function is called and the value stored in memory for later use/saving. Optionally,
 * the value can be immediately written to QSettings.
 *
 * A getter function must be a const member function that takes no arguments and returns a type known to QVariant.
 * New types can be made known to QVariant using the Q_DECLARE_METATYPE macro; see the QVariant documentation
 * for details. An example:
 *
 *
 *     class MyClass : public SettingsStorage()
 *     {
 *     public:
 *         MyClass();
 *
 *         int getInt() const { return d_int; }
 *
 *     private:
 *         int d_int = 1;
 *     };
 *
 *     MyClass::MyClass() : SettingsStorage({},false)
 *     {
 *         registerGetter("myInt",this,&MyClass::getInt);
 *         int i = get<int>("myInt");
 *         // i == 1
 *
 *         d_int = 10;
 *         int j = get<int>("myInt");
 *         // j == 10
 *
 *         QVariant k = unRegisterGetter("myInt",false);
 *         //k == 10; do not write 10 to QSettings
 *
 *         d_int = 20;
 *         int l = get<int>("myInt");
 *         //l == 10
 *     }
 *
 *
 */
class SettingsStorage
{
public:

    /*!
     * \brief Used in constructor to indicate whether a hardware subkey is read from settings
     */
    enum Type {
        General, /*!< Use keys list explicitly */
        Hardware /*!< If keys list has 1 entry, look up subKey for this hardware. */
    };

    /*!
     * \brief Constructor
     *
     * Create a QSettings object initialized to the group (and any subgroups) in the keys list with the indicated scope, and reads all values and arrays
     * associated with that (sub)group. If keys is empty, the group is set to "Blackchirp".
     *
     * \param keys The list of group/subgroup keys, passed to QSettings::beginGroup in order
     * \param type If set to SettingsStorage::Hardware and the length of keys is 1, the subKey for the current hardware will be read from settings and selected.
     * \param systemWide If true, use QSettings::SystemScope. Otherwise, use QSettings::UserScope
     */
    SettingsStorage(const QStringList keys = QStringList(), Type type = General, bool systemWide = true);

    /*!
     * \brief Constructor that explicitly sets organization name and application name (used for unit tests; should not be used directly).
     *
     * \param orgName Organization name passed to QSettings constructor
     * \param appName Application name passed to QSettings construstor
     * \param keys The list of group/subgroup keys, passed to QSettings::beginGroup in order
     * \param type If set to SettingsStorage::Hardware and the length of keys is 1, the subKey for the current hardware will be read from settings and selected.
     * \param systemWide If true, use QSettings::SystemScope. Otherwise, use QSettings::UserScope
     */
    SettingsStorage(const QString orgName, const QString appName, const QStringList keys = QStringList(), Type type = General, bool systemWide = true);

    /*!
     * \brief Convenience constructor for accessing General settings (avoiding Hardware behavior)
     *
     * This form of the constructor is intended to give more convenient access to user settings, but it can be used
     * for any set of keys. This constructor will not look up keys associated with hardware.
     *
     * \param systemWide If true, use QSettings::SystemScope. Otherwise, use QSettings::UserScope
     * \param keys The list of group/subgroup keys, passed to QSettings::beginGroup in order
     */
    explicit SettingsStorage(bool systemWide, const QStringList keys = QStringList());

    /*!
     * \brief Convenience constructor for a single key
     *
     * \param key The key for the group in QSettings. If "", will be set to "Blackchirp"
     * \param type If set to Hardware, a subKey will be added (default: "virtual")
     * \param systemWide If true, use QSettings::SystemScope. Otherwise, use QSettings::UserScope
     */
    explicit SettingsStorage(const char *key, Type type = General, bool systemWide = true);

    /*!
     * \brief Convenience constructor for a single key
     *
     * \param key The key for the group in QSettings. If "", will be set to "Blackchirp"
     * \param type If set to Hardware, a subKey will be added (default: "virtual")
     * \param systemWide If true, use QSettings::SystemScope. Otherwise, use QSettings::UserScope
     */
    explicit SettingsStorage(const QString key, Type type = General, bool systemWide = true);

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
     * that getter function will be called. The optional `defaultValue` argument is returned
     * if the key is not found.
     *
     * \param key The key associated with the value
     * \param defaultValue The value returned if key is not present (default: QVariant())
     * \return QVariant The value
     */
    QVariant get(const QString key, const QVariant defaultValue = QVariant()) const;

    /*!
     * \brief Gets the value of a settting. Overloaded function.
     *
     * Attempts to convert the value to type T using QVariant::value<T>(). See Qt
     * documentation for a discussion. The optional `defaultValue` argument is returned
     * if the key is not found. If left blank, a default-constructed value is returned
     * for any missing keys.
     *
     * \param key The key associated with the value
     * \param defaultValue The value returned if key is not present
     * \return T The value, or a default constructed value if the key is not present
     */
    template<typename T>
    inline T get(const QString key, T defaultValue = QVariant().value<T>()) const { return (containsValue(key) ? get(key).value<T>() : defaultValue); };

    /*!
     * \brief Gets values associated with a list of keys. Overloaded function
     *
     * If a key in the list is not found, then it is skipped. The returned map may be empty.
     * Recommended usage:
     *
     *     SettingsStorage s(keys,systemWide);
     *     auto x = get( {"key1","key2","key3"} );
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
     * \return std::vector<SettingsMap> The array value. An empty vector is returned if the key is not found
     */
    std::vector<SettingsMap> getArray(const QString key) const;

    /*!
     * \brief Returns the size of the array value associated with key
     *
     * \param key The key of the array value
     * \return std::size_t The number of maps in the array (0 if key does not exist);
     */
    std::size_t getArraySize(const QString key) const;

    /*!
     * \brief Returns a single map from an array associated with a key
     * \param key The key of the array
     * \param i Index of the desired map
     * \return SettingsMap The selected map, which will be empty if key does not exist or if i is out of bounds for the array
     */
    SettingsMap getArrayMap(const QString key, std::size_t i) const;

    /*!
     * \brief Gets a single value from a map that is part of an array value
     *
     * Calls getArrayMap, then searches for mapKey within the returned map (which may be empty).
     * Returns the stored QVariant or the defaultValue argument, which defaults to QVariant().
     *
     * \param arrayKey The key of the array
     * \param i Index of the desired map
     * \param mapKey Key of the desired value within the map
     * \param defaultValue Value returned if arrayKey does not exist **or** i is out of bounds **or** mapKey does not exist (default: QVariant())
     * \return QVariant contining the desired value or defaultValue
     */
    QVariant getArrayValue(const QString arrayKey, std::size_t i, const QString mapKey, const QVariant defaultValue = QVariant()) const;


    /*!
     * \brief Overloaded function. See SettingsStorage::getArrayValue
     *
     * \param arrayKey The key of the array
     * \param i Index of the desired map
     * \param mapKey Key of the desired value within the map
     * \param defaultValue Value returned if `arrayKey` does not exist **or** `i` is out of bounds **or** `mapKey` does not exist (dedaults to a default-constructed value)
     * \return T The value or the defaultValue, as appropriate
     */
    template<typename T>
    inline T getArrayValue(const QString arrayKey, std::size_t i, const QString mapKey, T defaultValue = QVariant().value<T>()) const {
        auto m = getArrayMap(arrayKey,i);
        auto it = m.find(mapKey);
        if(it != m.end())
            return it->second.value<T>();

        return defaultValue;
    }

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
    bool registerGetter(const QString key, T* obj, Out (T::*getter)() const)
    {
        //cannot register a getter for an array value
        if(d_arrayValues.find(key) != d_arrayValues.end())
            return false;

        auto it = d_values.find(key);
        if(it != d_values.end())
            d_values.erase(it->first);

        auto f = [obj, getter](){ return (obj->*getter)(); };
        return d_getters.emplace(key,f).second;
    }

    /*!
     * \brief Removes a getter function
     *
     * If key matches a previously registered getter, then the getter function is removed from the
     * storage object. The getter itself is called, and the value is stored in the values map with the same
     * key amd returned.
     *
     * An empty QVariant is returned if no getter was found.
     *
     * If the write parameter is true, the current value is written to persistent storage.
     *
     * \param key The key associated with the getter
     * \param write If true, value is written to persistent storage
     * \return QVariant containing return value of getter call, or empty QVariant if no getter was found.
     */
    QVariant unRegisterGetter(const QString key, bool write = true);

    /*!
     * \brief Removes all getter functions
     *
     * All getters are removed and their values are transferred into the values map.
     * If write is true, then the values are also written to persistent storage.
     *
     * \param write If true, values are written to persistent storage
     */
    void clearGetters(bool write = true);


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
    QVariant getOrSetDefault(const QString key, QVariant defaultValue);

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
    bool set(const QString key, const QVariant value, bool write = true);

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
    std::map<QString,bool> setMultiple(const SettingsMap m, bool write = true);

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
    void setArray(const QString key, const std::vector<SettingsMap> &array, bool write = true);


    /*!
     * \brief Appends a new map onto an array value
     *
     * If the key does not match an existing array cariable, it is added. By default,
     * the new array is not written to settings immediately. This is because QSettings
     * essentially requires rewriting the entire array every time, and this function
     * is intended to be called as part of a loop.
     *
     * \param key The key of the array value
     * \param map The new map to append
     * \param write If true, QSettings is updated immediately (default false)
     */
    void appendArrayMap(const QString key, const SettingsMap &map, bool write = false);

    /*!
     * \brief Clears a value and removes it from QSettings
     *
     * This clears a value (or getter) and immediately removes the key from QSettings.
     * If the key is not found, no action is taken.
     *
     * \param key The key to clear
     */
    void clearValue(const QString key);

    /*!
     * \brief Write all values to QSettings.
     */
    void save();

    /*!
     * \brief Reads all values from settings file (see note about getters)
     *
     * This function reads all values from the QSettings storage. It does so by clearing
     * out the values and arrayValues maps, and reading in all keys and array groups found
     * in the settings file.
     *
     * If a getter has been registered, that key will be skipped. First call
     * SettingsStorage::unRegisterGetter or SettingsStorage::clearGetters if you wish to
     * these keys to be re-read.
     *
     */
    void readAll();

private:
    explicit SettingsStorage(const QStringList keys, Type type, QSettings::Scope scope);
    explicit SettingsStorage(const QString orgName, const QString appName, const QStringList keys, Type type, QSettings::Scope scope);

    SettingsMap d_values;

    std::map<QString, SettingsGetter> d_getters;
    std::map<QString,std::vector<SettingsMap>> d_arrayValues;

    QSettings d_settings;

    void writeArray(const QString key);


};

#endif // SETTINGSSTORAGE_H
