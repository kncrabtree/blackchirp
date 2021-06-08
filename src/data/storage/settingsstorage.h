#ifndef SETTINGSSTORAGE_H
#define SETTINGSSTORAGE_H

#include <QString>
#include <QVariant>
#include <QList>
#include <QSettings>

using Getter = std::function<QVariant()>;

class SettingsStorage
{
private:
    SettingsStorage(const QString mainKey, const QStringList subKeys, QSettings::Scope scope);
public:
    SettingsStorage(const QString mainKey, const QStringList subKeys = QStringList(), bool systemWide = true);
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
    T get(const QString key) const;



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
     * \return QVariant containing the original value stored. If no value was stored, returns QVariant()
     */
    template<class T, typename Out>
    bool registerGetter(QString key, T* obj, Out (T::*getter)() const);


    /*!
     * \brief registerDefaultSetting
     * \param key
     * \param defaultValue
     * \return
     */
    QVariant getDefaultSetting(QString key, QVariant defaultValue);


    bool set(QString key, QVariant value, bool write = true);



private:
    QString d_mainKey;
    QStringList d_subKeys;
    bool d_systemWide;

    std::map<QString,QVariant> d_values;

    std::map<QString, Getter> d_getters;
    std::map<QString,std::vector<QVariant>> d_arrayValues;
    QSettings d_settings;

    void readAll();

};

#endif // SETTINGSSTORAGE_H
