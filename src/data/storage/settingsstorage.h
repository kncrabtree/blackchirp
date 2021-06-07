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

    QVariant get(const QString key) const { return d_values.value(key); };

    template<typename T>
    T get(const QString key) const { return d_values.value(key).value<T>(); }

    /*!
     * \brief Registers a getter function for a given setting
     *
     * If a getter is associated with a key, then the value is removed from the values
     * dictionary, and instead the getter function will be called to access the value.
     *
     * The getter function will be called when saving the settings to disk. By default,
     * the settings file will not be immediately updated. Set the write parameter to true
     * to immediately write the current value from the getter function to disk.
     *
     * This function only works for single values, not arrays.
     *
     * \param key The key for the value to be stored
     * \param getter A function pointer to a QVariant
     * \return
     */

    template<class T, typename Out>
    QVariant registerGetter(QString key, T* obj, Out (T::*getter)());



private:
    QString d_mainKey;
    QStringList d_subKeys;
    bool d_systemWide;

    QVariantMap d_values;

    QMap<QString, Getter> d_getters;
    QMap<QString,QVariantList> d_arrayValues;
    QSettings d_settings;

    int test() { return 1; }
    double test2() { return 2.0; }


    void readAll();

};


#endif // SETTINGSSTORAGE_H
