#include "settingsstorage.h"

#include <QApplication>

SettingsStorage::SettingsStorage(const QString mainKey, const QStringList subKeys, QSettings::Scope scope) : d_mainKey(mainKey), d_subKeys(subKeys), d_settings{scope,QApplication::organizationName(),QApplication::applicationName()}
{
    if(!d_mainKey.isEmpty())
        d_settings.beginGroup(mainKey);
    for(auto sk : subKeys)
        d_settings.beginGroup(sk);

    readAll();
}

SettingsStorage::SettingsStorage(const QString mainKey, const QStringList subKeys, bool systemWide) : SettingsStorage(mainKey,subKeys, systemWide ? QSettings::SystemScope : QSettings::UserScope)
{

}

bool SettingsStorage::containsValue(const QString key) const
{
    ///TODO: c++20 will have std::map.contains; can just return d_values.contains(key) || d_getters.contains(key)

    //search for key in values map
    auto it = d_values.find(key);
    if(it != d_values.end())
        return true;

    //search for key in getters map
    auto it2 = d_getters.find(key);
    if(it2 != d_getters.end())
        return true;

    //if we've made it here, the key wasn't found
    return false;
}

bool SettingsStorage::containsArray(const QString key) const
{
    return (d_arrayValues.find(key) != d_arrayValues.end());
}

QVariant SettingsStorage::get(const QString key) const
{
    //search for key in values map; return if found
    auto it = d_values.find(key);
    if(it != d_values.end())
        return it->second;

    //search for key in getters map; call getter if found
    auto it2 = d_getters.find(key);
    if(it2 != d_getters.end())
        return it2->second();

    //key wasn't found; return empty QVariant
    return QVariant();
}

QVariant SettingsStorage::getDefaultSetting(QString key, QVariant defaultValue)
{
    if(containsValue(key))
        return get(key);

    //if we reach this point, there is no value. Store it in settings.
    set(key,defaultValue,true);
    return defaultValue;
}

bool SettingsStorage::set(QString key, QVariant value, bool write)
{
    //make sure there is no getter or array associated with this key
    if(containsArray(key))
        return false;

    if(d_getters.find(key) != d_getters.end())
        return false;

    d_values.insert_or_assign(key,value);

    if(write)
    {
        d_settings.setValue(key,value);
        d_settings.sync();
    }

    return true;
}


template<typename T>
T SettingsStorage::get(const QString key) const
{
    return get(key).value<T>();
}

void SettingsStorage::readAll()
{
    auto arrays = d_settings.childGroups();
    auto keys = d_settings.childKeys();

    for(auto a : arrays)
    {
        int n = d_settings.beginReadArray(a);
        std::vector<QVariant> l;
        for(int i=0; i<n; ++i)
        {
            d_settings.setArrayIndex(i);
            auto arrayKeys = d_settings.childKeys();
            std::map<QString,QVariant> m;
            for(auto k : arrayKeys)
                m.insert_or_assign(k,d_settings.value(k));
            l.push_back(QMap(m));
        }
        d_settings.endArray();
        d_arrayValues.insert_or_assign(a,l);
    }

    for(auto k : keys)
        d_values.insert_or_assign(k,d_settings.value(k));
}

template<class T, typename Out>
bool SettingsStorage::registerGetter(QString key, T *obj, Out (T::*getter)() const)
{
    //cannot register a getter for an array value
    if(d_arrayValues.find(key) != d_arrayValues.end())
        return false;

    auto it = d_values.find(key);
    if(it != d_values.end())
        d_values.erase(it->first);

    auto f = [obj, getter](){ return (obj->*getter)(); };
    d_getters.emplace({key,f});

    return true;
}
