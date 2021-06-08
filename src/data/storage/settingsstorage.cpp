#include "settingsstorage.h"

#include <QApplication>

SettingsStorage::SettingsStorage(const QStringList keys, QSettings::Scope scope) : d_settings{scope,QApplication::organizationName(),QApplication::applicationName()}
{
    for(auto k : keys)
        d_settings.beginGroup(k);

    readAll();
}

SettingsStorage::SettingsStorage(const QStringList keys, bool systemWide) : SettingsStorage(keys, systemWide ? QSettings::SystemScope : QSettings::UserScope)
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

auto SettingsStorage::get(const std::vector<QString> keys) const
{
    std::map<QString, QVariant> out;
    for(auto key : keys)
    {
        if(containsValue(key))
            out.insert({key,get(key)});
    }

    return out;
}

auto SettingsStorage::getArray(const QString key) const
{
    if(containsArray(key))
        return d_arrayValues.at(key);
    else
        return std::vector<std::map<QString,QVariant>>();
}

auto SettingsStorage::getArrayValue(const QString key, std::size_t i) const
{
    auto v = getArray(key);
    if(i < v.size())
        return v.at(i);

    return std::map<QString,QVariant>();
}

QVariant SettingsStorage::getOrSetDefault(QString key, QVariant defaultValue)
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

auto SettingsStorage::setMultiple(std::map<QString, QVariant> m, bool write)
{
    std::map<QString,bool> out;
    for( auto it = m.cbegin(); it != m.cend(); ++it )
    {
        bool success = set(it->first,it->second);
        out.insert({it->first,success});
        if(success && write)
            d_settings.setValue(it->first,it->second);
    }

    if(write)
        d_settings.sync();

    return out;
}

void SettingsStorage::setArray(QString key, std::vector<std::map<QString, QVariant> > array, bool write)
{
    //passing an empty array will erase the value from the settings array and from QSettings
    d_arrayValues.insert_or_assign(key,array);
    if(write)
    {
        if(array.empty())
        {
            if(d_settings.contains(key))
                d_settings.remove(key);
        }
        else
        {
            d_settings.beginWriteArray(key,array.size());
            for(std::size_t i = 0; i < array.size(); ++i)
            {
                d_settings.setArrayIndex(i);
                auto map = array.at(i);
                for(auto it = map.cbegin(); it != map.cend(); ++it)
                    d_settings.setValue(it->first,it->second);
            }
            d_settings.endArray();
        }
        d_settings.sync();
    }
}

void SettingsStorage::save()
{
    for(auto it = d_values.cbegin(); it != d_values.cend(); ++it)
        d_settings.setValue(it->first,it->second);

    for(auto it = d_getters.cbegin(); it != d_getters.cend(); ++it)
        d_settings.setValue(it->first,it->second());

    for(auto it = d_arrayValues.cbegin(); it != d_arrayValues.cend(); ++it)
    {
        auto l = it->second;
        d_settings.beginWriteArray(it->first);
        for(std::size_t i = 0; i < l.size(); ++i)
        {
            d_settings.setArrayIndex(i);
            auto m = l.at(i);
            for(auto it2 = m.cbegin(); it2 != m.cend(); ++it)
                d_settings.setValue(it2->first,it2->second);
        }
        d_settings.endArray();
    }

    d_settings.sync();
}


template<typename T>
T SettingsStorage::get(const QString key) const
{
    return get(key).value<T>();
}

void SettingsStorage::readAll()
{
    auto keys = d_settings.childKeys();

    for(auto k : keys)
    {
        int n = d_settings.beginReadArray(k);
        if(n > 0) //key points to an array
        {
            std::vector<std::map<QString,QVariant>> l;
            l.reserve(n);
            for(int i=0; i<n; ++i)
            {
                d_settings.setArrayIndex(i);
                auto arrayKeys = d_settings.childKeys();
                std::map<QString,QVariant> m;
                for(auto k2 : arrayKeys)
                    m.insert_or_assign(k2,d_settings.value(k2));
                l.push_back(m);
            }
            d_settings.endArray();
            d_arrayValues.insert_or_assign(k,l);
        }
        else //key points to a single value
        {
            d_settings.endArray();
            d_values.insert_or_assign(k,d_settings.value(k));
        }


    }
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
