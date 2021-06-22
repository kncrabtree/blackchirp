#include "settingsstorage.h"

SettingsStorage::SettingsStorage(const QStringList keys, Type type, QSettings::Scope scope) : d_settings{scope,QCoreApplication::organizationName(),QCoreApplication::applicationName()}
{
    d_settings.setFallbacksEnabled(false);

    if(keys.isEmpty())
        d_settings.beginGroup(QString("Blackchirp"));
    else
    {
        if( (type == Hardware) && (keys.size() == 1) )
        {
            d_settings.beginGroup(keys.first());
            d_settings.beginGroup(d_settings.value("subKey","virtual").toString());
        }
        else
        {
            for(auto k : keys)
                d_settings.beginGroup(k);
        }
    }

    readAll();
}

SettingsStorage::SettingsStorage(const QString orgName, const QString appName, const QStringList keys, Type type, QSettings::Scope scope) : d_settings{scope,orgName,appName}
{
    d_settings.setFallbacksEnabled(false);

    if(keys.isEmpty())
        d_settings.beginGroup(QString("Blackchirp"));
    else
    {
        if( (type == Hardware) && (keys.size() == 1) )
        {
            d_settings.beginGroup(keys.first());
            d_settings.beginGroup(d_settings.value("subKey","virtual").toString());
        }
        else
        {
            for(auto k : keys)
                d_settings.beginGroup(k);
        }
    }

    readAll();
}

SettingsStorage::SettingsStorage(const QStringList keys, Type type, bool systemWide) :
    SettingsStorage(keys, type, (systemWide ? QSettings::SystemScope : QSettings::UserScope))
{

}

SettingsStorage::SettingsStorage(const QString orgName, const QString appName, const QStringList keys, Type type, bool systemWide) :
    SettingsStorage(orgName, appName, keys, type, (systemWide ? QSettings::SystemScope : QSettings::UserScope))
{

}

SettingsStorage::SettingsStorage(bool systemWide, const QStringList keys) :
    SettingsStorage(keys, General, (systemWide ? QSettings::SystemScope : QSettings::UserScope))
{

}

SettingsStorage::SettingsStorage(const char *key, SettingsStorage::Type type, bool systemWide) :
    SettingsStorage(QStringList(QString(key)),type,(systemWide ? QSettings::SystemScope : QSettings::UserScope))
{

}

SettingsStorage::SettingsStorage(const QString key, Type type, bool systemWide) :
    SettingsStorage(QStringList(key),type,(systemWide ? QSettings::SystemScope : QSettings::UserScope))
{

}

SettingsStorage::~SettingsStorage()
{
    save();
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

QVariant SettingsStorage::get(const QString key, const QVariant &defaultValue) const
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
    return defaultValue;
}

SettingsMap SettingsStorage::getMultiple(const std::vector<QString> keys) const
{
    SettingsMap out;
    for(auto key : keys)
    {
        if(containsValue(key))
            out.insert({key,get(key)});
    }

    return out;
}

std::vector<SettingsMap> SettingsStorage::getArray(const QString key) const
{
    if(containsArray(key))
        return d_arrayValues.at(key);
    else
        return std::vector<SettingsMap>();
}

std::size_t SettingsStorage::getArraySize(const QString key) const
{
    if(containsArray(key))
        return d_arrayValues.at(key).size();

    return 0;
}

SettingsMap SettingsStorage::getArrayMap(const QString key, std::size_t i) const
{
    auto v = getArray(key);
    if(i < v.size())
        return v.at(i);

    return SettingsMap();
}

QVariant SettingsStorage::getArrayValue(const QString arrayKey, std::size_t i, const QString mapKey, const QVariant defaultValue) const
{
    auto m = getArrayMap(arrayKey,i);
    auto it = m.find(mapKey);
    if(it != m.end())
        return it->second;

    return defaultValue;
}

QVariant SettingsStorage::unRegisterGetter(const QString key, bool write)
{
    auto it = d_getters.find(key);
    if(it == d_getters.end())
        return QVariant();

    QVariant out = it->second();
    d_values.insert_or_assign(it->first,out);

    if(write)
    {
        d_settings.setValue(it->first,out);
        d_settings.sync();
    }

    d_getters.erase(it);
    return out;
}

void SettingsStorage::clearGetters(bool write)
{
    auto it = d_getters.begin();
    while(it != d_getters.end())
    {
        QVariant v= it->second();
        d_values.insert_or_assign(it->first,v);

        if(write)
            d_settings.setValue(it->first,v);

        it = d_getters.erase(it);
    }

    if(write)
        d_settings.sync();
}

QVariant SettingsStorage::getOrSetDefault(const QString key, const QVariant defaultValue)
{
    if(containsValue(key))
    {
        auto out = get(key);
        if(out.isValid())
            return out;
    }

    //ensure this key doesn't match an array
    if(containsArray(key))
        return QVariant();

    //if we reach this point, there is no value. Store it in settings.
    set(key,defaultValue,true);
    return defaultValue;
}

void SettingsStorage::setDefault(const QString key, const QVariant defaultValue)
{
    if(containsValue(key) || containsArray(key))
        return;

    set(key,defaultValue,true);
}

bool SettingsStorage::set(const QString key, const QVariant &value, bool write)
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

bool SettingsStorage::setArrayValue(const QString arrayKey, std::size_t i, const QString key, const QVariant &value, bool write)
{
    if(!containsArray(arrayKey))
        return false;

    if(i >= d_arrayValues.at(arrayKey).size())
        return false;

    d_arrayValues[arrayKey][i].insert_or_assign(key,value);

    if(write)
        writeArray(arrayKey);

    return true;
}

void SettingsStorage::appendArrayMap(const QString key, const SettingsMap &map, bool write)
{
    if(containsArray(key))
        d_arrayValues[key].push_back(map);
    else
        d_arrayValues.insert({key, {map}});

    if(write)
        writeArray(key);
}

void SettingsStorage::clearValue(const QString key)
{
    auto it = d_values.find(key);
    if(it != d_values.end())
    {
        d_settings.remove(key);
        d_values.erase(it);
        d_settings.sync();
        return;
    }

    auto it2 = d_getters.find(key);
    if(it2 != d_getters.end())
    {
        d_settings.remove(key);
        d_getters.erase(it2);
        d_settings.sync();
        return;
    }

}

std::map<QString,bool> SettingsStorage::setMultiple(const std::map<QString, QVariant> m, bool write)
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

void SettingsStorage::setArray(const QString key, const std::vector<std::map<QString, QVariant> > &array, bool write)
{
    //passing an empty array will erase the value from the settings array and from QSettings
    d_arrayValues.insert_or_assign(key,array);
    if(write)
    {
        if(array.empty())
        {
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


void SettingsStorage::writeArray(const QString key)
{
    d_settings.remove(key);
    d_settings.beginWriteArray(key);
    auto l = d_arrayValues.at(key);
    for(std::size_t i = 0; i < l.size(); ++i)
    {
        d_settings.setArrayIndex(i);
        auto m = l.at(i);
        for(auto it = m.cbegin(); it != m.cend(); ++it)
            d_settings.setValue(it->first,it->second);
    }
    d_settings.endArray();
    d_settings.sync();
}

void SettingsStorage::save()
{
    for(auto it = d_values.cbegin(); it != d_values.cend(); ++it)
        d_settings.setValue(it->first,it->second);

    for(auto it = d_getters.cbegin(); it != d_getters.cend(); ++it)
        d_settings.setValue(it->first,it->second());

    for(auto it = d_arrayValues.cbegin(); it != d_arrayValues.cend(); ++it)
    {
        writeArray(it->first);
    }

    d_settings.sync();
}

void SettingsStorage::readAll()
{
    d_values.clear();
    d_arrayValues.clear();

    auto keys = d_settings.childKeys();
    auto groups = d_settings.childGroups();

    for(auto g : groups)
    {
        int n = d_settings.beginReadArray(g);
        if(n > 0) //key points to an array
        {
            std::vector<SettingsMap> l;
            l.reserve(n);
            for(int i=0; i<n; ++i)
            {
                d_settings.setArrayIndex(i);
                auto arrayKeys = d_settings.childKeys();
                SettingsMap m;
                for(auto k2 : arrayKeys)
                    m.insert_or_assign(k2,d_settings.value(k2));
                l.push_back(m);
            }
            d_settings.endArray();
            d_arrayValues.insert_or_assign(g,l);
        }
        else //key points to a non-array subgroup
        {
            d_settings.endArray();
        }


    }

    for(auto k : keys)
    {
        if(d_getters.find(k) == d_getters.end())
            d_values.insert_or_assign(k,d_settings.value(k));
    }
}


