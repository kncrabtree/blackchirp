#include "settingsstorage.h"

#include <QApplication>

SettingsStorage::SettingsStorage(const QString mainKey, const QStringList subKeys, QSettings::Scope scope) : d_mainKey(mainKey), d_subKeys(subKeys), d_settings{scope,QApplication::organizationName(),QApplication::applicationName()}
{
    if(!d_mainKey.isEmpty())
        d_settings.beginGroup(mainKey);
    for(auto sk : subKeys)
        d_settings.beginGroup(sk);

    readAll();

    registerGetter("test",this,&SettingsStorage::test);
    registerGetter("test2",this,&SettingsStorage::test2);
}

SettingsStorage::SettingsStorage(const QString mainKey, const QStringList subKeys, bool systemWide) : SettingsStorage(mainKey,subKeys, systemWide ? QSettings::SystemScope : QSettings::UserScope)
{

}

void SettingsStorage::readAll()
{
    auto arrays = d_settings.childGroups();
    auto keys = d_settings.childKeys();

    for(auto a : arrays)
    {
        int n = d_settings.beginReadArray(a);
        QVariantList l;
        for(int i=0; i<n; ++i)
        {
            d_settings.setArrayIndex(i);
            auto arrayKeys = d_settings.childKeys();
            QVariantMap m;
            for(auto k : arrayKeys)
                m.insert(k,d_settings.value(k));
            l << m;
        }
        d_arrayValues.insert(a,l);
        d_settings.endArray();
    }

    for(auto k : keys)
        d_values.insert(k,d_settings.value(k));
}

template<class T, typename Out>
QVariant SettingsStorage::registerGetter(QString key, T *obj, Out (T::*getter)())
{
    QVariant o;
    if(d_values.contains(key))
        o = d_values.take(key);

    auto f = [obj, getter](){ return (obj->*getter)(); };
    d_getters.insert(key,f);

    return o;
}
