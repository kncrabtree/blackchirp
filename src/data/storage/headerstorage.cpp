#include "headerstorage.h"

HeaderStorage::HeaderStorage(const QString objKey) : d_headerKey{objKey}
{

}

int HeaderStorage::headerIndex() const
{
    auto p = BC::Key::parseKey(d_headerKey);
    return p.second;
}

void HeaderStorage::store(const QString key, const QVariant val, const QString unit)
{
    ValueUnit v{val,unit};
    d_values.insert_or_assign(key,v);
}

void HeaderStorage::storeArrayValue(const QString arrayKey, std::size_t index, const QString key, const QVariant val, const QString unit)
{
    auto it = d_arrayValues.find(arrayKey);
    if(it != d_arrayValues.end())
    {
        while(index >= it->second.size())
            it->second.push_back({});
        ValueUnit v{val,unit};
        it->second[index].insert_or_assign(key,v);
        return;
    }

    HeaderArray l;
    while(index >= l.size())
        l.push_back({});
;
    ValueUnit v{val,unit};
    l[index].insert_or_assign(key,v);

    d_arrayValues.emplace(arrayKey,l);
}

std::size_t HeaderStorage::arrayStoreSize(const QString key) const
{
    auto it = d_arrayValues.find(key);
    if(it != d_arrayValues.end())
        return it->second.size();

    return 0;
}

HeaderStorage::HeaderStrings HeaderStorage::getStrings()
{

    prepareToStore();

    storeValues();

    HeaderStrings out;
    for(auto it = d_values.begin(); it != d_values.end(); it++)
    {
        auto key = it->first;
        auto [val,unit] = it->second;
        out.insert({d_headerKey,{QString(""),QString(""),key,val.toString(),unit}});
    }

    for(auto it = d_arrayValues.begin(); it != d_arrayValues.end(); ++it)
    {
        auto arrKey = it->first;
        for(size_t i=0; i < it->second.size(); ++i)
        {
            auto &m = it->second.at(i);
            for(auto it2 = m.begin(); it2 != m.end(); ++it2)
            {
                auto key = it2->first;
                auto [val,unit] = it2->second;
                out.insert({d_headerKey,{arrKey,QString::number(i),key,val.toString(),unit}});
            }
        }
    }

    for(auto child : d_children)
        out.merge(child->getStrings());

    d_values.clear();
    d_arrayValues.clear();

    return out;
}

void HeaderStorage::prepareToStore()
{
    d_children.clear();

    prepareChildren();

    for(auto child : d_children)
        child->prepareToStore();

}

bool HeaderStorage::storeLine(const QVariantList l)
{
    //no check to make sure l.size() == 6! This should
    //be performed before calling this function.

    auto objKey{l.at(0).toString()};
    auto arrKey{l.at(1).toString()};
    auto index{l.at(2).toString()};
    auto key{l.at(3).toString()};
    auto val{l.at(4).toString()};
    auto unit{l.at(5).toString()};

    if(objKey != d_headerKey)
    {
        for(auto child : d_children)
        {
            if(child->storeLine(l))
                return true;
        }

        return false;
    }

    if(key.isEmpty() || val.isEmpty())
        return false;

    if(!arrKey.isEmpty())
    {
        bool ok = false;
        std::size_t i = index.toUInt(&ok);
        if(!ok)
            return false;

        storeArrayValue(arrKey,i,key,QVariant::fromValue(val),unit);
    }
    else
        store(key,QVariant::fromValue(val),unit);

    return true;
}

void HeaderStorage::readComplete()
{
    retrieveValues();

    for(auto child : d_children)
        child->readComplete();
}

void HeaderStorage::addChild(HeaderStorage *other)
{
    if(other)
        d_children.push_back(other);
}

HeaderStorage *HeaderStorage::removeChild(HeaderStorage *child)
{
    HeaderStorage *out = nullptr;

    for(auto it = d_children.begin(); it != d_children.end(); ++it)
    {
        if((*it) == child)
            return *d_children.erase(it);
    }

    return out;
}
