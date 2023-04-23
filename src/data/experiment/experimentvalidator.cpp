#include "experimentvalidator.h"

#include <data/storage/blackchirpcsv.h>

ExperimentValidator::ExperimentValidator() : HeaderStorage(BC::Store::Validator::key)
{

}

bool ExperimentValidator::validate(const QString key, const QVariant val)
{
    bool out = true;

    if(!d_valMap.empty())
    {
        auto l = key.split(".",Qt::SkipEmptyParts);
        if(l.size() >= 2)
        {
            bool ok = false;
            double v = val.toDouble(&ok);
            if(ok)
            {
                auto objKey = l.constFirst();
                auto valKey = l.constLast();

                auto it = d_valMap.find(objKey);
                if(it != d_valMap.end())
                {
                    auto &om = it->second;
                    auto it2 = om.find(valKey);
                    if(it2 != om.end())
                    {
                        auto min = it2->second.first;
                        auto max = it2->second.second;
                        if( (v < min) || (v > max) )
                        {
                            out = false;
                            d_errorString = QString("Reading for %1 (%2) is outside the allowed range (%3 - %4). Aborting.")
                                    .arg(key).arg(val.toString()).
                                    arg(QVariant(min).toString()).arg(QVariant(max).toString());
                        }
                    }
                }
            }
        }
    }

    return out;
}

void ExperimentValidator::storeValues()
{
    using namespace BC::Store::Validator;
    std::size_t index = 0;
    for(auto &[ok,m] : d_valMap)
    {
        for(auto &[vk,pair] : m)
        {
            storeArrayValue(condition,index,objKey,ok);
            storeArrayValue(condition,index,valKey,vk);
            storeArrayValue(condition,index,min,pair.first);
            storeArrayValue(condition,index,max,pair.second);
            ++index;
        }
    }
}

void ExperimentValidator::retrieveValues()
{
    using namespace BC::Store::Validator;
    auto size = arrayStoreSize(condition);
    for(std::size_t i = 0; i<size; ++i)
    {
        auto ok = retrieveArrayValue(condition,i,objKey,QString(""));
        auto vk = retrieveArrayValue(condition,i,valKey,QString(""));
        auto minv = retrieveArrayValue(condition,i,min,0.0);
        auto maxv = retrieveArrayValue(condition,i,max,1.0);

        auto it = d_valMap.find(ok);
        if(it != d_valMap.end())
            it->second.insert({vk,{minv,maxv}});
        else
            d_valMap.insert( {ok, {{vk,{minv,maxv}}}} );
    }
}
