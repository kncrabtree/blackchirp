#include "experimentvalidator.h"

ExperimentValidator::ExperimentValidator()
{

}

bool ExperimentValidator::validate(const QString key, const QVariant val)
{
    bool out = true;

    if(!d_valMap.empty())
    {
        auto l = key.split(".",QString::SkipEmptyParts);
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
                            d_errorString = QString("Reading for %1 (%2) is outside the allowed range. (%3 - %4). Aborting.")
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
