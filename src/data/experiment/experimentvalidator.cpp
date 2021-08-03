#include "experimentvalidator.h"

#include <data/storage/blackchirpcsv.h>

ExperimentValidator::ExperimentValidator()
{

}

ExperimentValidator::ExperimentValidator(BlackchirpCSV *csv, int num, QString path)
{
    auto d = BlackchirpCSV::exptDir(num,path);
    QFile val = d.absoluteFilePath(BC::CSV::validationFile);
    if(val.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        while(!val.atEnd())
        {
            auto l = csv->readLine(val);
            if(l.isEmpty())
                continue;

            if(l.constFirst().toString() == QString("objKey"))
                continue;

            if(l.size() != 4)
                continue;

            auto objKey = l.at(0).toString();
            auto valKey = l.at(1).toString();
            auto min = l.at(2).toDouble();
            auto max = l.at(3).toDouble();

            auto it = d_valMap.find(objKey);
            if(it != d_valMap.end())
                it->second.insert({valKey,{min,max}});
            else
                d_valMap.insert( {objKey, {{valKey,{min,max}}}} );
        }
    }
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

bool ExperimentValidator::saveValidation(int num)
{
    QDir d(BlackchirpCSV::exptDir(num));
    QFile val(d.absoluteFilePath(BC::CSV::validationFile));
    if(!val.open(QIODevice::WriteOnly|QIODevice::Text))
        return false;

    QTextStream t(&val);
    BlackchirpCSV::writeLine(t,{"objKey","valKey","min","max"});
    for(auto objit = d_valMap.cbegin(); objit != d_valMap.cend(); ++objit)
    {
        auto m = objit->second;
        for(auto valit = m.cbegin(); valit != m.cend(); ++valit)
            BlackchirpCSV::writeLine(t,{objit->first,valit->first,valit->second.first,valit->second.second});
    }

    return true;
}
