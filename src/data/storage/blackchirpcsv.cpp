#include "blackchirpcsv.h"

//#include <gui/plot/blackchirpplotcurve.h>

BlackchirpCSV::BlackchirpCSV()
{

}

bool BlackchirpCSV::writeXY(QIODevice &device, const QVector<QPointF> d, const QString prefix)
{
    using namespace BC::CSV;

    QTextStream t(&device);

    if(prefix.isEmpty())
        t << x << del << y;
    else
        t << prefix << sep << x << del << prefix << sep << y;

    for(auto it = d.constBegin(); it != d.constEnd(); it++)
        t << nl << QVariant{it->x()}.toString() << del << QVariant{it->y()}.toString();

    return true;
}

bool BlackchirpCSV::writeMultiple(QIODevice &device, const std::vector<QVector<QPointF> > &l, const std::vector<QString> &n)
{
    using namespace BC::CSV;

    QTextStream t(&device);

    //create a title for every element of l, and determine max size
    int max = 0;
    for(std::size_t i=0; i<l.size(); ++i)
    {
        max = qMax(max,l.at(i).size());
        if(i > 0)
            t << del;
        if(i < n.size())
            t << n.at(i) << sep << x << del << n.at(i) << sep << y;
        else
            t << x << i << del << y << i;
    }

    for(int i = 0; i<max; ++i)
    {
        t << nl;

        for(std::size_t j = 0; j < l.size(); ++j)
        {
            if(j > 0)
                t << del;
            if(i <l.at(j).size())
                t << QVariant{l.at(j).at(i).x()}.toString() << del << QVariant{l.at(j).at(i).y()}.toString();
            else
                t << del;
        }

    }
    return true;
}

bool BlackchirpCSV::writeHeader(QIODevice &device, const std::multimap<QString, std::tuple<QString, QString, QString, QString, QString> > header)
{
    using namespace BC::CSV;

    QTextStream t(&device);

    t << ok << del << ak << del << ai << del << vk << del << vv << del << vu;

    for(auto it = header.cbegin(); it != header.cend(); ++it)
        t << nl << it->first << del
          << std::get<0>(it->second) << del
          << std::get<1>(it->second) << del
          << std::get<2>(it->second) << del
          << std::get<3>(it->second) << del
          << std::get<4>(it->second);

    return true;
}

void BlackchirpCSV::writeLine(QTextStream &t, const QVariantList l)
{
    using namespace BC::CSV;

    if(l.isEmpty())
        return;

    t << l.constFirst().toString();
    int num = l.size();
    for(int i=1; i<num; i++)
        t << del << l.at(i).toString();
    t << nl;
}
