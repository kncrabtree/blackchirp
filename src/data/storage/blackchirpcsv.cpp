#include "blackchirpcsv.h"

//#include <gui/plot/blackchirpplotcurve.h>

BlackchirpCSV::BlackchirpCSV()
{

}

bool BlackchirpCSV::writeXY(QIODevice &device, const QVector<QPointF> d, const QString prefix)
{
    if(!device.open(QIODevice::WriteOnly))
        return false;

    QTextStream t(&device);
    t.setRealNumberNotation(d_notation);
    t.setRealNumberPrecision(d_precision);

    if(prefix.isEmpty())
        t << x << del << y;
    else
        t << prefix << sep << x << del << prefix << sep << y;

    for(auto it = d.constBegin(); it != d.constEnd(); it++)
        t << nl << it->x() << del << it->y();

    return true;
}
bool BlackchirpCSV::writeMultiple(QIODevice &device, const std::vector<QVector<QPointF> > &l, const std::vector<QString> &n)
{
    if(!device.open(QIODevice::WriteOnly|QIODevice::Text))
        return false;

    QTextStream t(&device);
    t.setRealNumberNotation(d_notation);
    t.setRealNumberPrecision(d_precision);

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
                t << l.at(j).at(i).x() << del << l.at(j).at(i).y();
            else
                t << del;
        }

    }

    return true;
}
