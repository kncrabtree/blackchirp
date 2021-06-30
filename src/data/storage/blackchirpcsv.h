#ifndef BLACKCHIRPCSV_H
#define BLACKCHIRPCSV_H

#include <QIODevice>
#include <QTextStream>
#include <QVector>
#include <QPointF>

//class BlackchirpPlotCurve;

namespace BC::CSV {

static const QString del{","};
static const QString nl{"\n"};
static const QString x{"x"};
static const QString y{"y"};
static const QString sep{"_"};

static const QString ok{"ObjKey"};
static const QString ak{"ArrayKey"};
static const QString ai{"ArrayIndex"};
static const QString vk{"ValueKey"};
static const QString vv{"Value"};
static const QString vu{"Units"};

}

/*!
 * \brief Convenience class for reading/writing CSV files
 *
 *
 *
 */
class BlackchirpCSV
{
public:
    BlackchirpCSV();

    bool writeXY(QIODevice &device, const QVector<QPointF> d, const QString prefix = "");
    bool writeMultiple(QIODevice &device, const std::vector<QVector<QPointF>> &l, const std::vector<QString> &n = {});

    template<typename T>
    bool writeY(QIODevice &device, const QVector<T> d, QString title="")
    {
        using namespace BC::CSV;
        if(!device.open(QIODevice::WriteOnly | QIODevice::Text))
            return false;

        QTextStream t(&device);

        if(title.isEmpty())
            t << y;
        else
            t << title;

        for(auto it = d.constBegin(); it != d.constEnd(); it++)
            t << nl << QVariant{*it}.toString();

        device.close();
        return true;
    }

    template<typename T>
    bool writeYMultiple(QIODevice &device, std::initializer_list<QString> titles, std::initializer_list<QVector<T>> l)
    {
        using namespace BC::CSV;
        if(titles.size() != l.size())
            return false;

        if(!device.open(QIODevice::WriteOnly | QIODevice::Text))
            return false;

        QTextStream t(&device);
        QVector<QVector<T>> list{l};

        auto it = titles.begin();
        int max = 0;
        for(int i = 0; it != titles.end(); ++it, ++i)
        {
            if(it != titles.begin())
                t << del;
            t << *it;
            max = qMax(max,list.at(i).size());
        }

        for(int i=0; i<max; ++i)
        {
            t << nl;
            for(int j=0; j<list.size(); j++)
            {
                if(j>0)
                    t << del;
                if(i < list.at(j).size())
                    t << QVariant{list.at(j).at(i)}.toString();
            }
        }

        device.close();
        return true;

    }

    bool writeHeader(QIODevice &device, const std::multimap<QString,std::tuple<QString,QString,QString,QString,QString>> header);

};

#endif // BLACKCHIRPCSV_H
