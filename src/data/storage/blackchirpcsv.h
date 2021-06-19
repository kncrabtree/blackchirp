#ifndef BLACKCHIRPCSV_H
#define BLACKCHIRPCSV_H

#include <QIODevice>
#include <QTextStream>
#include <QVector>
#include <QPointF>

//class BlackchirpPlotCurve;

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

    void setScientificNotation(bool sci = true) { sci ? d_notation = QTextStream::ScientificNotation :
                                                d_notation = QTextStream::FixedNotation; }
    void setPrecision(int p) { d_precision = qBound(0,p,15); }

    bool writeXY(QIODevice &device, const QVector<QPointF> d, const QString prefix = "");
    bool writeMultiple(QIODevice &device, const std::vector<QVector<QPointF>> &l, const std::vector<QString> &n = {});

    template<typename T>
    bool writeY(QIODevice &device, const QVector<T> d, QString title="")
    {
        if(!device.open(QIODevice::WriteOnly | QIODevice::Text))
            return false;

        QTextStream t(&device);
        t.setRealNumberNotation(d_notation);
        t.setRealNumberPrecision(d_precision);

        if(title.isEmpty())
            t << y;
        else
            t << title;

        for(auto it = d.constBegin(); it != d.constEnd(); it++)
            t << nl << *it;

        return true;
    }

    template<typename T>
    bool writeYMultiple(QIODevice &device, std::initializer_list<QString> titles, std::initializer_list<QVector<T>> l)
    {
        if(titles.size() != l.size())
            return false;

        if(!device.open(QIODevice::WriteOnly | QIODevice::Text))
            return false;

        QTextStream t(&device);
        t.setRealNumberNotation(d_notation);
        t.setRealNumberPrecision(d_precision);

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
                    t << list.at(j).at(i);
            }
        }

        return true;

    }

    const QString del{","};
    const QString nl{"\n"};
    const QString x{"x"};
    const QString y{"y"};
    const QString sep{"_"};

private:
    QTextStream::RealNumberNotation d_notation{QTextStream::FixedNotation};
    int d_precision{6};
};

#endif // BLACKCHIRPCSV_H
