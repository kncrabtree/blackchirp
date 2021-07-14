#ifndef BLACKCHIRPCSV_H
#define BLACKCHIRPCSV_H

#define _STR(x) #x
#define STRINGIFY(x) _STR(x)

#include <QIODevice>
#include <QTextStream>
#include <QVector>
#include <QPointF>
#include <QDir>
#include <data/experiment/fid.h>

//class BlackchirpPlotCurve;

namespace BC::CSV {

static const QString del{";"};
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

static const QString configFile("config.csv");
static const QString headerFile("header.csv");
static const QString chirpFile("chirps.csv");
static const QString clockFile("clocks.csv");
static const QString auxFile("auxdata.csv");

static const QString fidparams{"fidparams.csv"};
static const QString fidDir("fid");

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

    static bool writeXY(QIODevice &device, const QVector<QPointF> d, const QString prefix = "");
    static bool writeMultiple(QIODevice &device, const std::vector<QVector<QPointF>> &l, const std::vector<QString> &n = {});

    template<typename T>
    static bool writeY(QIODevice &device, const QVector<T> d, QString title="")
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
    static bool writeYMultiple(QIODevice &device, std::initializer_list<QString> titles, std::initializer_list<QVector<T>> l)
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

    static bool writeHeader(QIODevice &device, const std::multimap<QString,std::tuple<QString,QString,QString,QString,QString>> header);

    static void writeLine(QTextStream &t, const QVariantList l);

    static QString formatInt64(qint64 n);
    static void writeFidList(QIODevice &device, const FidList l);

    static QVariantList readLine(QIODevice &device);
    static QVector<qint64> readFidLine(QIODevice &device);

    /*!
     * \brief Checks for existence of experiment directory
     * \param Experiment number
     * \return True if experiment folder exists
     */
    static bool exptDirExists(int num);
    static bool createExptDir(int num);
    static QDir exptDir(int num, QString path="");
    static QDir logDir();
    static QDir textExportDir();
    static QDir trackingDir();

};

#endif // BLACKCHIRPCSV_H
