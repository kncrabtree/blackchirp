#ifndef BLACKCHIRPCSV_H
#define BLACKCHIRPCSV_H

#define _STR(x) #x
#define STRINGIFY(x) _STR(x)

#include <QIODevice>
#include <QTextStream>
#include <QVector>
#include <QPointF>
#include <QDir>
#include <QLatin1StringView>

#include <data/experiment/fid.h>
#include <data/bcglobals.h>

namespace BC::CSV {

inline constexpr QLatin1StringView del{";"};
inline constexpr QLatin1StringView altDel{"|"}; // Alternative delimiter for QStringList serialization
inline constexpr QLatin1StringView nl{"\n"};
inline constexpr QLatin1StringView x{"x"};
inline constexpr QLatin1StringView y{"y"};
inline constexpr QLatin1StringView sep{"_"};

inline constexpr QLatin1StringView ok{"ObjKey"};
inline constexpr QLatin1StringView ak{"ArrayKey"};
inline constexpr QLatin1StringView ai{"ArrayIndex"};
inline constexpr QLatin1StringView vk{"ValueKey"};
inline constexpr QLatin1StringView vv{"Value"};
inline constexpr QLatin1StringView vu{"Units"};

inline constexpr QLatin1StringView versionFile{"version.csv"};
inline constexpr QLatin1StringView validationFile{"validation.csv"};
inline constexpr QLatin1StringView objectivesFile{"objectives.csv"};
inline constexpr QLatin1StringView hwFile{"hardware.csv"};
inline constexpr QLatin1StringView headerFile{"header.csv"};
inline constexpr QLatin1StringView chirpFile{"chirps.csv"};
inline constexpr QLatin1StringView markersFile{"markers.csv"};
inline constexpr QLatin1StringView clockFile{"clocks.csv"};
inline constexpr QLatin1StringView auxFile{"auxdata.csv"};

inline constexpr QLatin1StringView majver{"BCMajorVersion"};
inline constexpr QLatin1StringView minver{"BCMinorVersion"};
inline constexpr QLatin1StringView patchver{"BCPatchVersion"};
inline constexpr QLatin1StringView relver{"BCReleaseVersion"};
inline constexpr QLatin1StringView buildver{"BCBuildVersion"};

inline constexpr QLatin1StringView fidparams{"fidparams.csv"};
inline constexpr QLatin1StringView fidDir{"fid"};

inline constexpr QLatin1StringView lifparams{"lifparams.csv"};
inline constexpr QLatin1StringView lifDir{"lif"};

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
    BlackchirpCSV(const int num, const QString path);

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

    static void writeLine(QTextStream &t, const std::vector<QVariant> l);

    static QString formatInt64(qint64 n);
    static void writeFidList(QIODevice &device, const FidList l);

    static bool writeVersionFile(int num);

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

    QVariantList readLine(QIODevice &device);
    QVector<qint64> readFidLine(QIODevice &device);

    int majorVersion() const;
    int minorVersion() const;
    int patchVersion() const;
    QString releaseVersion() const;
    QString buildVersion() const;

private:
    std::map<QString,QVariant> d_configMap;
    QString d_delimiter;

};

#endif // BLACKCHIRPCSV_H
