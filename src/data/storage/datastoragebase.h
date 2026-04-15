#ifndef DATASTORAGEBASE_H
#define DATASTORAGEBASE_H

#include <memory>
#include <QString>
#include <QMutex>
#include <QVariant>
#include <QLatin1StringView>

class BlackchirpCSV;

namespace BC::Key::DS {
inline constexpr QLatin1StringView proc{"processing.csv"};
}

class DataStorageBase
{
public:
    DataStorageBase(int number = -1, QString path = "");
    virtual ~DataStorageBase();

    const int d_number;
    const QString d_path;

    virtual void advance() =0;
    virtual void save() =0;
    virtual void start() =0;
    virtual void finish() =0;

protected:
    std::unique_ptr<QMutex> pu_mutex;
    std::unique_ptr<BlackchirpCSV> pu_csv;

    void writeMetadata(QString file, const std::map<QString,QVariant> &dat,QString dir = "");
    void readMetadata(QString file, std::map<QString,QVariant> &out, QString dir = "");
};

#endif // DATASTORAGEBASE_H
