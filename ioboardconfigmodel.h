#ifndef IOBOARDCONFIGMODEL_H
#define IOBOARDCONFIGMODEL_H

#include <QAbstractTableModel>
#include <QList>

#include "datastructs.h"
#include "ioboardconfig.h"

class IOBoardConfigModel : public QAbstractTableModel
{
	Q_OBJECT
public:
    IOBoardConfigModel(const  QMap<int,BlackChirp::IOBoardChannel> l, int numChannels, int numReserved, QString prefix = QString("AIN"), QObject *parent = nullptr);
    ~IOBoardConfigModel();

	void saveToSettings();
    QMap<int,BlackChirp::IOBoardChannel> getConfig();

private:
    QMap<int,BlackChirp::IOBoardChannel> d_channelConfig;
    int d_reserved;
    int d_numChannels;
    QString d_prefix;

	// QAbstractItemModel interface
public:
	int rowCount(const QModelIndex &parent) const;
	int columnCount(const QModelIndex &parent) const;
	QVariant data(const QModelIndex &index, int role) const;
	bool setData(const QModelIndex &index, const QVariant &value, int role);
	QVariant headerData(int section, Qt::Orientation orientation, int role) const;
	Qt::ItemFlags flags(const QModelIndex &index) const;
};

#endif // IOBOARDANALOGCONFIGMODEL_H
