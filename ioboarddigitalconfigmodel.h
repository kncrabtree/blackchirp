#ifndef IOBOARDDIGITALCONFIGMODEL_H
#define IOBOARDDIGITALCONFIGMODEL_H

#include <QAbstractTableModel>

#include "ioboardconfig.h"

class IOBoardDigitalConfigModel : public QAbstractTableModel
{
	Q_OBJECT
public:
    IOBoardDigitalConfigModel(const IOBoardConfig c, QObject *parent = nullptr);
	~IOBoardDigitalConfigModel();

    QMap<int,QPair<bool,QString>> getConfig();

	// QAbstractItemModel interface
	int rowCount(const QModelIndex &parent) const;
	int columnCount(const QModelIndex &parent) const;
	QVariant data(const QModelIndex &index, int role) const;
	bool setData(const QModelIndex &index, const QVariant &value, int role);
	QVariant headerData(int section, Qt::Orientation orientation, int role) const;
	Qt::ItemFlags flags(const QModelIndex &index) const;



private:
    QMap<int,QPair<bool,QString>> d_channelConfig;
    int d_reserved;
    int d_numChannels;

};

#endif // IOBOARDDIGITALCONFIGMODEL_H
