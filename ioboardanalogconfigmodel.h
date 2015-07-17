#ifndef IOBOARDANALOGCONFIGMODEL_H
#define IOBOARDANALOGCONFIGMODEL_H

#include <QAbstractTableModel>
#include <QList>

#include "datastructs.h"
#include "ioboardconfig.h"

class IOBoardAnalogConfigModel : public QAbstractTableModel
{
	Q_OBJECT
public:
    IOBoardAnalogConfigModel(const IOBoardConfig c, QObject *parent = nullptr);
	~IOBoardAnalogConfigModel();

	void saveToSettings();
    QMap<int,QPair<bool,QString>> getConfig();

private:
    QMap<int,QPair<bool,QString>> d_channelConfig;
    int d_reserved;
    int d_numChannels;

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
