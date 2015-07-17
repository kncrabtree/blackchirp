#include "ioboarddigitalconfigmodel.h"

IOBoardDigitalConfigModel::IOBoardDigitalConfigModel(const IOBoardConfig c, QObject *parent) :
	QAbstractTableModel(parent)
{
    d_channelConfig = c.digitalList();
    d_reserved = c.reservedDigitalChannels();
    d_numChannels = c.numDigitalChannels();
}

IOBoardDigitalConfigModel::~IOBoardDigitalConfigModel()
{

}

QMap<int, QPair<bool, QString> > IOBoardDigitalConfigModel::getConfig()
{
    return d_channelConfig;
}



int IOBoardDigitalConfigModel::rowCount(const QModelIndex &parent) const
{
	Q_UNUSED(parent)

    return d_numChannels;
}

int IOBoardDigitalConfigModel::columnCount(const QModelIndex &parent) const
{
	Q_UNUSED(parent)

	return 2;
}

QVariant IOBoardDigitalConfigModel::data(const QModelIndex &index, int role) const
{
	QVariant out;
    if(!d_channelConfig.contains(index.row()))
		return out;

	if(role == Qt::CheckStateRole)
	{
		if(index.column() == 0)
            out = (d_channelConfig.value(index.row()).first ? Qt::Checked : Qt::Unchecked);
	}
	else if(role == Qt::DisplayRole)
	{
		switch (index.column()) {
		case 0:
			break;
		case 1:
            out = d_channelConfig.value(index.row()).second;
			break;
		default:
			break;
		}
	}
	else if(role == Qt::EditRole)
	{
		switch(index.column()) {
		case 0:
			break;
		case 1:
            out = d_channelConfig.value(index.row()).second;
			break;
		}
	}

	return out;
}

bool IOBoardDigitalConfigModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if(!d_channelConfig.contains(index.row()) || index.column() < 0 || index.column() > 1)
		return false;

	if(role == Qt::CheckStateRole)
	{
		if(index.column() == 0)
		{
            d_channelConfig[index.row()].first = value.toBool();
			emit dataChanged(index,index);
			return true;
		}
	}

	if(role != Qt::EditRole)
		return false;


	if(index.column() == 1)
	{
        d_channelConfig[index.row()].second = value.toString();
		emit dataChanged(index,index);
		return true;
	}

	return false;
}

QVariant IOBoardDigitalConfigModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if(role == Qt::DisplayRole)
	{
		if(orientation == Qt::Horizontal)
		{
			if(section == 0)
				return QString("On");
			if(section == 1)
				return QString("Name");
		}
		else if(orientation == Qt::Vertical)
		{
            if(section >= 0 && section < d_channelConfig.size())
                return QString("DIN%1").arg(section+d_reserved);
		}
	}

	return QVariant();
}

Qt::ItemFlags IOBoardDigitalConfigModel::flags(const QModelIndex &index) const
{
	Qt::ItemFlags out = Qt::NoItemFlags;
	switch(index.column()) {
	case 0:
		out = Qt::ItemIsEditable | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable | Qt::ItemIsEnabled;
		break;
	case 1:
		out = Qt::ItemIsEditable | Qt::ItemIsSelectable | Qt::ItemIsEnabled;
		break;
	default:
		break;
	}

	return out;
}
