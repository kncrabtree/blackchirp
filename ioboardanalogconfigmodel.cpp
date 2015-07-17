#include "ioboardanalogconfigmodel.h"

IOBoardAnalogConfigModel::IOBoardAnalogConfigModel(const IOBoardConfig c, QObject *parent) :
	QAbstractTableModel(parent)
{
    d_channelConfig = c.analogList();
    d_reserved = c.reservedAnalogChannels();
    d_numChannels = c.numAnalogChannels();

}

IOBoardAnalogConfigModel::~IOBoardAnalogConfigModel()
{

}

QMap<int, QPair<bool, QString> > IOBoardAnalogConfigModel::getConfig()
{
    return d_channelConfig;
}



int IOBoardAnalogConfigModel::rowCount(const QModelIndex &parent) const
{
	Q_UNUSED(parent)

    return d_channelConfig.size();
}

int IOBoardAnalogConfigModel::columnCount(const QModelIndex &parent) const
{
	Q_UNUSED(parent)

    return 2;
}

QVariant IOBoardAnalogConfigModel::data(const QModelIndex &index, int role) const
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

bool IOBoardAnalogConfigModel::setData(const QModelIndex &index, const QVariant &value, int role)
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

	bool out = true;
	switch(index.column()) {
    case 1:
        d_channelConfig[index.row()].second = value.toString();
		emit dataChanged(index,index);
		break;
    default:
        out = false;
        break;
	}
	return out;
}

QVariant IOBoardAnalogConfigModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	QVariant out;
	if(role != Qt::DisplayRole)
		return out;

	if(orientation == Qt::Horizontal)
	{
		switch (section) {
		case 0:
			out = QString("On");
			break;
        case 1:
			out = QString("Name");
			break;
		}
	}
	else
	{
        if(d_channelConfig.contains(section))
            out = QString("AIN%1").arg(section+d_reserved);
	}

	return out;
}

Qt::ItemFlags IOBoardAnalogConfigModel::flags(const QModelIndex &index) const
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
