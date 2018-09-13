#include "ioboardconfigmodel.h"

IOBoardConfigModel::IOBoardConfigModel(QString prefix, QObject *parent) :
    QAbstractTableModel(parent), d_prefix(prefix)
{
}

IOBoardConfigModel::~IOBoardConfigModel()
{

}

QMap<int, BlackChirp::IOBoardChannel> IOBoardConfigModel::getConfig()
{
    return d_channelConfig;
}

void IOBoardConfigModel::setFromConfig(const IOBoardConfig c)
{
    if(d_prefix == QString("AIN"))
    {
        d_channelConfig = c.analogList();
        d_reserved = c.reservedAnalogChannels();
        d_numChannels = c.numAnalogChannels();
    }
    else if(d_prefix == QString("DIN"))
    {
        d_channelConfig = c.digitalList();
        d_reserved = c.reservedDigitalChannels();
        d_numChannels = c.numDigitalChannels();
    }

    emit dataChanged(index(0,0),index(d_channelConfig.size(),3));
}



int IOBoardConfigModel::rowCount(const QModelIndex &parent) const
{
	Q_UNUSED(parent)

    return d_channelConfig.size();
}

int IOBoardConfigModel::columnCount(const QModelIndex &parent) const
{
	Q_UNUSED(parent)

    return 3;
}

QVariant IOBoardConfigModel::data(const QModelIndex &index, int role) const
{
	QVariant out;
    if(!d_channelConfig.contains(index.row()))
		return out;

	if(role == Qt::CheckStateRole)
	{
		if(index.column() == 0)
            out = (d_channelConfig.value(index.row()).enabled ? Qt::Checked : Qt::Unchecked);
        else if(index.column() == 1)
            out = (d_channelConfig.value(index.row()).plot ? Qt::Checked : Qt::Unchecked);
	}
	else if(role == Qt::DisplayRole)
	{
		switch (index.column()) {
        case 2:
            out = d_channelConfig.value(index.row()).name;
			break;
		default:
			break;
		}
	}
	else if(role == Qt::EditRole)
	{
		switch(index.column()) {
        case 2:
            out = d_channelConfig.value(index.row()).name;
			break;
		}
	}

	return out;
}

bool IOBoardConfigModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if(!d_channelConfig.contains(index.row()) || index.column() < 0 || index.column() > 2)
		return false;

	if(role == Qt::CheckStateRole)
	{
		if(index.column() == 0)
		{
            d_channelConfig[index.row()].enabled = value.toBool();
			emit dataChanged(index,index);
			return true;
		}
        else if(index.column() == 1)
        {
            d_channelConfig[index.row()].plot = value.toBool();
            emit dataChanged(index,index);
            return true;
        }
	}

	bool out = true;
	switch(index.column()) {
    case 2:
        d_channelConfig[index.row()].name = value.toString();
		emit dataChanged(index,index);
		break;
    default:
        out = false;
        break;
	}
	return out;
}

QVariant IOBoardConfigModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	QVariant out;

    if(role == Qt::DisplayRole)
    {
        if(orientation == Qt::Horizontal)
        {
            switch (section) {
            case 0:
                out = QString("On");
                break;
            case 2:
                out = QString("Name");
                break;
            case 1:
                out = QString("Plot");
                break;
            }
        }
        else
        {
            if(d_channelConfig.contains(section))
                out = d_prefix + QString::number(section+d_reserved);
        }
    }
    else if(role == Qt::ToolTipRole)
    {
        if(orientation == Qt::Horizontal)
        {
            switch (section) {
            case 0:
                out = QString("Check if you wish this channel to be read at each time point.");
                break;
            case 2:
                out = QString("A name for this channel");
                break;
            case 1:
                out = QString("Check if you want the data from this channel to appear on the tracking plots.");
                break;
            }
        }
    }

	return out;
}

Qt::ItemFlags IOBoardConfigModel::flags(const QModelIndex &index) const
{
	Qt::ItemFlags out = Qt::NoItemFlags;
	switch(index.column()) {
	case 0:
    case 1:
        out = Qt::ItemIsUserCheckable | Qt::ItemIsEnabled;
		break;
    case 2:
		out = Qt::ItemIsEditable | Qt::ItemIsSelectable | Qt::ItemIsEnabled;
		break;
	default:
		break;
	}

	return out;
}
