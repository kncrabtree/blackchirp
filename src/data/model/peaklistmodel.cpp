#include <data/model/peaklistmodel.h>

PeakListModel::PeakListModel(QObject *parent) : QAbstractTableModel(parent)
{

}



int PeakListModel::rowCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return d_peakList.size();
}

int PeakListModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return 2;
}

QVariant PeakListModel::data(const QModelIndex &index, int role) const
{
    if(index.row() < 0 || index.row() >= d_peakList.size())
        return QVariant();

    if(role == Qt::DisplayRole)
    {
        switch(index.column())
        {
        case 0:
            return QString::number(d_peakList.at(index.row()).x(),'f',3);
            break;
        case 1:
            return QString::number(d_peakList.at(index.row()).y(),'e',3);
            break;
        }
    }
    else if(role == Qt::EditRole)
    {
        switch (index.column())
        {
        case 0:
            return d_peakList.at(index.row()).x();
            break;
        case 1:
            return d_peakList.at(index.row()).y();
            break;
        }
    }

    return QVariant();
}

QVariant PeakListModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if(role == Qt::DisplayRole)
    {
        if(orientation == Qt::Horizontal)
        {
            if(section == 0)
                return QString("Freq (MHz)");
            else if(section == 1)
                return QString("Int (%1 pks)").arg(d_peakList.size());
        }
    }

    return QVariant();
}

Qt::ItemFlags PeakListModel::flags(const QModelIndex &index) const
{
    Q_UNUSED(index)
    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

void PeakListModel::setPeakList(const QVector<QPointF> l)
{
    if(!d_peakList.isEmpty())
    {
        beginRemoveRows(QModelIndex(),0,d_peakList.size()-1);
        d_peakList.clear();
        endRemoveRows();
    }

    if(!l.isEmpty())
    {
        beginInsertRows(QModelIndex(),0,l.size()-1);
        d_peakList = l;
        endInsertRows();
    }
}

void PeakListModel::removePeaks(QVector<int> rows)
{
    if(rows.isEmpty())
        return;

    std::sort(rows.begin(),rows.end());
    for(int i=rows.size()-1; i>=0; i--)
    {
        if(rows.at(i) >= 0 && rows.at(i) < d_peakList.size())
        {
            beginRemoveRows(QModelIndex(),rows.at(i),rows.at(i));
            d_peakList.removeAt(rows.at(i));
            endRemoveRows();
        }
    }
}

void PeakListModel::scalingChanged(double scf)
{
    for(int i=0; i<d_peakList.size(); i++)
        d_peakList[i].setY(d_peakList.at(i).y()*scf);

    emit dataChanged(index(0,1),index(d_peakList.size()-1,1));
}

void PeakListModel::clearPeakList()
{
    setPeakList({});
}

QVector<QPointF> PeakListModel::peakList()
{
    return d_peakList;
}
