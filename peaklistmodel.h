#ifndef PEAKLISTMODEL_H
#define PEAKLISTMODEL_H

#include <QAbstractTableModel>

#include <QVector>
#include <QPointF>

class PeakListModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    PeakListModel(QObject *parent = nullptr);

    // QAbstractItemModel interface
    int rowCount(const QModelIndex &parent) const;
    int columnCount(const QModelIndex &parent) const;
    QVariant data(const QModelIndex &index, int role) const;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;
    Qt::ItemFlags flags(const QModelIndex &index) const;

    void setPeakList(const QList<QPointF> l);
    void removePeaks(QList<int> rows);
    void scalingChanged(double scf);
    void clearPeakList();
    QList<QPointF> peakList();

private:
    QList<QPointF> d_peakList;
};

#endif // PEAKLISTMODEL_H
