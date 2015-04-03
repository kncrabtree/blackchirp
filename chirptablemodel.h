#ifndef CHIRPTABLEMODEL_H
#define CHIRPTABLEMODEL_H

#include <QAbstractTableModel>
#include <QList>

class ChirpTableModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    ChirpTableModel(QObject *parent = 0);
    ~ChirpTableModel();

    // QAbstractItemModel interface
    int rowCount(const QModelIndex &parent) const;
    int columnCount(const QModelIndex &parent) const;
    QVariant data(const QModelIndex &index, int role) const;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;
    bool setData(const QModelIndex &index, const QVariant &value, int role);
    bool removeRows(int row, int count, const QModelIndex &parent);
    Qt::ItemFlags flags(const QModelIndex &index) const;

    struct ChirpSegment {
        double startFreqMHz;
        double endFreqMHz;
        double durationNs;
    };

    void addSegment(double start, double end, double dur, int pos);
    void moveSegments(int first, int last, int delta);
    void removeSegments(QList<int> rows);

signals:
    void modelChanged();

private:
    QList<ChirpSegment> d_segmentList;


};

#endif // CHIRPTABLEMODEL_H
