#ifndef CHIRPTABLEMODEL_H
#define CHIRPTABLEMODEL_H

#include <QAbstractTableModel>

#include <QList>

#include "chirpconfig.h"

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

    void addSegment(double start, double end, double dur, int pos);
    void moveSegments(int first, int last, int delta);
    void removeSegments(QList<int> rows);

    QList<BlackChirp::ChirpSegment> segmentList() const;

signals:
    void modelChanged();

private:
    QList<BlackChirp::ChirpSegment> d_segmentList;


};

#include <QStyledItemDelegate>

class DoubleSpinBoxDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    DoubleSpinBoxDelegate(QObject *parent=0);

    // QAbstractItemDelegate interface
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;
    void setEditorData(QWidget *editor, const QModelIndex &index) const;
    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;
};


#endif // CHIRPTABLEMODEL_H
