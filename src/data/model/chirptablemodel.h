#ifndef CHIRPTABLEMODEL_H
#define CHIRPTABLEMODEL_H

#include <QAbstractTableModel>

#include <QList>

#include <src/data/experiment/rfconfig.h>

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

    void addSegment(double start, double end, double dur, int pos, bool empty=false);
    void moveSegments(int first, int last, int delta);
    void removeSegments(QList<int> rows);
    double calculateAwgFrequency(double f) const;
    double calculateChirpFrequency(double f) const;

    QList<QList<BlackChirp::ChirpSegment>> chirpList() const;
    void setRfConfig(const RfConfig c){ d_currentRfConfig = c; }
    RfConfig getRfConfig();

public slots:
    void setCurrentChirp(int i);
    void setApplyToAll(bool a2a) { d_applyToAll = a2a; }
    void setNumChirps(int num);

signals:
    void modelChanged();

private:
    QList<QList<BlackChirp::ChirpSegment>> d_chirpList;
    QList<BlackChirp::ChirpSegment> d_currentData;
    RfConfig d_currentRfConfig;
    int d_currentChirp;
    bool d_applyToAll;


};

#include <QStyledItemDelegate>

class ChirpDoubleSpinBoxDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    ChirpDoubleSpinBoxDelegate(QObject *parent=0);

    // QAbstractItemDelegate interface
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;
    void setEditorData(QWidget *editor, const QModelIndex &index) const;
    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;
};


#endif // CHIRPTABLEMODEL_H
