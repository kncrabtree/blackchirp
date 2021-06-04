#ifndef CLOCKTABLEMODEL_H
#define CLOCKTABLEMODEL_H

#include <QAbstractTableModel>

#include <src/data/experiment/rfconfig.h>
#include <src/data/datastructs.h>

class ClockTableModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    explicit ClockTableModel(QObject *parent = nullptr);
    struct ClockHwInfo {
        int index;
        bool used;
        QString key;
        int output;
        QString name;
    };

    QList<ClockHwInfo> getHwInfo() const { return d_hwInfo; }
    void setConfig(const RfConfig c);
    RfConfig getRfConfig() const;

public slots:
    void setCommonLo(bool b);

private:
    RfConfig d_rfConfig;
    QList<BlackChirp::ClockType> d_clockTypes;
    QList<ClockHwInfo> d_hwInfo;
    QMap<BlackChirp::ClockType,int> d_clockAssignments;

    // QAbstractItemModel interface
public:
    virtual int rowCount(const QModelIndex &parent) const;
    virtual int columnCount(const QModelIndex &parent) const;
    virtual QVariant data(const QModelIndex &index, int role) const;
    virtual bool setData(const QModelIndex &index, const QVariant &value, int role);
    virtual QVariant headerData(int section, Qt::Orientation orientation, int role) const;
    virtual Qt::ItemFlags flags(const QModelIndex &index) const;
};

#include <QStyledItemDelegate>

class ClockTableDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    ClockTableDelegate(QObject *parent = nullptr);

    // QAbstractItemDelegate interface
    virtual QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;
    virtual void setEditorData(QWidget *editor, const QModelIndex &index) const;
    virtual void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
    virtual void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;
};

#endif // CLOCKTABLEMODEL_H
