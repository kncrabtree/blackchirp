#ifndef CLOCKTABLEMODEL_H
#define CLOCKTABLEMODEL_H

#include <QAbstractTableModel>
#include <data/storage/settingsstorage.h>

#include <data/experiment/rfconfig.h>

namespace BC::Key::ClockTable {
static const QString ctKey("ClockConfigTable");
static const QString ctClocks("clocks");
static const QString ctClockType("type");
static const QString ctHwKey("hwKey");
static const QString ctOutput("output");
static const QString ctOp("operation");
static const QString ctFactor("factor");
static const QString ctFreq("freqMHz");
}

class ClockTableModel : public QAbstractTableModel, public SettingsStorage
{
    Q_OBJECT
public:
    explicit ClockTableModel(QObject *parent = nullptr);
    ~ClockTableModel();

    struct ClockHwInfo {
        int index;
        bool used;
        QString name;
        QString hwKey;
        int output;
    };

    QList<ClockHwInfo> getHwInfo() const { return d_hwInfo; }
    void setFromConfig(const RfConfig &c);
    void toRfConfig(RfConfig &c) const;
    QString getHwKey(RfConfig::ClockType type) const;

public slots:
    void setCommonLo(bool b);

private:
    bool d_commonUpDownLO;
    QList<ClockHwInfo> d_hwInfo;
    QVector<RfConfig::ClockType> d_clockTypes;
    QHash<RfConfig::ClockType,RfConfig::ClockFreq> d_clockConfigs;
    QHash<RfConfig::ClockType,int> d_clockAssignments;

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
