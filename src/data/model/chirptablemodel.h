#ifndef CHIRPTABLEMODEL_H
#define CHIRPTABLEMODEL_H

#include <QAbstractTableModel>
#include <data/storage/settingsstorage.h>

#include <QList>
#include <memory>

#include <data/experiment/rfconfig.h>

namespace BC::Key::ChirpTableModel {
inline constexpr QLatin1StringView key{"ChirpTableModel"};
inline constexpr QLatin1StringView ctChirps{"chirps"};
inline constexpr QLatin1StringView chirpIndex{"chirpIndex"};
inline constexpr QLatin1StringView segIndex{"segmentIndex"};
inline constexpr QLatin1StringView start{"startFreqMHz"};
inline constexpr QLatin1StringView end{"endFreqMHz"};
inline constexpr QLatin1StringView duration{"durationUs"};
inline constexpr QLatin1StringView empty{"empty"};
}

class ChirpTableModel : public QAbstractTableModel, public SettingsStorage
{
    Q_OBJECT
public:
    ChirpTableModel(QObject *parent = 0);
    ~ChirpTableModel();

    bool d_allIdentical;

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
    double awgMin() const { return d_awgMin; }
    double awgMax() const { return d_awgMax; }

    QVector<QVector<ChirpConfig::ChirpSegment>> chirpList() const;
    void initialize(RfConfig *p);
    void setFromRfConfig(RfConfig *p);

public slots:
    void setCurrentChirp(int i);
    void setNumChirps(int num);

signals:
    void modelChanged();

private:
    QVector<QVector<ChirpConfig::ChirpSegment>> d_chirpList;
    QVector<ChirpConfig::ChirpSegment> d_currentData;
    RfConfig *p_rfConfig;
    int d_currentChirp;
    double d_awgMin,d_awgMax;



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
