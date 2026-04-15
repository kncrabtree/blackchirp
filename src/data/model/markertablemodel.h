#ifndef MARKERTABLEMODEL_H
#define MARKERTABLEMODEL_H

#include <QAbstractTableModel>
#include <QVector>

#include <data/experiment/chirpconfig.h>
#include <data/storage/settingsstorage.h>

namespace BC::Key::MarkerTableModel {
inline constexpr QLatin1StringView key{"MarkerTableModel"};
inline constexpr QLatin1StringView markers{"markers"};
inline constexpr QLatin1StringView chName{"name"};
inline constexpr QLatin1StringView chRole{"role"};
inline constexpr QLatin1StringView chStart{"startUs"};
inline constexpr QLatin1StringView chEnd{"endUs"};
inline constexpr QLatin1StringView chEnabled{"enabled"};
}

class MarkerTableModel : public QAbstractTableModel, public SettingsStorage
{
    Q_OBJECT
public:
    explicit MarkerTableModel(QObject *parent = nullptr);
    ~MarkerTableModel();

    // QAbstractItemModel interface
    int rowCount(const QModelIndex &parent) const override;
    int columnCount(const QModelIndex &parent) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
    bool setData(const QModelIndex &index, const QVariant &value, int role) override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;

    void setFromChirpConfig(const ChirpConfig &cc);
    QVector<MarkerChannel> markerChannels() const { return d_channels; }
    int markerCount() const { return d_markerCount; }

signals:
    void modelChanged();

private:
    int d_markerCount;
    QVector<MarkerChannel> d_channels;

    void saveToSettings();
};

#include <QStyledItemDelegate>

class MarkerDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    explicit MarkerDelegate(QObject *parent = nullptr);

    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                          const QModelIndex &index) const override;
    void setEditorData(QWidget *editor, const QModelIndex &index) const override;
    void setModelData(QWidget *editor, QAbstractItemModel *model,
                      const QModelIndex &index) const override;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option,
                              const QModelIndex &index) const override;
};

#endif // MARKERTABLEMODEL_H
