#ifndef PEAKLISTEXPORTDIALOG_H
#define PEAKLISTEXPORTDIALOG_H

#include <QDialog>
#include <QAbstractTableModel>

#include <QVector>
#include <QPair>
#include <QSortFilterProxyModel>

#include <data/model/peaklistmodel.h>
#include <data/storage/settingsstorage.h>

namespace Ui {
class PeakListExportDialog;
}

namespace BC::Key {
inline constexpr QLatin1StringView plExport{"peakListExport"};
inline constexpr QLatin1StringView plAscii{"ascii"};
inline constexpr QLatin1StringView plDipoleEn{"dipoleEnabled"};
inline constexpr QLatin1StringView plDipole{"dipole"};
inline constexpr QLatin1StringView plDrOnlyEn{"drOnlyEnabled"};
inline constexpr QLatin1StringView plDrOnlyThresh{"drOnlyThresh"};
inline constexpr QLatin1StringView plDefaultShots{"defaultShots"};
inline constexpr QLatin1StringView plDrPower{"drPower"};
inline constexpr QLatin1StringView plDrPowerEn{"drPowerEnabled"};
inline constexpr QLatin1StringView plShotsTab{"shotsTable"};
inline constexpr QLatin1StringView plShots{"shots"};
inline constexpr QLatin1StringView plIntensity{"intensity"};
}

class ShotsModel;

class PeakListExportDialog : public QDialog, public SettingsStorage
{
    Q_OBJECT

public:
    explicit PeakListExportDialog(const QVector<QPointF> peakList, int number, QWidget *parent = 0);
    ~PeakListExportDialog();

public slots:
    void toggleButtons();
    void insertShot();
    void removeShots();
    void removePeaks();

private:
    Ui::PeakListExportDialog *ui;

    int d_number;
    QVector<QPointF> d_peakList;
    ShotsModel *p_sm;
    PeakListModel *p_pm;
    QSortFilterProxyModel *p_proxy;

    // QDialog interface
public slots:
    void accept();
};

class ShotsModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    explicit ShotsModel(QObject *parent = nullptr);
    void setList(const QVector<QPair<int,double>> l);
    QVector<QPair<int,double>> shotsList() const;
    void addEntry();
    void insertEntry(int pos);
    void removeEntries(QVector<int> rows);

private:
    QVector<QPair<int,double>> d_shotsList;

    // QAbstractItemModel interface
public:
    int rowCount(const QModelIndex &parent) const;
    int columnCount(const QModelIndex &parent) const;
    QVariant data(const QModelIndex &index, int role) const;
    bool setData(const QModelIndex &index, const QVariant &value, int role);
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;
    Qt::ItemFlags flags(const QModelIndex &index) const;
};



#endif // PEAKLISTEXPORTDIALOG_H
