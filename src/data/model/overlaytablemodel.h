#ifndef OVERLAYTABLEMODEL_H
#define OVERLAYTABLEMODEL_H

#include <QAbstractTableModel>
#include <QVector>
#include <memory>
#include <data/experiment/overlaybase.h>

class OverlayTableModel : public QAbstractTableModel
{
    Q_OBJECT
public:
    explicit OverlayTableModel(QObject *parent = nullptr);
    virtual ~OverlayTableModel();

    // QAbstractItemModel interface
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    bool setData(const QModelIndex &index, const QVariant &value, int role) override;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;

    // Overlay management
    void addOverlay(std::shared_ptr<OverlayBase> overlay);
    void removeOverlay(int row);
    void removeOverlays(const QVector<int>& rows);
    void clearOverlays();
    std::shared_ptr<OverlayBase> getOverlay(int row) const;
    QVector<std::shared_ptr<OverlayBase>> getAllOverlays() const;

    // Base column indices (public for access by other classes)
    enum BaseColumns {
        ConfigureColumn = 0,
        LabelColumn = 1,
        PlotIdColumn = 2,
        SourceFileColumn = 3,
        BaseColumnCount = 4
    };

protected:
    // Virtual functions for derived classes to extend columns
    virtual int getAdditionalColumnCount() const { return 0; }
    virtual QVariant getAdditionalColumnData(int row, int column, int role) const { Q_UNUSED(row) Q_UNUSED(column) Q_UNUSED(role) return QVariant(); }
    virtual QVariant getAdditionalHeaderData(int column, int role) const { Q_UNUSED(column) Q_UNUSED(role) return QVariant(); }
    virtual bool setAdditionalColumnData(int row, int column, const QVariant &value, int role) { Q_UNUSED(row) Q_UNUSED(column) Q_UNUSED(value) Q_UNUSED(role) return false; }
    virtual Qt::ItemFlags getAdditionalColumnFlags(int row, int column) const { Q_UNUSED(row) Q_UNUSED(column) return Qt::ItemIsEnabled | Qt::ItemIsSelectable; }

private:
    QVector<std::shared_ptr<OverlayBase>> d_overlays;
};

// Derived class for BCExperiment overlays
class BCExperimentOverlayModel : public OverlayTableModel
{
    Q_OBJECT
public:
    explicit BCExperimentOverlayModel(QObject *parent = nullptr);

protected:
    // Additional columns for BCExperiment
    int getAdditionalColumnCount() const override;
    QVariant getAdditionalColumnData(int row, int column, int role) const override;
    QVariant getAdditionalHeaderData(int column, int role) const override;
    bool setAdditionalColumnData(int row, int column, const QVariant &value, int role) override;
    Qt::ItemFlags getAdditionalColumnFlags(int row, int column) const override;

private:
    enum AdditionalColumns {
        AdditionalColumnCount = 0
    };
};

#endif // OVERLAYTABLEMODEL_H
