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

    // Column indices (public for access by other classes)
    enum Columns {
        ConfigureColumn = 0,
        EnabledColumn = 1,
        LabelColumn = 2,
        PlotIdColumn = 3,
        OverlayTypeColumn = 4,
        SourceFileColumn = 5,
        ColumnCount = 6
    };

private:
    QVector<std::shared_ptr<OverlayBase>> d_overlays;
    
    // Helper method to get friendly overlay type name
    QString getOverlayTypeName(OverlayBase::OverlayType type) const;
};

#endif // OVERLAYTABLEMODEL_H
