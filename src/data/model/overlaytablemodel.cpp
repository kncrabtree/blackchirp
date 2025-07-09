
#include "overlaytablemodel.h"
#include <data/experiment/overlaytypes.h>
#include <QColor>

OverlayTableModel::OverlayTableModel(QObject *parent)
    : QAbstractTableModel(parent)
{
}

OverlayTableModel::~OverlayTableModel()
{
    // Note: We don't delete overlays here as they may be owned elsewhere
}

int OverlayTableModel::rowCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return d_overlays.size();
}

int OverlayTableModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return ColumnCount;
}

QVariant OverlayTableModel::data(const QModelIndex &index, int role) const
{
    if (index.row() < 0 || index.row() >= d_overlays.size())
        return QVariant();

    auto overlay = d_overlays.at(index.row());
    if (!overlay)
        return QVariant();

    if (role == Qt::DisplayRole || role == Qt::EditRole)
    {
        switch (index.column())
        {
        case ConfigureColumn:
            return QString::fromUtf8("⚙");
        case EnabledColumn:
            return overlay->getEnabled();
        case LabelColumn:
            return overlay->getLabel();
        case PlotIdColumn:
            return overlay->getPlotId();
        case OverlayTypeColumn:
            return getOverlayTypeName(overlay->type());
        case SourceFileColumn:
            return overlay->getSourceFile();
        }
    }
    else if (role == Qt::TextAlignmentRole)
    {
        // Center-align all columns
        return Qt::AlignCenter;
    }

    return QVariant();
}

bool OverlayTableModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (role != Qt::EditRole)
        return false;

    if (index.row() < 0 || index.row() >= d_overlays.size())
        return false;

    auto overlay = d_overlays.at(index.row());
    if (!overlay)
        return false;

    // Handle editable columns
    switch (index.column())
    {
    case ConfigureColumn:
        // Configure column is not editable - handled by delegate
        return false;
    case EnabledColumn:
        // Enabled column is editable via checkbox delegate
        overlay->setEnabled(value.toBool());
        emit dataChanged(index, index);
        return true;
    case LabelColumn:
        // Label is not editable - handled in configuration dialog
        return false;
    case PlotIdColumn:
        // PlotId is not editable - handled in configuration dialog
        return false;
    case OverlayTypeColumn:
        // Overlay type is not editable - determined at creation time
        return false;
    case SourceFileColumn:
        // Source file is not editable
        return false;
    default:
        return false;
    }

    return false;
}

QVariant OverlayTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (orientation != Qt::Horizontal)
        return QVariant();

    if (role == Qt::DisplayRole)
    {
        switch (section)
        {
        case ConfigureColumn:
            return QString::fromUtf8("⚙");
        case EnabledColumn:
            return QString::fromUtf8("👁"); // Eye symbol for visibility
        case LabelColumn:
            return QString("Label");
        case PlotIdColumn:
            return QString("Plot ID");
        case OverlayTypeColumn:
            return QString("Type");
        case SourceFileColumn:
            return QString("Source File");
        }
    }

    return QVariant();
}

Qt::ItemFlags OverlayTableModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return Qt::NoItemFlags;

    Qt::ItemFlags flags = Qt::ItemIsEnabled | Qt::ItemIsSelectable;

    // EnabledColumn is editable via checkbox delegate
    if (index.column() == EnabledColumn) {
        flags |= Qt::ItemIsEditable;
    }
    
    return flags;
}

void OverlayTableModel::addOverlay(std::shared_ptr<OverlayBase> overlay)
{
    if (!overlay)
        return;

    beginInsertRows(QModelIndex(), d_overlays.size(), d_overlays.size());
    d_overlays.append(overlay);
    endInsertRows();
}

void OverlayTableModel::removeOverlay(int row)
{
    if (row < 0 || row >= d_overlays.size())
        return;

    beginRemoveRows(QModelIndex(), row, row);
    d_overlays.removeAt(row);
    endRemoveRows();
}

void OverlayTableModel::removeOverlays(const QVector<int>& rows)
{
    if (rows.isEmpty())
        return;

    // Sort rows in descending order to remove from end to beginning
    QVector<int> sortedRows = rows;
    std::sort(sortedRows.begin(), sortedRows.end(), std::greater<int>());

    for (int row : sortedRows)
    {
        removeOverlay(row);
    }
}

void OverlayTableModel::clearOverlays()
{
    if (d_overlays.isEmpty())
        return;

    beginRemoveRows(QModelIndex(), 0, d_overlays.size() - 1);
    d_overlays.clear();
    endRemoveRows();
}

std::shared_ptr<OverlayBase> OverlayTableModel::getOverlay(int row) const
{
    if (row < 0 || row >= d_overlays.size())
        return nullptr;

    return d_overlays.at(row);
}

QVector<std::shared_ptr<OverlayBase>> OverlayTableModel::getAllOverlays() const
{
    return d_overlays;
}

QString OverlayTableModel::getOverlayTypeName(OverlayBase::OverlayType type) const
{
    switch (type)
    {
    case OverlayBase::BCExperiment:
        return QString("BC Experiment");
    case OverlayBase::SPCAT:
        return QString("SPCAT");
    case OverlayBase::GenericXY:
        return QString("Generic XY");
    default:
        return QString("Unknown");
    }
}
