
#include "overlaytablemodel.h"
#include <data/experiment/overlaytypes.h>

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
    return BaseColumnCount + getAdditionalColumnCount();
}

QVariant OverlayTableModel::data(const QModelIndex &index, int role) const
{
    if (index.row() < 0 || index.row() >= d_overlays.size())
        return QVariant();

    auto overlay = d_overlays.at(index.row());
    if (!overlay)
        return QVariant();

    // Handle base columns
    if (index.column() < BaseColumnCount)
    {
        if (role == Qt::DisplayRole || role == Qt::EditRole)
        {
            switch (index.column())
            {
            case LabelColumn:
                return overlay->getLabel();
            case PlotIdColumn:
                return overlay->getPlotId();
            case YScaleColumn:
                return overlay->getYScale();
            case YOffsetColumn:
                return overlay->getYOffset();
            case XOffsetColumn:
                return overlay->getXOffset();
            case SourceFileColumn:
                return overlay->getSourceFile();
            }
        }
    }
    else
    {
        // Handle additional columns from derived classes
        int additionalColumn = index.column() - BaseColumnCount;
        return getAdditionalColumnData(index.row(), additionalColumn, role);
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

    // Handle base columns (only editable ones)
    if (index.column() < BaseColumnCount)
    {
        switch (index.column())
        {
        case LabelColumn:
            // Label is not editable
            return false;
        case PlotIdColumn:
            overlay->setPlotId(value.toString());
            break;
        case YScaleColumn:
            overlay->setYScale(value.toDouble());
            break;
        case YOffsetColumn:
            overlay->setYOffset(value.toDouble());
            break;
        case XOffsetColumn:
            overlay->setXOffset(value.toDouble());
            break;
        case SourceFileColumn:
            // Source file is not editable
            return false;
        default:
            return false;
        }

        emit dataChanged(index, index);
        return true;
    }
    else
    {
        // Handle additional columns from derived classes
        int additionalColumn = index.column() - BaseColumnCount;
        if (setAdditionalColumnData(index.row(), additionalColumn, value, role))
        {
            emit dataChanged(index, index);
            return true;
        }
    }

    return false;
}

QVariant OverlayTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (orientation != Qt::Horizontal)
        return QVariant();

    if (role == Qt::DisplayRole)
    {
        // Handle base column headers
        if (section < BaseColumnCount)
        {
            switch (section)
            {
            case LabelColumn:
                return QString("Label");
            case PlotIdColumn:
                return QString("Plot ID");
            case YScaleColumn:
                return QString("Y Scale");
            case YOffsetColumn:
                return QString("Y Offset");
            case XOffsetColumn:
                return QString("X Offset");
            case SourceFileColumn:
                return QString("Source File");
            }
        }
        else
        {
            // Handle additional column headers from derived classes
            int additionalColumn = section - BaseColumnCount;
            return getAdditionalHeaderData(additionalColumn, role);
        }
    }

    return QVariant();
}

Qt::ItemFlags OverlayTableModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return Qt::NoItemFlags;

    // Handle base columns
    if (index.column() < BaseColumnCount)
    {
        Qt::ItemFlags flags = Qt::ItemIsEnabled | Qt::ItemIsSelectable;

        // Label and Source File columns are not editable, others are
        if (index.column() != LabelColumn && index.column() != SourceFileColumn)
            flags |= Qt::ItemIsEditable;

        return flags;
    }
    else
    {
        // Handle additional columns from derived classes
        int additionalColumn = index.column() - BaseColumnCount;
        return getAdditionalColumnFlags(index.row(), additionalColumn);
    }
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

// BCExperimentOverlayModel implementation

BCExperimentOverlayModel::BCExperimentOverlayModel(QObject *parent)
    : OverlayTableModel(parent)
{
}

int BCExperimentOverlayModel::getAdditionalColumnCount() const
{
    return AdditionalColumnCount;
}

QVariant BCExperimentOverlayModel::getAdditionalColumnData(int row, int column, int role) const
{
    Q_UNUSED(row)
    Q_UNUSED(column)
    Q_UNUSED(role)
    // No additional columns for BCExperiment overlays anymore
    return QVariant();
}

QVariant BCExperimentOverlayModel::getAdditionalHeaderData(int column, int role) const
{
    Q_UNUSED(column)
    Q_UNUSED(role)
    // No additional columns for BCExperiment overlays anymore
    return QVariant();
}

bool BCExperimentOverlayModel::setAdditionalColumnData(int row, int column, const QVariant &value, int role)
{
    Q_UNUSED(row)
    Q_UNUSED(column)
    Q_UNUSED(value)
    Q_UNUSED(role)
    // No additional columns for BCExperiment overlays anymore
    return false;
}

Qt::ItemFlags BCExperimentOverlayModel::getAdditionalColumnFlags(int row, int column) const
{
    Q_UNUSED(row)
    Q_UNUSED(column)
    // No additional columns for BCExperiment overlays anymore
    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}
