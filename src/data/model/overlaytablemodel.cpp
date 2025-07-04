
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

    OverlayBase* overlay = d_overlays.at(index.row());
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

    OverlayBase* overlay = d_overlays.at(index.row());
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

        // Label column is not editable, others are
        if (index.column() != LabelColumn)
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

void OverlayTableModel::addOverlay(OverlayBase* overlay)
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

OverlayBase* OverlayTableModel::getOverlay(int row) const
{
    if (row < 0 || row >= d_overlays.size())
        return nullptr;

    return d_overlays.at(row);
}

QVector<OverlayBase*> OverlayTableModel::getAllOverlays() const
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
    OverlayBase* overlay = getOverlay(row);
    if (!overlay)
        return QVariant();

    // Try to cast to BCExperimentOverlay to access additional data
    auto bcOverlay = dynamic_cast<BCExpOverlay*>(overlay);
    if (!bcOverlay)
        return QVariant();

    if (role == Qt::DisplayRole || role == Qt::EditRole)
    {
        switch (column)
        {
        case FrameColumn:
            // TODO: Add getter for frame in BCExperimentOverlay
            return QString("N/A"); // Placeholder until getter is implemented
        case SourceFileColumn:
            return bcOverlay->getSourceFile();
        }
    }

    return QVariant();
}

QVariant BCExperimentOverlayModel::getAdditionalHeaderData(int column, int role) const
{
    if (role == Qt::DisplayRole)
    {
        switch (column)
        {
        case FrameColumn:
            return QString("Frame");
        case SourceFileColumn:
            return QString("Source File");
        }
    }

    return QVariant();
}

bool BCExperimentOverlayModel::setAdditionalColumnData(int row, int column, const QVariant &value, int role)
{
    if (role != Qt::EditRole)
        return false;

    OverlayBase* overlay = getOverlay(row);
    if (!overlay)
        return false;

    auto bcOverlay = dynamic_cast<BCExpOverlay*>(overlay);
    if (!bcOverlay)
        return false;

    switch (column)
    {
    case FrameColumn:
        // TODO: Add setter for frame in BCExperimentOverlay
        return false; // Not editable for now
    case SourceFileColumn:
        bcOverlay->setSourceFile(value.toString());
        return true;
    }

    return false;
}

Qt::ItemFlags BCExperimentOverlayModel::getAdditionalColumnFlags(int row, int column) const
{
    Q_UNUSED(row)

    Qt::ItemFlags flags = Qt::ItemIsEnabled | Qt::ItemIsSelectable;

    switch (column)
    {
    case FrameColumn:
        // TODO: Make editable when getter/setter are implemented
        break;
    case SourceFileColumn:
        flags |= Qt::ItemIsEditable;
        break;
    }

    return flags;
}
