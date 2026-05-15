
#include "overlaytablemodel.h"
#include <data/experiment/overlaytypes.h>
#include <QColor>
#include <QMimeData>
#include <QDataStream>
#include <QIODevice>
#include <algorithm>

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
        case CommentColumn:
            return overlay->getComment();
        }
    }
    else if (role == Qt::TextAlignmentRole)
    {
        // Center-align all columns
        return Qt::AlignCenter;
    }
    else if (role == Qt::ToolTipRole)
    {
        QString comment = overlay->getComment();
        if (!comment.isEmpty())
            return comment;
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
    case CommentColumn:
        // Comment is editable with semicolon validation
        {
            QString comment = value.toString();
            if (comment.contains(';')) {
                // Reject comments containing semicolons to maintain CSV format integrity
                return false;
            }
            overlay->setComment(comment);
            emit dataChanged(index, index);
            return true;
        }
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
        case CommentColumn:
            return QString("Comment");
        }
    }

    return QVariant();
}

Qt::ItemFlags OverlayTableModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return Qt::ItemIsDropEnabled; // Allow drops on empty areas

    Qt::ItemFlags flags = Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsDragEnabled;

    // EnabledColumn is editable via checkbox delegate
    if (index.column() == EnabledColumn) {
        flags |= Qt::ItemIsEditable;
    }
    
    // CommentColumn is directly editable
    if (index.column() == CommentColumn) {
        flags |= Qt::ItemIsEditable;
    }
    
    // Enable dropping between rows
    flags |= Qt::ItemIsDropEnabled;
    
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
    case OverlayBase::Catalog:
        return QString("Catalog");
    case OverlayBase::GenericXY:
        return QString("Generic XY");
    default:
        return QString("Unknown");
    }
}

// Drag and drop implementation
Qt::DropActions OverlayTableModel::supportedDropActions() const
{
    return Qt::MoveAction;
}

QStringList OverlayTableModel::mimeTypes() const
{
    return QStringList() << "application/x-overlay-row";
}

QMimeData *OverlayTableModel::mimeData(const QModelIndexList &indexes) const
{
    if (indexes.isEmpty())
        return nullptr;
        
    QMimeData *mimeData = new QMimeData;
    QByteArray data;
    QDataStream stream(&data, QIODevice::WriteOnly);
    
    // Store the row number of the first selected item
    // (we only support single row drag for simplicity)
    int row = indexes.first().row();
    stream << row;
    
    mimeData->setData("application/x-overlay-row", data);
    return mimeData;
}

bool OverlayTableModel::dropMimeData(const QMimeData *data, Qt::DropAction action, int row, int column, const QModelIndex &parent)
{
    Q_UNUSED(column)
    
    if (action != Qt::MoveAction)
        return false;
        
    if (!data->hasFormat("application/x-overlay-row"))
        return false;
        
    QByteArray itemData = data->data("application/x-overlay-row");
    QDataStream stream(&itemData, QIODevice::ReadOnly);
    int fromRow;
    stream >> fromRow;
    
    // Determine drop position  
    // The 'row' parameter is the insertion point where the item should be inserted
    int insertionPoint;
    if (parent.isValid()) {
        insertionPoint = parent.row();
    } else if (row == -1) {
        insertionPoint = d_overlays.size(); // Drop at end
    } else {
        insertionPoint = qBound(0, row, d_overlays.size()); // Ensure row is within bounds
    }
    
    // Use insertionPoint directly - this is where the user wants the item to be inserted
    int toRow = insertionPoint;
    
    // Validate fromRow
    if (fromRow < 0 || fromRow >= d_overlays.size()) {
        return false;
    }
    
    // Don't move if dropping on the same position
    if (fromRow == toRow) {
        return false;
    }
        
    moveOverlay(fromRow, toRow);
    return true;
}

void OverlayTableModel::moveOverlay(int fromRow, int toRow)
{
    if (fromRow < 0 || fromRow >= d_overlays.size() || 
        toRow < 0 || toRow > d_overlays.size() || 
        fromRow == toRow)
        return;
        
    // toRow is the insertion point - use it directly for beginMoveRows
    // destinationChild is where the item will appear BEFORE
    int destinationChild = toRow;
    
    // Check bounds for beginMoveRows
    if (destinationChild < 0 || destinationChild > d_overlays.size()) {
        return;
    }
    
    if (!beginMoveRows(QModelIndex(), fromRow, fromRow, QModelIndex(), destinationChild)) {
        return; // Move operation not allowed
    }
    
    // Move the overlay in the underlying vector
    auto overlay = d_overlays.takeAt(fromRow);
    
    // Calculate insert position after takeAt() has shifted the vector
    int insertPos;
    if (fromRow < toRow) {
        // Moving down: after takeAt(fromRow), insertion point shifts back by 1
        insertPos = toRow - 1;
    } else {
        // Moving up: insertion point stays the same
        insertPos = toRow;
    }
    
    d_overlays.insert(insertPos, overlay);
    
    endMoveRows();
}

void OverlayTableModel::sort(int column, Qt::SortOrder order)
{
    if (d_overlays.isEmpty())
        return;
        
    emit layoutAboutToBeChanged();
    
    // Sort the overlays based on the column
    std::sort(d_overlays.begin(), d_overlays.end(), [this, column, order](const std::shared_ptr<OverlayBase>& a, const std::shared_ptr<OverlayBase>& b) {
        QVariant aValue, bValue;
        
        switch (column) {
        case ConfigureColumn:
            // Sort by label for configure column (since configure button doesn't have sortable data)
            aValue = a->getLabel();
            bValue = b->getLabel();
            break;
        case EnabledColumn:
            aValue = a->getEnabled();
            bValue = b->getEnabled();
            break;
        case LabelColumn:
            aValue = a->getLabel();
            bValue = b->getLabel();
            break;
        case PlotIdColumn:
            aValue = a->getPlotId();
            bValue = b->getPlotId();
            break;
        case OverlayTypeColumn:
            // Sort by user-friendly type name, not enum value
            aValue = getOverlayTypeName(a->type());
            bValue = getOverlayTypeName(b->type());
            break;
        case CommentColumn:
            aValue = a->getComment();
            bValue = b->getComment();
            break;
        default:
            return false; // No sorting for unknown columns
        }
        
        // Handle different data types for comparison
        if (aValue.metaType() == QMetaType::fromType<QString>() && bValue.metaType() == QMetaType::fromType<QString>()) {
            if (order == Qt::AscendingOrder) {
                return aValue.toString().compare(bValue.toString(), Qt::CaseInsensitive) < 0;
            } else {
                return aValue.toString().compare(bValue.toString(), Qt::CaseInsensitive) > 0;
            }
        } else if (aValue.metaType() == QMetaType::fromType<bool>() && bValue.metaType() == QMetaType::fromType<bool>()) {
            if (order == Qt::AscendingOrder) {
                return aValue.toBool() < bValue.toBool();
            } else {
                return aValue.toBool() > bValue.toBool();
            }
        } else {
            // For other types, convert to string and compare
            QString aStr = aValue.toString();
            QString bStr = bValue.toString();
            if (order == Qt::AscendingOrder) {
                return aStr.compare(bStr, Qt::CaseInsensitive) < 0;
            } else {
                return aStr.compare(bStr, Qt::CaseInsensitive) > 0;
            }
        }
    });
    
    emit layoutChanged();
}
