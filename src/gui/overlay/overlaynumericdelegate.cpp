#include "overlaynumericdelegate.h"
#include <QDoubleSpinBox>

OverlayNumericDelegate::OverlayNumericDelegate(QObject *parent)
    : QStyledItemDelegate(parent)
{
}

QWidget *OverlayNumericDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    Q_UNUSED(option)
    Q_UNUSED(index)

    QDoubleSpinBox *editor = new QDoubleSpinBox(parent);
    
    // Set range to allow very large values
    editor->setRange(-1e10, 1e10);
    
    // Set decimal places to 4 for high precision
    editor->setDecimals(4);
    
    // Set step size for convenient editing
    editor->setSingleStep(1.0);
    
    // Disable keyboard tracking to prevent real-time updates while typing
    editor->setKeyboardTracking(false);
    
    return editor;
}

void OverlayNumericDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    double value = index.model()->data(index, Qt::EditRole).toDouble();
    QDoubleSpinBox *spinBox = static_cast<QDoubleSpinBox*>(editor);
    spinBox->setValue(value);
}

void OverlayNumericDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
    QDoubleSpinBox *spinBox = static_cast<QDoubleSpinBox*>(editor);
    spinBox->interpretText();
    model->setData(index, spinBox->value(), Qt::EditRole);
}

void OverlayNumericDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    Q_UNUSED(index)
    editor->setGeometry(option.rect);
}