#include "overlaycheckboxdelegate.h"
#include <QCheckBox>
#include <QApplication>
#include <QPainter>
#include <QStyleOptionButton>
#include <QMouseEvent>

OverlayCheckBoxDelegate::OverlayCheckBoxDelegate(QObject *parent)
    : QStyledItemDelegate(parent)
{
}

QWidget *OverlayCheckBoxDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    Q_UNUSED(parent)
    Q_UNUSED(option)
    Q_UNUSED(index)

    // This is a paint + editorEvent toggle delegate: the checkbox is drawn by
    // paint() and toggled by editorEvent(). Returning an editor here would
    // spawn a real QCheckBox on top of the painted one whenever an edit
    // trigger fires (the column is ItemIsEditable), producing a stray
    // duplicate checkbox in the cell.
    return nullptr;
}

void OverlayCheckBoxDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    QCheckBox *checkBox = qobject_cast<QCheckBox*>(editor);
    if (checkBox) {
        bool value = index.model()->data(index, Qt::EditRole).toBool();
        checkBox->setChecked(value);
    }
}

void OverlayCheckBoxDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
    QCheckBox *checkBox = qobject_cast<QCheckBox*>(editor);
    if (checkBox) {
        model->setData(index, checkBox->isChecked(), Qt::EditRole);
    }
}

void OverlayCheckBoxDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    Q_UNUSED(index)
    editor->setGeometry(option.rect);
}

void OverlayCheckBoxDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    // Draw the checkbox
    QStyleOptionButton checkBoxOption;
    checkBoxOption.state = QStyle::State_Enabled;
    
    if (index.data(Qt::EditRole).toBool()) {
        checkBoxOption.state |= QStyle::State_On;
    } else {
        checkBoxOption.state |= QStyle::State_Off;
    }
    
    if (option.state & QStyle::State_Selected) {
        checkBoxOption.state |= QStyle::State_Selected;
    }
    
    // Center the checkbox in the cell
    QRect checkBoxRect = QApplication::style()->subElementRect(QStyle::SE_CheckBoxIndicator, &checkBoxOption);
    checkBoxOption.rect = QRect(option.rect.x() + (option.rect.width() - checkBoxRect.width()) / 2,
                               option.rect.y() + (option.rect.height() - checkBoxRect.height()) / 2,
                               checkBoxRect.width(), checkBoxRect.height());
    
    QApplication::style()->drawControl(QStyle::CE_CheckBox, &checkBoxOption, painter);
}

bool OverlayCheckBoxDelegate::editorEvent(QEvent *event, QAbstractItemModel *model, const QStyleOptionViewItem &option, const QModelIndex &index)
{
    if (event->type() == QEvent::MouseButtonPress || event->type() == QEvent::MouseButtonRelease) {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        if (mouseEvent->button() == Qt::LeftButton) {
            if (event->type() == QEvent::MouseButtonRelease) {
                // Toggle the checkbox value
                bool currentValue = index.data(Qt::EditRole).toBool();
                model->setData(index, !currentValue, Qt::EditRole);
            }
            return true;
        }
    }
    
    return QStyledItemDelegate::editorEvent(event, model, option, index);
}