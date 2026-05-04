#include "overlayconfiguredelegate.h"
#include <QPainter>
#include <QApplication>
#include <QMouseEvent>

OverlayConfigureDelegate::OverlayConfigureDelegate(QObject *parent)
    : QStyledItemDelegate(parent)
{
}

void OverlayConfigureDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    QStyleOptionButton buttonOption = getButtonStyleOption(option, index);
    
    // Set the button text to the gear symbol
    buttonOption.text = index.data(Qt::DisplayRole).toString();
    
    // Paint the button
    QApplication::style()->drawControl(QStyle::CE_PushButton, &buttonOption, painter);
}

bool OverlayConfigureDelegate::editorEvent(QEvent *event, QAbstractItemModel *model, const QStyleOptionViewItem &option, const QModelIndex &index)
{
    Q_UNUSED(model)
    
    if (event->type() == QEvent::MouseButtonPress) {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        if (mouseEvent->button() == Qt::LeftButton) {
            // Check if click is within the button area
            if (option.rect.contains(mouseEvent->pos())) {
                emit configureClicked(index);
                return true;
            }
        }
    }
    
    return false;
}

QSize OverlayConfigureDelegate::sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    QStyleOptionButton buttonOption = getButtonStyleOption(option, index);
    return QApplication::style()->sizeFromContents(QStyle::CT_PushButton, &buttonOption, QSize(), nullptr);
}

QStyleOptionButton OverlayConfigureDelegate::getButtonStyleOption(const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    QStyleOptionButton buttonOption;
    buttonOption.rect = option.rect;
    buttonOption.text = index.data(Qt::DisplayRole).toString();
    buttonOption.state = QStyle::State_Enabled;
    
    // Add some margin to the button
    buttonOption.rect.adjust(2, 2, -2, -2);
    
    if (option.state & QStyle::State_MouseOver) {
        buttonOption.state |= QStyle::State_MouseOver;
    }
    
    if (option.state & QStyle::State_Selected) {
        buttonOption.state |= QStyle::State_HasFocus;
    }
    
    return buttonOption;
}