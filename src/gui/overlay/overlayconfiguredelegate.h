#ifndef OVERLAYCONFIGUREDELEGATE_H
#define OVERLAYCONFIGUREDELEGATE_H

#include <QStyledItemDelegate>
#include <QStyleOptionButton>
#include <QApplication>
#include <QMouseEvent>

class OverlayConfigureDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    explicit OverlayConfigureDelegate(QObject *parent = nullptr);

    // QAbstractItemDelegate interface
    void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
    bool editorEvent(QEvent *event, QAbstractItemModel *model, const QStyleOptionViewItem &option, const QModelIndex &index) override;
    QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const override;

signals:
    void configureClicked(const QModelIndex &index);

private:
    QStyleOptionButton getButtonStyleOption(const QStyleOptionViewItem &option, const QModelIndex &index) const;
};

#endif // OVERLAYCONFIGUREDELEGATE_H